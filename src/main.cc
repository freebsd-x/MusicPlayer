// MusicPlayer.xapp — audio player using FFmpeg decode + OSS /dev/dsp output.
//
// /dev/dsp is opened before XApplication::Run() (before cap_enter()), so the
// background decode thread can write audio inside the Capsicum sandbox.
//
// Architecture:
//   PlaybackEngine  — shared state (atomics); owns the decode/write thread.
//   Main thread     — xkit UI + XBus callbacks (StateChanged / MediaKey).
//   Playback thread — FFmpeg decode loop; writes PCM to dsp_fd; sends
//                     ReportState to compositor once per second and on track end.

#include <xkit/application.h>
#include <xkit/widget.h>
#include <xkit/xbus_client.h>
#include <shared/protocol.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
}

#include <sys/soundcard.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace fbx::xkit;
using namespace fbx::xrender;
using namespace fbx::shared;

// ── Playback engine state (shared between main thread and playback thread) ────

struct PlaybackEngine {
    std::atomic<bool>     playing{false};
    std::atomic<bool>     quit{false};
    std::atomic<bool>     at_end{false};
    std::atomic<bool>     seek_requested{false};
    std::atomic<uint32_t> seek_target_ms{0};
    std::atomic<uint32_t> position_ms{0};
    std::atomic<uint32_t> duration_ms{0};
    std::atomic<uint32_t> volume{80};
    std::atomic<int>      skip{0};
    std::mutex            path_mtx;
    std::string           current_path;
};

// ── Track metadata ────────────────────────────────────────────────────────────

struct TrackMeta {
    std::string path;
    std::string title;
    std::string artist;
    std::string album;
    uint32_t    duration_ms{0};
};

static TrackMeta extractMeta(const std::string& path) noexcept {
    TrackMeta m;
    m.path  = path;
    m.title = path.substr(path.rfind('/') + 1);

    AVFormatContext* fc = nullptr;
    if (avformat_open_input(&fc, path.c_str(), nullptr, nullptr) != 0) return m;
    avformat_find_stream_info(fc, nullptr);

    if (fc->duration != AV_NOPTS_VALUE && fc->duration > 0)
        m.duration_ms = static_cast<uint32_t>(fc->duration / 1000);

    auto tag = [&](const char* key) -> std::string {
        AVDictionaryEntry* e = av_dict_get(fc->metadata, key, nullptr, AV_DICT_IGNORE_SUFFIX);
        return e ? std::string(e->value) : std::string{};
    };
    if (auto t = tag("title");  !t.empty()) m.title  = t;
    m.artist = tag("artist");
    m.album  = tag("album");

    avformat_close_input(&fc);
    return m;
}

// ── Formatting helpers ────────────────────────────────────────────────────────

static std::string fmtDuration(uint32_t ms) noexcept {
    uint32_t s   = ms / 1000;
    uint32_t min = s / 60;
    char buf[12];
    std::snprintf(buf, sizeof(buf), "%u:%02u", min, s % 60);
    return buf;
}

static std::string truncate(const std::string& s, size_t max) noexcept {
    if (s.size() <= max) return s;
    return s.substr(0, max - 1) + "…";
}

// ── Send state to compositor ──────────────────────────────────────────────────

static void reportState(XBusClient& xbus, const PlaybackEngine& eng,
                        const TrackMeta& track) noexcept {
    MediaPlayerState s{};
    std::strncpy(s.title_,  track.title.c_str(),  sizeof(s.title_)  - 1);
    std::strncpy(s.artist_, track.artist.c_str(), sizeof(s.artist_) - 1);
    std::strncpy(s.album_,  track.album.c_str(),  sizeof(s.album_)  - 1);
    s.position_ms_ = eng.position_ms.load();
    s.duration_ms_ = eng.duration_ms.load();
    s.volume_      = eng.volume.load();
    s.is_playing_  = eng.playing.load() ? 1 : 0;
    s.shuffle_     = 0;
    s.repeat_      = 0;
    xbus.Call("com.freebsdx.MediaPlayer",
        static_cast<uint32_t>(MediaPlayerOpcode::ReportState),
        {reinterpret_cast<const uint8_t*>(&s), sizeof(s)},
        nullptr);
}

// ── Playback thread ───────────────────────────────────────────────────────────

static void playbackThread(int dsp_fd, std::shared_ptr<PlaybackEngine> eng,
                           std::shared_ptr<std::vector<TrackMeta>> playlist,
                           std::shared_ptr<std::atomic<int>> cur_idx,
                           XBusClient* xbus_ptr) {
    while (!eng->quit.load()) {
        if (!eng->playing.load()) { usleep(10'000); continue; }

        std::string path;
        int idx = cur_idx->load();
        if (idx < 0 || idx >= static_cast<int>(playlist->size())) {
            eng->playing = false;
            continue;
        }
        {
            std::lock_guard<std::mutex> lk(eng->path_mtx);
            path = eng->current_path;
        }
        if (path.empty()) { eng->playing = false; continue; }

        AVFormatContext* fc = nullptr;
        if (avformat_open_input(&fc, path.c_str(), nullptr, nullptr) != 0) {
            eng->playing = false; eng->at_end = true; continue;
        }
        avformat_find_stream_info(fc, nullptr);

        const AVCodec* codec = nullptr;
        int asi = av_find_best_stream(fc, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
        if (asi < 0) { avformat_close_input(&fc); eng->playing = false; continue; }

        AVCodecContext* cc = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(cc, fc->streams[asi]->codecpar);
        avcodec_open2(cc, codec, nullptr);

        SwrContext* swr = swr_alloc();
        AVChannelLayout stereo = AV_CHANNEL_LAYOUT_STEREO;
        av_opt_set_chlayout  (swr, "in_chlayout",    &cc->ch_layout, 0);
        av_opt_set_chlayout  (swr, "out_chlayout",   &stereo,        0);
        av_opt_set_int       (swr, "in_sample_rate",  cc->sample_rate, 0);
        av_opt_set_int       (swr, "out_sample_rate", 44100,           0);
        av_opt_set_sample_fmt(swr, "in_sample_fmt",   cc->sample_fmt,  0);
        av_opt_set_sample_fmt(swr, "out_sample_fmt",  AV_SAMPLE_FMT_S16, 0);
        swr_init(swr);

        if (dsp_fd >= 0) {
            int fmt = AFMT_S16_LE, ch = 2, rate = 44100;
            ioctl(dsp_fd, SNDCTL_DSP_SETFMT,   &fmt);
            ioctl(dsp_fd, SNDCTL_DSP_CHANNELS, &ch);
            ioctl(dsp_fd, SNDCTL_DSP_SPEED,    &rate);
        }

        AVPacket* pkt   = av_packet_alloc();
        AVFrame*  frame = av_frame_alloc();
        std::vector<int16_t> pcm;

        time_t last_report = std::time(nullptr);
        bool done = false;

        while (!done) {
            if (eng->quit.load())           { done = true; break; }
            if (!eng->playing.load())       { usleep(5'000); continue; }
            if (eng->skip.load() != 0)      { done = true; break; }

            if (eng->seek_requested.exchange(false)) {
                uint32_t target = eng->seek_target_ms.load();
                const AVRational& tb = fc->streams[asi]->time_base;
                int64_t ts = av_rescale_q(
                    static_cast<int64_t>(target),
                    {1, 1000}, tb);
                av_seek_frame(fc, asi, ts, AVSEEK_FLAG_BACKWARD);
                avcodec_flush_buffers(cc);
                swr_init(swr);
            }

            int ret = av_read_frame(fc, pkt);
            if (ret < 0) { eng->at_end = true; done = true; break; }
            if (pkt->stream_index != asi) { av_packet_unref(pkt); continue; }

            if (pkt->pts != AV_NOPTS_VALUE) {
                double pts_s = av_q2d(fc->streams[asi]->time_base) *
                               static_cast<double>(pkt->pts);
                eng->position_ms = static_cast<uint32_t>(pts_s * 1000.0);
            }

            avcodec_send_packet(cc, pkt);
            while (avcodec_receive_frame(cc, frame) == 0) {
                int out_samples = swr_get_out_samples(swr, frame->nb_samples);
                if (out_samples <= 0) continue;
                pcm.resize(static_cast<size_t>(out_samples) * 2);
                uint8_t* out_ptr = reinterpret_cast<uint8_t*>(pcm.data());
                int converted = swr_convert(swr, &out_ptr, out_samples,
                    const_cast<const uint8_t**>(frame->data), frame->nb_samples);
                if (converted > 0 && dsp_fd >= 0) {
                    uint32_t vol = eng->volume.load();
                    for (int i = 0; i < converted * 2; ++i)
                        pcm[i] = static_cast<int16_t>(pcm[i] * vol / 100);
                    write(dsp_fd, pcm.data(),
                          static_cast<size_t>(converted) * 2 * sizeof(int16_t));
                }
            }
            av_packet_unref(pkt);

            time_t now = std::time(nullptr);
            if (now - last_report >= 1 && xbus_ptr) {
                last_report = now;
                idx = cur_idx->load();
                if (idx >= 0 && idx < static_cast<int>(playlist->size()))
                    reportState(*xbus_ptr, *eng, (*playlist)[static_cast<size_t>(idx)]);
            }
        }

        av_frame_free(&frame);
        av_packet_free(&pkt);
        swr_free(&swr);
        avcodec_free_context(&cc);
        avformat_close_input(&fc);

        idx = cur_idx->load();
        if (xbus_ptr && idx >= 0 && idx < static_cast<int>(playlist->size()))
            reportState(*xbus_ptr, *eng, (*playlist)[static_cast<size_t>(idx)]);
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    int dsp_fd = open("/dev/dsp", O_WRONLY);
    if (dsp_fd < 0)
        std::fprintf(stderr, "[MusicPlayer] /dev/dsp unavailable — audio disabled\n");

    // ── Shared state ──────────────────────────────────────────────────────────
    auto playlist  = std::make_shared<std::vector<TrackMeta>>();
    auto cur_idx   = std::make_shared<std::atomic<int>>(-1);
    auto eng       = std::make_shared<PlaybackEngine>();

    auto rebuild_list        = std::make_shared<std::function<void()>>();
    auto update_now_playing  = std::make_shared<std::function<void()>>();
    auto play_track          = std::make_shared<std::function<void(int)>>();

    // Deferred callbacks for buttons that need xbus
    auto on_play = std::make_shared<std::function<void()>>();
    auto on_next = std::make_shared<std::function<void()>>();
    auto on_prev = std::make_shared<std::function<void()>>();
    auto on_fwd  = std::make_shared<std::function<void()>>();
    auto on_rew  = std::make_shared<std::function<void()>>();
    auto on_add  = std::make_shared<std::function<void()>>();

    // ── Left sidebar ──────────────────────────────────────────────────────────
    auto sidebar_list = std::make_shared<FlexColumn>();
    sidebar_list->setGap(1.0f);

    auto sidebar_scroll = std::make_shared<ScrollViewWidget>(sidebar_list);
    sidebar_scroll->setFlexGrow(1.0f);

    auto queue_info = std::make_shared<TextWidget>("No tracks", 10.0f);
    queue_info->setColor(Color{0.45f, 0.47f, 0.55f, 1.0f});
    queue_info->setPadding(8.0f);

    auto add_btn = std::make_shared<ButtonWidget>("+ Add Files",
        [on_add]() { if (*on_add) (*on_add)(); });
    add_btn->setFontSize(12.0f);
    add_btn->setPadding(10.0f);
    add_btn->setBackgroundColor(Color{0.18f, 0.20f, 0.30f, 1.0f});
    add_btn->setColor(Color{0.85f, 0.88f, 1.00f, 1.0f});
    add_btn->setMarginAll(8.0f);

    auto sidebar_header = std::make_shared<TextWidget>("Music", 16.0f);
    sidebar_header->setFontWeight(600);
    sidebar_header->setColor(Color{0.90f, 0.92f, 1.00f, 1.0f});
    sidebar_header->setPadding(12.0f);

    auto sidebar_col = std::make_shared<FlexColumn>();
    sidebar_col->addChild(sidebar_header);
    sidebar_col->addChild(add_btn);
    sidebar_col->addChild(sidebar_scroll);
    sidebar_col->addChild(queue_info);
    sidebar_col->setFixedSize(220.0f, 0.0f);
    sidebar_col->setBackgroundColor(Color{0.10f, 0.11f, 0.16f, 1.0f});

    // ── Now-playing area ──────────────────────────────────────────────────────
    auto art_placeholder = std::make_shared<TextWidget>("♪", 48.0f);
    art_placeholder->setColor(Color{0.35f, 0.38f, 0.55f, 1.0f});

    auto art_box = std::make_shared<FlexRow>();
    art_box->setFixedSize(160.0f, 160.0f);
    art_box->setBackgroundColor(Color{0.18f, 0.20f, 0.30f, 1.0f});
    art_box->setMainAxisAlignment(MainAxisAlignment::Center);
    art_box->setCrossAxisAlignment(CrossAxisAlignment::Center);
    art_box->addChild(art_placeholder);

    auto track_title = std::make_shared<TextWidget>("No Track", 18.0f);
    track_title->setFontWeight(600);
    track_title->setColor(Color{0.92f, 0.93f, 1.00f, 1.0f});

    auto track_sub = std::make_shared<TextWidget>("", 12.0f);
    track_sub->setColor(Color{0.55f, 0.58f, 0.70f, 1.0f});

    auto time_label = std::make_shared<TextWidget>("0:00 / 0:00", 11.0f);
    time_label->setColor(Color{0.50f, 0.52f, 0.65f, 1.0f});

    auto seek_slider = std::make_shared<SliderWidget>(0.0f, 0.0f, 1000.0f,
        [eng](float v) {
            eng->seek_target_ms  = static_cast<uint32_t>(v);
            eng->seek_requested  = true;
        });
    seek_slider->setFixedSize(340.0f, 0.0f);

    // ── Transport buttons ─────────────────────────────────────────────────────
    auto play_lbl = std::make_shared<TextWidget>("▶", 18.0f);
    play_lbl->setColor(Color{0.88f, 0.90f, 1.00f, 1.0f});

    auto mk_btn = [](const char* label, std::shared_ptr<std::function<void()>> cb) {
        auto b = std::make_shared<ButtonWidget>(label, [cb]() { if (*cb) (*cb)(); });
        b->setFontSize(18.0f);
        b->setPadding(10.0f);
        b->setBackgroundColor(Color{0.18f, 0.20f, 0.30f, 1.0f});
        b->setColor(Color{0.88f, 0.90f, 1.00f, 1.0f});
        b->setFixedSize(44.0f, 44.0f);
        return b;
    };

    auto prev_btn = mk_btn("|◀", on_prev);
    auto rew_btn  = mk_btn("◀◀", on_rew);
    auto fwd_btn  = mk_btn("▶▶", on_fwd);
    auto next_btn = mk_btn("▶|", on_next);

    auto play_btn = std::make_shared<ButtonWidget>(play_lbl,
        [on_play]() { if (*on_play) (*on_play)(); });
    play_btn->setFontSize(18.0f);
    play_btn->setPadding(10.0f);
    play_btn->setBackgroundColor(Color{0.18f, 0.20f, 0.30f, 1.0f});
    play_btn->setFixedSize(44.0f, 44.0f);

    auto transport_row = std::make_shared<FlexRow>();
    transport_row->setGap(8.0f);
    transport_row->setMainAxisAlignment(MainAxisAlignment::Center);
    transport_row->setCrossAxisAlignment(CrossAxisAlignment::Center);
    transport_row->addChild(prev_btn);
    transport_row->addChild(rew_btn);
    transport_row->addChild(play_btn);
    transport_row->addChild(fwd_btn);
    transport_row->addChild(next_btn);

    // ── Volume ────────────────────────────────────────────────────────────────
    auto vol_lbl = std::make_shared<TextWidget>("Vol 80%", 11.0f);
    vol_lbl->setColor(Color{0.55f, 0.58f, 0.70f, 1.0f});
    vol_lbl->setFixedSize(55.0f, 0.0f);

    auto vol_slider = std::make_shared<SliderWidget>(80.0f, 0.0f, 100.0f,
        [eng, vol_lbl](float v) {
            eng->volume = static_cast<uint32_t>(v);
            char buf[16];
            std::snprintf(buf, sizeof(buf), "Vol %.0f%%", v);
            vol_lbl->setText(buf);
        });
    vol_slider->setFixedSize(200.0f, 0.0f);

    auto vol_row = std::make_shared<FlexRow>();
    vol_row->setGap(8.0f);
    vol_row->setCrossAxisAlignment(CrossAxisAlignment::Center);
    vol_row->setMainAxisAlignment(MainAxisAlignment::Center);
    vol_row->addChild(vol_lbl);
    vol_row->addChild(vol_slider);

    auto np_col = std::make_shared<FlexColumn>();
    np_col->setPadding(24.0f);
    np_col->setGap(12.0f);
    np_col->setCrossAxisAlignment(CrossAxisAlignment::Center);
    np_col->addChild(art_box);
    np_col->addChild(track_title);
    np_col->addChild(track_sub);
    np_col->addChild(seek_slider);
    np_col->addChild(time_label);
    np_col->addChild(transport_row);
    np_col->addChild(vol_row);
    np_col->setFlexGrow(1.0f);

    // ── Root layout ───────────────────────────────────────────────────────────
    auto root_row = std::make_shared<FlexRow>();
    root_row->addChild(sidebar_col);
    root_row->addChild(np_col);

    auto root = std::make_shared<ContainerWidget>(root_row);
    root->setBackgroundColor(Color{0.12f, 0.13f, 0.19f, 1.0f});

    int saved_argc = argc;
    char** saved_argv = argv;

    return XApplication::Run(root, "/tmp/freebsdx.sock", false, false,
        [=](XBusClient& xbus) mutable {
            XBusClient* xbus_ptr = &xbus;
            XApplication::setTitle("Music Player");

            // ── rebuild_list ─────────────────────────────────────────────────
            *rebuild_list = [=]() {
                sidebar_list->clearChildren();
                for (int i = 0; i < static_cast<int>(playlist->size()); ++i) {
                    const auto& t = (*playlist)[static_cast<size_t>(i)];
                    bool active = (i == cur_idx->load());

                    auto num = std::make_shared<TextWidget>(std::to_string(i + 1) + ".", 10.0f);
                    num->setColor(Color{0.40f, 0.42f, 0.55f, 1.0f});
                    num->setFixedSize(24.0f, 0.0f);

                    auto t_lbl = std::make_shared<TextWidget>(truncate(t.title, 26), 12.0f);
                    t_lbl->setColor(Color{0.88f, 0.90f, 1.00f, 1.0f});

                    auto a_lbl = std::make_shared<TextWidget>(truncate(t.artist, 26), 10.0f);
                    a_lbl->setColor(Color{0.50f, 0.52f, 0.65f, 1.0f});

                    auto info_col = std::make_shared<FlexColumn>();
                    info_col->setGap(2.0f);
                    info_col->setFlexGrow(1.0f);
                    info_col->addChild(t_lbl);
                    if (!t.artist.empty()) info_col->addChild(a_lbl);

                    auto dur = std::make_shared<TextWidget>(fmtDuration(t.duration_ms), 10.0f);
                    dur->setColor(Color{0.40f, 0.42f, 0.55f, 1.0f});

                    auto info_row = std::make_shared<FlexRow>();
                    info_row->setPadding(8.0f);
                    info_row->setGap(8.0f);
                    info_row->setCrossAxisAlignment(CrossAxisAlignment::Center);
                    info_row->addChild(num);
                    info_row->addChild(info_col);
                    info_row->addChild(dur);

                    auto row_btn = std::make_shared<ButtonWidget>(info_row,
                        [i, play_track]() { if (*play_track) (*play_track)(i); });
                    row_btn->setBackgroundColor(active
                        ? Color{0.22f, 0.25f, 0.40f, 1.0f}
                        : Color{0.12f, 0.13f, 0.19f, 0.0f});
                    sidebar_list->addChild(row_btn);
                }

                if (playlist->empty()) {
                    queue_info->setText("No tracks");
                } else {
                    uint32_t total_ms = 0;
                    for (const auto& t : *playlist) total_ms += t.duration_ms;
                    char buf[64];
                    std::snprintf(buf, sizeof(buf), "%zu track%s · %s",
                        playlist->size(), playlist->size() == 1 ? "" : "s",
                        fmtDuration(total_ms).c_str());
                    queue_info->setText(buf);
                }
            };

            // ── update_now_playing ───────────────────────────────────────────
            *update_now_playing = [=]() {
                int idx = cur_idx->load();
                if (idx < 0 || idx >= static_cast<int>(playlist->size())) {
                    track_title->setText("No Track");
                    track_sub->setText("");
                    play_lbl->setText("▶");
                    time_label->setText("0:00 / 0:00");
                    return;
                }
                const auto& t = (*playlist)[static_cast<size_t>(idx)];
                track_title->setText(truncate(t.title, 32));

                std::string sub = t.artist;
                if (!t.album.empty()) sub += (sub.empty() ? "" : " — ") + t.album;
                track_sub->setText(truncate(sub, 40));

                play_lbl->setText(eng->playing.load() ? "⏸" : "▶");

                uint32_t pos = eng->position_ms.load();
                uint32_t dur = eng->duration_ms.load();
                time_label->setText(fmtDuration(pos) + " / " + fmtDuration(dur));

                if (dur > 0)
                    seek_slider->setValue(static_cast<float>(pos));
            };

            // ── play_track ───────────────────────────────────────────────────
            *play_track = [=](int idx) {
                if (idx < 0 || idx >= static_cast<int>(playlist->size())) return;
                cur_idx->store(idx);
                const auto& t = (*playlist)[static_cast<size_t>(idx)];
                {
                    std::lock_guard<std::mutex> lk(eng->path_mtx);
                    eng->current_path = t.path;
                }
                eng->duration_ms  = t.duration_ms;
                eng->position_ms  = 0;
                eng->at_end       = false;
                eng->skip         = 0;
                eng->playing      = true;

                if (*update_now_playing) (*update_now_playing)();
                if (*rebuild_list)       (*rebuild_list)();

                reportState(*xbus_ptr, *eng, t);
            };

            // ── Transport callbacks ──────────────────────────────────────────
            *on_play = [=]() {
                eng->playing = !eng->playing.load();
                if (*update_now_playing) (*update_now_playing)();
                int idx = cur_idx->load();
                if (idx >= 0 && idx < static_cast<int>(playlist->size()))
                    reportState(*xbus_ptr, *eng, (*playlist)[static_cast<size_t>(idx)]);
            };

            *on_next = [=]() {
                int next = cur_idx->load() + 1;
                if (next < static_cast<int>(playlist->size()) && *play_track)
                    (*play_track)(next);
            };

            *on_prev = [=]() {
                int prev = cur_idx->load() - 1;
                if (prev >= 0 && *play_track) (*play_track)(prev);
            };

            *on_fwd = [=]() {
                uint32_t pos = std::min(eng->position_ms.load() + 10'000u,
                                        eng->duration_ms.load());
                eng->seek_target_ms = pos;
                eng->seek_requested = true;
            };

            *on_rew = [=]() {
                uint32_t pos = eng->position_ms.load();
                eng->seek_target_ms = pos > 10'000u ? pos - 10'000u : 0u;
                eng->seek_requested = true;
            };

            *on_add = []() {
                // File dialog not available in sandbox; tracks loaded via argv
            };

            // ── Start playback thread ────────────────────────────────────────
            std::thread([eng, dsp_fd, playlist, cur_idx, xbus_ptr]() {
                playbackThread(dsp_fd, eng, playlist, cur_idx, xbus_ptr);
            }).detach();

            // ── Timer thread ─────────────────────────────────────────────────
            std::thread([eng, cur_idx, playlist, update_now_playing,
                         rebuild_list, play_track]() {
                while (!eng->quit.load()) {
                    usleep(500'000);
                    if (*update_now_playing) (*update_now_playing)();

                    if (eng->at_end.exchange(false)) {
                        int next = cur_idx->load() + 1;
                        if (next < static_cast<int>(playlist->size())) {
                            if (*play_track) (*play_track)(next);
                        } else {
                            eng->playing = false;
                            if (*update_now_playing) (*update_now_playing)();
                            if (*rebuild_list)       (*rebuild_list)();
                        }
                    }
                }
            }).detach();

            // ── XBus broadcast handler ───────────────────────────────────────
            xbus.OnBroadcast([=](const XBusMessage& msg) {
                using MO = MediaPlayerOpcode;
                if (std::strcmp(msg.service_name_, "com.freebsdx.MediaPlayer") != 0) return;

                if (msg.opcode_ == static_cast<uint32_t>(MO::StateChanged)) {
                    const auto* s = reinterpret_cast<const MediaPlayerState*>(msg.payload_);
                    eng->position_ms = s->position_ms_;
                    if (*update_now_playing) (*update_now_playing)();
                    return;
                }

                if (msg.opcode_ == static_cast<uint32_t>(MO::MediaKey)) {
                    const auto* k = reinterpret_cast<const MediaPlayerKeyRequest*>(msg.payload_);
                    switch (k->key_) {
                    case 0:
                        eng->playing = !eng->playing.load();
                        if (*update_now_playing) (*update_now_playing)();
                        break;
                    case 1:
                        eng->playing = false;
                        if (*update_now_playing) (*update_now_playing)();
                        break;
                    case 2: {
                        int n = cur_idx->load() + 1;
                        if (n < static_cast<int>(playlist->size()) && *play_track)
                            (*play_track)(n);
                        break;
                    }
                    case 3: {
                        int p = cur_idx->load() - 1;
                        if (p >= 0 && *play_track) (*play_track)(p);
                        break;
                    }
                    }
                    return;
                }
            });

            // ── Load files from command line ─────────────────────────────────
            for (int i = 1; i < saved_argc; ++i)
                playlist->push_back(extractMeta(saved_argv[i]));
            if (*rebuild_list) (*rebuild_list)();
            if (!playlist->empty() && *play_track) (*play_track)(0);
        });

    eng->quit = true;
    if (dsp_fd >= 0) close(dsp_fd);
    return 0;
}
