#include "video.h"

#include "common.h"
#include "config.h"
#include "state.h"

#include <windows.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfreadwrite.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <random>

namespace fs = std::filesystem;

namespace {

template <class T>
void release(T*& p) {
    if (p) { p->Release(); p = nullptr; }
}

std::wstring widen(const std::string& s) {
    if (s.empty()) return L"";
    int n = MultiByteToWideChar(CP_ACP, 0, s.data(), int(s.size()), nullptr, 0);
    std::wstring w(size_t(n), L'\0');
    MultiByteToWideChar(CP_ACP, 0, s.data(), int(s.size()), w.data(), n);
    return w;
}

std::mt19937& rng() {
    static std::mt19937 r{std::random_device{}()};
    return r;
}

} // namespace

// ---------------------------------------------------------------- VideoPlayer

VideoPlayer::VideoPlayer() = default;

VideoPlayer::~VideoPlayer() { release(reader_); }

bool VideoPlayer::open(const std::string& path, bool loop) {
    path_ = path;
    loop_ = loop;

    IMFAttributes* attrs = nullptr;
    MFCreateAttributes(&attrs, 1);
    if (attrs) attrs->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE); // lets it convert to RGB32
    HRESULT hr = MFCreateSourceReaderFromURL(widen(path).c_str(), attrs, &reader_);
    release(attrs);
    if (FAILED(hr)) {
        dream::logf("Cannot open video: %s (0x%08lx)", path.c_str(), static_cast<unsigned long>(hr));
        return false;
    }

    // Only the video stream, decoded to 32-bit RGB.
    reader_->SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS, FALSE);
    reader_->SetStreamSelection(MF_SOURCE_READER_FIRST_VIDEO_STREAM, TRUE);
    IMFMediaType* want = nullptr;
    MFCreateMediaType(&want);
    want->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    want->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
    hr = reader_->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, want);
    release(want);
    if (FAILED(hr)) {
        dream::logf("Cannot decode video (no H.264 decoder?): %s (0x%08lx)", path.c_str(), static_cast<unsigned long>(hr));
        release(reader_);
        return false;
    }

    IMFMediaType* cur = nullptr;
    reader_->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, &cur);
    UINT32 w = 0, h = 0;
    MFGetAttributeSize(cur, MF_MT_FRAME_SIZE, &w, &h);
    stride_ = long(MFGetAttributeUINT32(cur, MF_MT_DEFAULT_STRIDE, 0));
    release(cur);
    w_ = int(w);
    h_ = int(h);
    if (w_ <= 0 || h_ <= 0) { release(reader_); return false; }
    if (stride_ == 0) stride_ = w_ * 4;

    haveNext_ = readNext();
    if (!haveNext_) { release(reader_); return false; }
    return true;
}

bool VideoPlayer::readNext() {
    for (int tries = 0; tries < 100; tries++) {
        DWORD flags = 0;
        LONGLONG ts = 0;
        IMFSample* sample = nullptr;
        HRESULT hr = reader_->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, nullptr, &flags, &ts, &sample);
        if (FAILED(hr)) return false;
        if (flags & MF_SOURCE_READERF_ENDOFSTREAM) { release(sample); return false; }
        if (!sample) continue; // a gap in the stream; ask again

        IMFMediaBuffer* buf = nullptr;
        if (SUCCEEDED(sample->ConvertToContiguousBuffer(&buf))) {
            BYTE* data = nullptr;
            DWORD maxLen = 0, curLen = 0;
            if (SUCCEEDED(buf->Lock(&data, &maxLen, &curLen))) {
                next_.w = w_;
                next_.h = h_;
                next_.px.resize(size_t(w_) * size_t(h_));
                long absStride = stride_ < 0 ? -stride_ : stride_;
                for (int y = 0; y < h_; y++) {
                    int srcRow = stride_ < 0 ? h_ - 1 - y : y; // bottom-up frames are flipped
                    if (size_t(srcRow + 1) * size_t(absStride) > curLen) break;
                    std::memcpy(&next_.px[size_t(y) * size_t(w_)], data + size_t(srcRow) * size_t(absStride), size_t(w_) * 4);
                }
                buf->Unlock();
            }
            release(buf);
        }
        release(sample);
        if (rebase_) { baseTs_ = ts; rebase_ = false; }
        nextTs_ = ts;
        return true;
    }
    return false;
}

bool VideoPlayer::restart() {
    PROPVARIANT pos;
    PropVariantInit(&pos);
    pos.vt = VT_I8;
    pos.hVal.QuadPart = 0;
    if (FAILED(reader_->SetCurrentPosition(GUID_NULL, pos))) return false;
    rebase_ = true;
    return readNext();
}

void VideoPlayer::update(double nowMs) {
    if (!reader_ || finished_) return;
    if (!started_) {
        std::swap(cur_, next_);
        started_ = true;
        startMs_ = nowMs;
        haveNext_ = readNext();
    }
    // Catch up to the frame due now (skipping any we're too slow to show).
    while (haveNext_ && (nowMs - startMs_) * 10000.0 >= double(nextTs_ - baseTs_)) {
        std::swap(cur_, next_);
        haveNext_ = readNext();
        if (!haveNext_) {
            if (loop_ && restart()) {
                haveNext_ = true;
                startMs_ = nowMs;
            } else {
                finished_ = true;
            }
        }
    }
}

// ---------------------------------------------------------------- pools

VideoPools buildVideoPools() {
    VideoPools pools;
    for (const char* k : {"idle", "listening", "thinking", "talking", "flirtytalk", "sleeping"}) pools[k];

    std::error_code ec;
    if (!fs::is_directory(cfg().videosDir, ec)) return pools;
    std::vector<std::string> files;
    for (const auto& e : fs::directory_iterator(cfg().videosDir, ec)) {
        if (e.path().extension() == ".mp4") files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());

    for (const auto& f : files) {
        std::string name = fs::path(f).filename().string();
        if (name == "sleeping.mp4") { pools["sleeping"].push_back(f); continue; }
        for (const char* prefix : {"idle", "listening", "thinking", "talking", "flirtytalk"}) {
            if (name.rfind(prefix, 0) == 0) { pools[prefix].push_back(f); break; }
        }
    }
    for (const auto& [k, v] : pools) dream::logf("  %s: %zu video(s)", k.c_str(), v.size());
    return pools;
}

// ---------------------------------------------------------------- VideoStateManager

std::string VideoStateManager::pickRandom(const std::vector<std::string>& pool, const std::string& avoid) {
    if (pool.empty()) return "";
    if (pool.size() == 1) return pool[0];
    std::vector<std::string> choices;
    for (const auto& p : pool) {
        if (p != avoid) choices.push_back(p);
    }
    const auto& from = choices.empty() ? pool : choices;
    return from[std::uniform_int_distribution<size_t>(0, from.size() - 1)(rng())];
}

void VideoStateManager::load(const std::string& path) {
    auto p = std::make_unique<VideoPlayer>();
    if (p->open(path, true)) {
        player_ = std::move(p);
        currentPath_ = path;
    } else {
        player_.reset();
        currentPath_.clear();
    }
}

const Frame* VideoStateManager::getFrame(double nowMs, const std::string& state) {
    // A forced clip (the intro, a flirt clip) takes priority and plays once.
    std::string force = S().forceVideo();
    if (!force.empty() && !forcePlayer_) {
        auto p = std::make_unique<VideoPlayer>();
        if (p->open(force, false)) {
            dream::logf("Playing %s", fs::path(force).filename().string().c_str());
            forcePlayer_ = std::move(p);
        } else {
            S().setForceVideo("");
        }
    }
    if (forcePlayer_) {
        forcePlayer_->update(nowMs);
        if (forcePlayer_->finished()) {
            forcePlayer_.reset();
            S().setForceVideo("");
        } else if (forcePlayer_->hasFrame()) {
            return &forcePlayer_->frame();
        }
    }

    std::vector<std::string> pool = pools_[state];
    if (pool.empty()) pool = pools_["idle"];

    if (state != currentState_) {
        std::string avoid = state == "idle" ? prevIdlePath_ : "";
        std::string path = pickRandom(pool, avoid);
        if (state == "idle") prevIdlePath_ = path;
        currentState_ = state;
        if (!path.empty()) load(path);
        else player_.reset();
    }

    if (!player_) return nullptr;
    player_->update(nowMs);
    return player_->hasFrame() ? &player_->frame() : nullptr;
}
