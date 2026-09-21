// video.h - avatar video playback: decoding MP4 clips with Windows Media
// Foundation, and choosing which clip to show (dream.py's VideoPlayer and
// VideoStateManager).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct Frame {
    int w = 0, h = 0;
    std::vector<uint32_t> px; // BGRA (alpha unused), top-down
};

struct IMFSourceReader;

class VideoPlayer {
public:
    VideoPlayer();
    ~VideoPlayer();
    VideoPlayer(const VideoPlayer&) = delete;
    VideoPlayer& operator=(const VideoPlayer&) = delete;

    bool open(const std::string& path, bool loop);
    // Moves to the frame that's due at `nowMs` (monotonic clock).
    void update(double nowMs);
    const Frame& frame() const { return cur_; }
    bool hasFrame() const { return cur_.w > 0; }
    bool finished() const { return finished_; }
    const std::string& path() const { return path_; }

private:
    bool readNext();           // decodes the next frame into next_; false at end of stream
    bool restart();            // seeks back to the start (looping)

    IMFSourceReader* reader_ = nullptr;
    std::string path_;
    bool loop_ = true;
    bool finished_ = false;
    bool started_ = false;
    bool rebase_ = true;       // the next decoded frame defines time zero
    int w_ = 0, h_ = 0;
    long stride_ = 0;
    Frame cur_, next_;
    bool haveNext_ = false;
    long long nextTs_ = 0, baseTs_ = 0; // 100 ns units
    double startMs_ = 0;
};

// Clips grouped by what they show, from videos/ (same rules as dream.py).
using VideoPools = std::map<std::string, std::vector<std::string>>;
VideoPools buildVideoPools(); // idle, listening, thinking, talking, flirtytalk, sleeping

// Picks and plays the right clip for the current state.
class VideoStateManager {
public:
    explicit VideoStateManager(const VideoPools& pools) : pools_(pools) {}
    // The frame to show now, or nullptr for a blank screen.
    const Frame* getFrame(double nowMs, const std::string& state);

private:
    std::string pickRandom(const std::vector<std::string>& pool, const std::string& avoid = "");
    void load(const std::string& path);

    VideoPools pools_;
    std::unique_ptr<VideoPlayer> player_, forcePlayer_;
    std::string currentPath_, currentState_, prevIdlePath_;
};
