// state.h - the shared state every thread reads and writes (dream.py's `_state`).
#pragma once

#include <atomic>
#include <mutex>
#include <string>

class SharedState {
public:
    std::atomic<bool> running{true};
    std::atomic<bool> sleeping{false};
    std::atomic<bool> flirtPlayed{false};
    std::atomic<bool> sensorWake{false};    // mmWave PRESENT seen while sleeping
    std::atomic<bool> presenceSeen{false};  // mmWave PRESENT currently active - for the ABSENT farewell edge
    std::atomic<double> lastWakeTs{0};      // nowSec() of the last real interaction
    std::atomic<double> distanceM{-1};      // last mmWave range in meters, or -1

    // "idle", "listening", "thinking", "talking" or "sleeping" - picks the video pool.
    std::string state() const;
    void setState(const std::string& s);

    // Path of a clip that must play once, right now, ahead of the state's own
    // pool (the intro, a flirt clip). The display clears it when it finishes.
    std::string forceVideo() const;
    void setForceVideo(const std::string& path);

private:
    mutable std::mutex m_;
    std::string state_ = "idle";
    std::string force_;
};

SharedState& S();

// Call whenever the user actually interacts - resets all idle timers.
void touchInteraction();
