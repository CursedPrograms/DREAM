// audio.h - microphone capture and WAV playback (Windows waveIn / PlaySound).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct MicClip {
    std::vector<int16_t> samples;  // mono
    int rate = 16000;
    bool ok = false;               // false if the mic couldn't be opened
    bool heardSpeech = false;      // recordCommand(): speech was detected
};

// Number of recording devices Windows reports.
int micCount();

// Records exactly `seconds` (one wake-word clip).
MicClip recordFixed(double seconds);

// Records a spoken command: waits for speech, then stops after `silenceSec` of
// quiet, or at `maxSeconds`, or after `noSpeechSec` with no speech at all.
// `stopFlag` (optional) ends it early, e.g. when the program is closing.
MicClip recordCommand(double maxSeconds, double silenceSec = 1.5, double noSpeechSec = 6.0);

// Test hooks: use a WAV file as the microphone (played back in real time, then
// silence), and skip the speakers.
bool setMicFile(const std::string& path);
void setMuted(bool muted);

// Plays a WAV file and waits until it's done. Returns false if it couldn't play.
bool playWavFile(const std::string& path);

// Cuts off whatever playWavFile() is playing (from another thread).
void stopPlayback();
