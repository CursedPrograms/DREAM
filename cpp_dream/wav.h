// wav.h - reading and writing 16-bit PCM WAV files, and the 16 kHz mono float
// form Whisper wants.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Wav {
    int sampleRate = 0;
    std::vector<int16_t> samples; // mono
};

// Reads a 16-bit PCM WAV (any rate; multi-channel is mixed down to mono).
bool readWav(const std::string& path, Wav& out);

bool writeWav(const std::string& path, const std::vector<int16_t>& samples, int sampleRate);

// Mono int16 at `rate` -> mono float in [-1, 1] at 16 kHz (linear resample).
std::vector<float> toWhisperInput(const std::vector<int16_t>& samples, int rate);

// Peak absolute sample (0..32768).
int peakLevel(const std::vector<int16_t>& samples);
