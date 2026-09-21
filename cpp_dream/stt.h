// stt.h - speech to text with whisper.cpp built into the exe (the C++ stand-in
// for the PyTorch Whisper that dream.py uses).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Loads the model (models/ggml-tiny.en.bin). Safe to call once at startup on a
// background thread; transcribe() waits for it.
bool sttLoad(const std::string& modelPath);
bool sttReady();

// Text of a mono 16-bit clip recorded at `rate` Hz, or "" if nothing was said.
std::string sttTranscribe(const std::vector<int16_t>& samples, int rate);
