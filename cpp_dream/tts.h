// tts.h - text to speech through Piper (piper.exe), like dream.py's speak().
#pragma once

#include <string>

// Voices `text` into a WAV file. Returns false if Piper isn't available or failed.
bool ttsSynthesize(const std::string& text, const std::string& wavPath);
