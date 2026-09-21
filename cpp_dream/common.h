// common.h - small helpers shared by every part of dream.exe.
#pragma once

#include <string>
#include <vector>

namespace dream {

// The folder holding config.json and scripts/ (found by walking up from the exe,
// so it works from cpp_dream/build/ as well as from the repo root).
std::string repoRoot();

std::string toLower(std::string s);
std::string trim(const std::string& s);
bool containsAny(const std::string& haystack, const std::vector<std::string>& needles);
bool fileExists(const std::string& path);
bool dirExists(const std::string& path);

double nowMs();      // monotonic, milliseconds
double nowSec();     // monotonic, seconds
void sleepMs(int ms);

// printf-style, timestamped, thread-safe.
void logf(const char* fmt, ...);

// A fresh path inside the temp folder, e.g. tempPath("tts", ".wav").
std::string tempPath(const char* tag, const char* ext);
std::string tempDir();

} // namespace dream
