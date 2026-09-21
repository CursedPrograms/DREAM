// config.h - paths, settings from config.json, and the phrases DREAM reacts to.
// The phrase lists and timings mirror scripts/dream.py.
#pragma once

#include <string>
#include <vector>

struct Config {
    std::string root;          // repo root (config.json + scripts/)
    std::string charName;      // "DREAM"
    std::string systemPrompt;  // config.json's prompt with {name} filled in

    std::string ollamaHost = "localhost";
    int ollamaPort = 11434;
    std::string model = "phi3:mini";

    std::string videosDir;     // <root>/videos
    std::string voicesDir;     // <root>/voices
    std::string memoriesDir;   // <root>/memories
    std::string audioDir;      // <root>/audio (intro1.wav)
    std::string piperExe;      // piper.exe (from the venv, or next to dream.exe)
    std::string voiceModel;    // first .onnx in voices/
    std::string whisperModel;  // ggml-tiny.en.bin
};

// Loaded on first use.
const Config& cfg();
// Lets the self-test keep its memories out of the real memories/ folder.
void setMemoriesDir(const std::string& dir);

// ---- Timings (seconds) ----
inline double FLIRT_IDLE_TIMEOUT = 600;  // 10 minutes - flirt attention grab   (--flirt-after)
inline double SLEEP_IDLE_TIMEOUT = 900;  // 15 minutes - fall asleep              (--sleep-after)
inline constexpr double WAKE_SECONDS = 3;          // one wake-word listening clip
inline constexpr double RECORD_SECONDS = 16;       // longest spoken command
inline constexpr int RMS_THRESHOLD = 200;          // a clip whose peak is below this is silence

// Distance bands (meters) for the wake-up greeting.
inline constexpr double NEAR_DISTANCE_M = 1.0;
inline constexpr double FAR_DISTANCE_M = 3.0;

extern const char* STARTUP_TEXT;

extern const std::vector<std::string> WAKE_WORDS;
extern const std::vector<std::string> SLEEP_WAKE_WORDS;  // only heard while sleeping
extern const std::vector<std::string> WIFI_TRIGGERS;
extern const std::vector<std::string> STATS_TRIGGERS;
extern const std::vector<std::string> EXIT_WORDS;
extern const std::vector<std::string> ALARM_OFF_TRIGGERS;
extern const std::vector<std::string> ALARM_ON_TRIGGERS;
extern const std::vector<std::string> LIGHT_OFF_TRIGGERS;
extern const std::vector<std::string> LIGHT_RAINBOW_TRIGGERS;
extern const std::vector<std::string> LIGHT_COLOR_NAMES;
