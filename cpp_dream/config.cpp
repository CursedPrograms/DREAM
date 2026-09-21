#include "config.h"

#include "common.h"

#include <windows.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using dream::fileExists;

const char* STARTUP_TEXT = "ComCentre online. DREAM is ready. Say Hey DREAM to wake me.";

const std::vector<std::string> WAKE_WORDS = {
    "hey dream", "hey, dream", "hi dream", "hi, dream", "okay dream", "ok dream", "dream",
};
const std::vector<std::string> SLEEP_WAKE_WORDS = {"wake up", "wake up dream", "wake up, dream"};
const std::vector<std::string> WIFI_TRIGGERS = {
    "check wifi", "check the wifi", "wifi scan", "scan wifi",
    "scan the wifi", "who's on the wifi", "who is on the wifi",
    "check network", "network scan", "check connections",
    "what devices", "list devices", "show devices",
};
const std::vector<std::string> STATS_TRIGGERS = {
    "system stats", "cpu usage", "ram usage", "memory usage",
    "disk usage", "how's the system", "system status",
    "how are you doing", "check stats", "check the stats",
    "temperature", "cpu temp", "how hot", "system health",
};
const std::vector<std::string> EXIT_WORDS = {"goodbye", "exit", "quit", "bye", "shut down", "shutdown"};
const std::vector<std::string> ALARM_OFF_TRIGGERS = {
    "turn off the alarm", "disable the alarm", "disarm the alarm", "alarm off", "stop the alarm",
};
const std::vector<std::string> ALARM_ON_TRIGGERS = {
    "turn on the alarm", "enable the alarm", "arm the alarm", "alarm on",
};
const std::vector<std::string> LIGHT_OFF_TRIGGERS = {
    "lights off", "turn off the lights", "turn the lights off", "lights out",
};
const std::vector<std::string> LIGHT_RAINBOW_TRIGGERS = {
    "rainbow lights", "lights rainbow", "make the lights rainbow", "rainbow mode", "party lights",
};
const std::vector<std::string> LIGHT_COLOR_NAMES = {
    "red", "green", "blue", "yellow", "orange", "purple", "pink", "cyan", "white",
};

namespace {

std::string g_memoriesOverride;

std::string firstFile(const std::vector<std::string>& candidates) {
    for (const auto& c : candidates) {
        if (fileExists(c)) return c;
    }
    return "";
}

std::string exeDir() {
    wchar_t buf[MAX_PATH] = {0};
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return fs::path(buf).parent_path().string();
}

Config load() {
    Config c;
    c.root = dream::repoRoot();
    c.charName = "DREAM";
    c.systemPrompt = "You are DREAM, a helpful assistant.";

    std::ifstream f(fs::path(c.root) / "config.json");
    if (f) {
        try {
            auto j = nlohmann::json::parse(f);
            const auto& d = j.at("Config").at("DREAM");
            c.charName = d.value("CharName", c.charName);
            std::string prompt = d.value("SystemPrompt", c.systemPrompt);
            for (size_t pos; (pos = prompt.find("{name}")) != std::string::npos;) prompt.replace(pos, 6, c.charName);
            c.systemPrompt = prompt;
        } catch (const std::exception& e) {
            dream::logf("config.json: %s - using defaults", e.what());
        }
    } else {
        dream::logf("config.json not found under %s - using defaults", c.root.c_str());
    }

    c.videosDir = (fs::path(c.root) / "videos").string();
    c.voicesDir = (fs::path(c.root) / "voices").string();
    c.memoriesDir = (fs::path(c.root) / "memories").string();
    c.audioDir = (fs::path(c.root) / "audio").string();

    std::string exe = exeDir();
    c.piperExe = firstFile({
        (fs::path(c.root) / "venv311" / "Scripts" / "piper.exe").string(),
        (fs::path(c.root) / "venv" / "Scripts" / "piper.exe").string(),
        (fs::path(exe) / "piper.exe").string(),
        (fs::path(exe) / "piper" / "piper.exe").string(),
        (fs::path(c.root) / "piper" / "piper.exe").string(),
    });
    if (c.piperExe.empty()) c.piperExe = "piper.exe"; // hope it's on PATH

    std::error_code ec;
    if (fs::is_directory(c.voicesDir, ec)) {
        std::vector<std::string> voices;
        for (const auto& e : fs::directory_iterator(c.voicesDir, ec)) {
            if (e.path().extension() == ".onnx") voices.push_back(e.path().string());
        }
        std::sort(voices.begin(), voices.end());
        if (!voices.empty()) c.voiceModel = voices.front();
    }

    c.whisperModel = firstFile({
        (fs::path(exe) / "models" / "ggml-tiny.en.bin").string(),
        (fs::path(exe) / ".." / "models" / "ggml-tiny.en.bin").lexically_normal().string(),
        (fs::path(c.root) / "cpp_dream" / "models" / "ggml-tiny.en.bin").string(),
    });
    return c;
}

Config& mutableCfg() {
    static Config c = load();
    return c;
}

} // namespace

const Config& cfg() { return mutableCfg(); }

void setMemoriesDir(const std::string& dir) { mutableCfg().memoriesDir = dir; }
