#include "common.h"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <thread>

namespace fs = std::filesystem;

namespace dream {

std::string repoRoot() {
    static const std::string root = [] {
        std::vector<fs::path> starts;
        wchar_t buf[MAX_PATH] = {0};
        if (GetModuleFileNameW(nullptr, buf, MAX_PATH)) starts.push_back(fs::path(buf).parent_path());
        starts.push_back(fs::current_path());
        for (const auto& start : starts) {
            for (fs::path d = start;; d = d.parent_path()) {
                if (fs::exists(d / "config.json") && fs::exists(d / "scripts")) return d.string();
                if (d == d.parent_path()) break;
            }
        }
        return fs::current_path().string();
    }();
    return root;
}

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return s;
}

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) a++;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) b--;
    return s.substr(a, b - a);
}

bool containsAny(const std::string& haystack, const std::vector<std::string>& needles) {
    for (const auto& n : needles) {
        if (haystack.find(n) != std::string::npos) return true;
    }
    return false;
}

bool fileExists(const std::string& path) {
    std::error_code ec;
    return fs::is_regular_file(path, ec);
}

bool dirExists(const std::string& path) {
    std::error_code ec;
    return fs::is_directory(path, ec);
}

double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

double nowSec() { return nowMs() / 1000.0; }

void sleepMs(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

void logf(const char* fmt, ...) {
    static std::mutex m;
    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);

    std::time_t t = std::time(nullptr);
    std::tm tmv;
    localtime_s(&tmv, &t);
    std::lock_guard<std::mutex> lock(m);
    std::printf("[%02d:%02d:%02d] %s\n", tmv.tm_hour, tmv.tm_min, tmv.tm_sec, msg);
    std::fflush(stdout);
}

std::string tempDir() {
    static const std::string dir = [] {
        wchar_t buf[MAX_PATH] = {0};
        GetTempPathW(MAX_PATH, buf);
        fs::path p = fs::path(buf) / "dream_cpp";
        std::error_code ec;
        fs::create_directories(p, ec);
        return p.string();
    }();
    return dir;
}

std::string tempPath(const char* tag, const char* ext) {
    static std::atomic<unsigned> counter{0};
    char name[128];
    std::snprintf(name, sizeof(name), "%s_%lu_%u%s", tag, static_cast<unsigned long>(GetCurrentProcessId()), counter++, ext);
    return (fs::path(tempDir()) / name).string();
}

} // namespace dream
