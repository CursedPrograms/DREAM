#include "memory.h"

#include "common.h"
#include "config.h"

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>

namespace fs = std::filesystem;

namespace {

struct Entry {
    std::time_t when;
    std::string kind, text;
};

std::string memoriesPath() { return (fs::path(cfg().memoriesDir) / "memories.txt").string(); }
std::string milestonesPath() { return (fs::path(cfg().memoriesDir) / "mymilestones.txt").string(); }

// Only the latest fact matters for these (a new name replaces the old one).
bool singleValue(const std::string& kind) {
    return kind == "name" || kind == "home" || kind == "job" || kind == "birthday";
}

std::vector<Entry> readMemories() {
    std::vector<Entry> out;
    std::ifstream f(memoriesPath());
    static const std::regex line(R"(\[(.*?)\]\s*\((\w+)\)\s*(.*))");
    std::string s;
    while (std::getline(f, s)) {
        s = dream::trim(s);
        std::smatch m;
        if (!std::regex_match(s, m, line)) continue;
        std::tm tmv{};
        if (std::sscanf(m[1].str().c_str(), "%d-%d-%d %d:%d", &tmv.tm_year, &tmv.tm_mon, &tmv.tm_mday, &tmv.tm_hour, &tmv.tm_min) != 5) continue;
        tmv.tm_year -= 1900;
        tmv.tm_mon -= 1;
        tmv.tm_isdst = -1;
        std::time_t when = std::mktime(&tmv);
        if (when == std::time_t(-1)) continue;
        out.push_back({when, m[2].str(), m[3].str()});
    }
    return out;
}

void appendLine(const std::string& path, const std::string& line) {
    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    std::ofstream f(path, std::ios::app);
    f << line << "\n";
}

std::string timestamp() {
    std::time_t t = std::time(nullptr);
    std::tm tmv;
    localtime_s(&tmv, &t);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M", &tmv);
    return buf;
}

std::string age(std::time_t when, std::time_t now) {
    long seconds = long(std::max<double>(0, std::difftime(now, when)));
    auto plural = [](long n, const char* unit) { return std::to_string(n) + " " + unit + (n != 1 ? "s" : "") + " ago"; };
    if (seconds < 3600) {
        long minutes = seconds / 60;
        return minutes < 1 ? "just now" : plural(minutes, "minute");
    }
    if (seconds < 86400) return plural(seconds / 3600, "hour");
    long days = seconds / 86400;
    if (days == 1) return "yesterday";
    if (days < 30) return std::to_string(days) + " days ago";
    std::tm tmv;
    localtime_s(&tmv, &when);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tmv);
    return std::string("on ") + buf;
}

} // namespace

std::vector<std::pair<std::string, std::string>> extractMemories(const std::string& userText) {
    static const std::regex name(R"(\bmy name is ([A-Z][a-zA-Z'-]{1,20})\b)", std::regex::icase);
    static const std::regex pet(R"(\bi (?:have|own) an? (dog|cat|bird|fish|rabbit|hamster)(?: named ([A-Z][a-zA-Z'-]{1,20}))?\b)", std::regex::icase);
    static const std::regex home(R"(\bi live in ([A-Za-z][A-Za-z\s]{1,30}?)[.,!]?$)", std::regex::icase);
    static const std::regex job(R"(\bi work as an? ([A-Za-z][A-Za-z\s]{1,30}?)[.,!]?$)", std::regex::icase);
    static const std::regex birthday(R"(\bmy birthday is ([A-Za-z0-9,\s]{3,30}?)[.,!]?$)", std::regex::icase);

    std::vector<std::pair<std::string, std::string>> found;
    std::smatch m;
    if (std::regex_search(userText, m, name)) found.push_back({"name", "Human's name is " + m[1].str() + "."});
    if (std::regex_search(userText, m, pet)) {
        std::string kind = dream::toLower(m[1].str());
        found.push_back({"pet", "Human has a " + kind + (m[2].matched ? " named " + m[2].str() + "." : ".")});
    }
    if (std::regex_search(userText, m, home)) found.push_back({"home", "Human lives in " + dream::trim(m[1].str()) + "."});
    if (std::regex_search(userText, m, job)) found.push_back({"job", "Human works as a " + dream::trim(m[1].str()) + "."});
    if (std::regex_search(userText, m, birthday)) found.push_back({"birthday", "Human's birthday is " + dream::trim(m[1].str()) + "."});
    return found;
}

Remembered remember(const std::string& kind, const std::string& text) {
    std::vector<Entry> entries = readMemories();
    std::string lowerText = dream::toLower(text);
    bool kindKnown = false;
    for (const auto& e : entries) {
        if (e.kind == kind && dream::toLower(e.text) == lowerText) return Remembered::Duplicate;
        if (e.kind == kind) kindKnown = true;
    }
    std::string ts = timestamp();
    appendLine(memoriesPath(), "[" + ts + "] (" + kind + ") " + text);
    if (kindKnown) return Remembered::Memory;
    appendLine(milestonesPath(), "[" + ts + "] " + text);
    return Remembered::Milestone;
}

std::string memoryPromptBlock(size_t limit) {
    // Latest-only for single-value kinds, every distinct fact otherwise; keep
    // the list in order of when each was last learned.
    std::vector<Entry> kept;
    for (const auto& e : readMemories()) {
        auto same = [&](const Entry& k) {
            if (k.kind != e.kind) return false;
            return singleValue(e.kind) || dream::toLower(k.text) == dream::toLower(e.text);
        };
        kept.erase(std::remove_if(kept.begin(), kept.end(), same), kept.end());
        kept.push_back(e);
    }
    std::stable_sort(kept.begin(), kept.end(), [](const Entry& a, const Entry& b) { return a.when < b.when; });
    if (kept.empty()) return "";
    if (kept.size() > limit) kept.erase(kept.begin(), kept.end() - long(limit));

    std::time_t now = std::time(nullptr);
    std::string block = "\n\nThings you remember about the human:\n";
    for (const auto& e : kept) block += "- " + e.text + " (learned " + age(e.when, now) + ")\n";
    block.pop_back(); // trailing newline
    return block;
}
