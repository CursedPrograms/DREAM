#include "brain.h"

#include "audio.h"
#include "common.h"
#include "config.h"
#include "llm.h"
#include "memory.h"
#include "sensors.h"
#include "state.h"
#include "stt.h"
#include "sysinfo.h"
#include "tts.h"
#include "video.h"
#include "wav.h"

#include <cstdio>
#include <filesystem>
#include <random>
#include <regex>
#include <thread>

namespace fs = std::filesystem;
using dream::logf;

namespace {

enum class Trigger { None, Wake, Wifi };

std::string wakeGreeting() {
    double d = S().distanceM;
    if (d < 0) return "Yes?";
    if (d < NEAR_DISTANCE_M) return "Whoa, hi! Yes?";
    if (d > FAR_DISTANCE_M) return "Yes? I hear you over there.";
    return "Yes?";
}

// Only matches a colour when it's clearly about the lights, so ordinary chat
// ("I like the color blue") doesn't trigger the RGB board.
std::string matchLightColor(const std::string& lower) {
    static const std::regex about(R"(\blight(s)?\b|\bmake it\b|\bturn it\b)");
    if (!std::regex_search(lower, about)) return "";
    for (const auto& name : LIGHT_COLOR_NAMES) {
        if (std::regex_search(lower, std::regex("\\b" + name + "\\b"))) return name;
    }
    return "";
}

// Listens in short clips until the wake word (or "wake up", or the sensor
// seeing someone, while asleep) is heard.
Trigger listenForWakeWord() {
    logf(S().sleeping ? "Sleeping - listening for 'Wake Up'..." : "Listening for wake word 'Hey DREAM'...");
    if (!S().sleeping) S().setState("idle");

    while (S().running) {
        if (S().sleeping && S().sensorWake) {
            S().sensorWake = false;
            logf("Presence detected - waking DREAM!");
            exitSleep();
            return Trigger::Wake;
        }

        MicClip clip = recordFixed(WAKE_SECONDS);
        if (!clip.ok) { dream::sleepMs(500); continue; } // no usable microphone: keep trying
        if (peakLevel(clip.samples) <= RMS_THRESHOLD) continue;

        std::string text = dream::toLower(sttTranscribe(clip.samples, clip.rate));
        if (text.empty()) continue;
        logf("Wake check: '%s'", text.c_str());

        if (S().sleeping) {
            if (dream::containsAny(text, SLEEP_WAKE_WORDS)) {
                logf("Wake-up word detected - waking DREAM!");
                exitSleep();
                return Trigger::Wake;
            }
            continue; // ignore everything else while sleeping
        }
        if (dream::containsAny(text, WAKE_WORDS)) {
            logf("Wake word detected!");
            touchInteraction();
            return Trigger::Wake;
        }
        if (dream::containsAny(text, WIFI_TRIGGERS)) {
            touchInteraction();
            return Trigger::Wifi;
        }
    }
    return Trigger::None;
}

void doWifiScan() {
    S().setState("thinking");
    speak("Scanning the network. Give me a moment.");
    S().setState("thinking");
    std::string summary = runNetworkScan();
    speak(summary);
}

// Handles one spoken command. Returns false when the user asked to quit.
bool handleCommand(const std::string& userText, std::vector<Turn>& history) {
    std::string lower = dream::toLower(userText);

    if (dream::containsAny(lower, EXIT_WORDS)) {
        speak("Goodbye.");
        S().running = false;
        return false;
    }
    if (dream::containsAny(lower, WIFI_TRIGGERS)) { doWifiScan(); return true; }
    if (dream::containsAny(lower, STATS_TRIGGERS)) { speak(buildStatsSummary()); return true; }

    if (dream::containsAny(lower, ALARM_OFF_TRIGGERS)) {
        speak(sendSensorCommand("ALARM OFF") ? "Alarm disabled." : "I can't reach the alarm board.");
        return true;
    }
    if (dream::containsAny(lower, ALARM_ON_TRIGGERS)) {
        speak(sendSensorCommand("ALARM ON") ? "Alarm enabled." : "I can't reach the alarm board.");
        return true;
    }
    if (dream::containsAny(lower, LIGHT_OFF_TRIGGERS)) {
        speak(sendSensorCommand("RGB OFF") ? "Lights off." : "I can't reach the lights.");
        return true;
    }
    if (dream::containsAny(lower, LIGHT_RAINBOW_TRIGGERS)) {
        speak(sendSensorCommand("RGB RAINBOW") ? "Rainbow mode." : "I can't reach the lights.");
        return true;
    }
    std::string colour = matchLightColor(lower);
    if (!colour.empty()) {
        std::string upper = colour;
        for (auto& c : upper) c = char(std::toupper(static_cast<unsigned char>(c)));
        speak(sendSensorCommand("RGB " + upper) ? "Lights " + colour + "." : "I can't reach the lights.");
        return true;
    }

    for (const auto& [kind, fact] : extractMemories(userText)) {
        if (remember(kind, fact) == Remembered::Milestone) logf("Milestone: %s", fact.c_str());
    }

    std::string response = askLlm(userText, history);
    history.push_back({false, userText});
    history.push_back({true, response});
    if (history.size() > 12) history.erase(history.begin(), history.end() - 12);
    speak(response);
    return true;
}

void startupAnnouncement() {
    // Same order as dream.py without the lipsync cache: the intro clip (with its
    // own audio if there is one), then she says she's ready.
    std::string introVideo = (fs::path(cfg().videosDir) / "intro1.mp4").string();
    std::string introAudio = (fs::path(cfg().audioDir) / "intro1.wav").string();
    if (dream::fileExists(introVideo)) {
        S().setState("talking");
        S().setForceVideo(introVideo);
        if (dream::fileExists(introAudio)) {
            playWavFile(introAudio);
        } else {
            while (!S().forceVideo().empty() && S().running) dream::sleepMs(100); // let the clip play through
        }
        S().setState("idle");
    }
    speak(STARTUP_TEXT);
}

} // namespace

namespace {

// Short lines she says again and again ("Yes?", "Bye for now.", the startup
// line) are voiced once and kept, like dream.py's lipsync cache, so they play
// at once instead of waiting for Piper to start up. Keyed by text + voice.
std::string cachedSpeechPath(const std::string& text) {
    if (text.size() > 80) return "";
    char key[32];
    std::snprintf(key, sizeof(key), "%016llx", static_cast<unsigned long long>(std::hash<std::string>{}(text + "|" + cfg().voiceModel)));
    std::error_code ec;
    fs::path dir = fs::path(dream::tempDir()) / "speech_cache";
    fs::create_directories(dir, ec);
    return (dir / (std::string(key) + ".wav")).string();
}

} // namespace

void speak(const std::string& text) {
    if (text.empty()) return;
    logf("DREAM: %s", text.c_str());

    std::string cached = cachedSpeechPath(text);
    bool useCache = !cached.empty();
    std::string wav = useCache ? cached : dream::tempPath("tts", ".wav");
    bool have = useCache && dream::fileExists(cached);
    if (!have) have = ttsSynthesize(text, wav);

    if (have) {
        S().setState("talking");
        playWavFile(wav);
    }
    if (!useCache) {
        for (int i = 0; i < 10; i++) { // Windows briefly keeps the file locked after playback
            std::error_code ec;
            fs::remove(wav, ec);
            if (!fs::exists(wav)) break;
            dream::sleepMs(50);
        }
    }
    S().setState("idle");
}

void enterSleep() {
    if (S().sleeping) return;
    logf("DREAM is falling asleep...");
    S().sleeping = true;
    S().setState("sleeping");
}

void exitSleep() {
    if (!S().sleeping) return;
    logf("DREAM is waking up!");
    S().sleeping = false;
    S().setState("idle");
    touchInteraction();
}

void handleSensorLine(const std::string& line) {
    if (line == "PRESENT") {
        S().presenceSeen = true;
        if (S().sleeping) S().sensorWake = true; // the wake loop wakes her, like the spoken "wake up"
        else touchInteraction();
    } else if (line == "ABSENT") {
        if (S().presenceSeen) {
            S().presenceSeen = false;
            if (!S().sleeping && S().state() == "idle") std::thread([] { speak("Bye for now."); }).detach();
        }
    } else if (line.rfind("RANGE", 0) == 0) {
        double meters = 0;
        if (std::sscanf(line.c_str(), "RANGE %lf", &meters) == 1) {
            S().distanceM = meters;
            if (!S().sleeping) touchInteraction();
        }
    } else if (line == "MOTION") {
        logf("Motion - the alarm went off");
    } else if (line.rfind("RADAR_ERROR", 0) == 0) {
        logf("Sensor board: %s", line.c_str());
    }
}

void timersLoop() {
    // dream.py's flirt_watcher and sleep_watcher, together: flirt at 10 minutes
    // idle, sleep at 15. Anything happening resets the clock.
    while (S().running) {
        for (int i = 0; i < 50 && S().running; i++) dream::sleepMs(100); // every 5 s
        if (S().sleeping) continue;
        if (S().state() != "idle") { touchInteraction(); continue; }

        double elapsed = dream::nowSec() - S().lastWakeTs;
        if (elapsed >= SLEEP_IDLE_TIMEOUT) {
            logf("Idle for %.0fs - entering sleep mode", elapsed);
            enterSleep();
        } else if (elapsed >= FLIRT_IDLE_TIMEOUT && !S().flirtPlayed) {
            S().flirtPlayed = true;
            VideoPools pools = buildVideoPools();
            const auto& flirts = pools["flirtytalk"];
            if (flirts.empty()) {
                logf("Flirt timeout but no flirtytalk videos found.");
            } else {
                static std::mt19937 rng{std::random_device{}()};
                const std::string& chosen = flirts[std::uniform_int_distribution<size_t>(0, flirts.size() - 1)(rng)];
                logf("Flirt timeout - queueing %s", fs::path(chosen).filename().string().c_str());
                S().setForceVideo(chosen);
            }
        }
    }
}

void voiceLoop() {
    std::vector<std::string> models;
    if (!ollamaAvailable(models)) {
        logf("Ollama not running - voice loop disabled. Display will still open.");
        return;
    }
    std::string list;
    for (const auto& m : models) list += (list.empty() ? "" : ", ") + m;
    logf("Ollama: %s", list.empty() ? "no models" : list.c_str());

    if (!sttLoad(cfg().whisperModel)) {
        logf("Speech recognition unavailable - voice loop disabled. Display will still open.");
        return;
    }
    logf("Whisper ready.");
    if (micCount() == 0) logf("No microphone found - I can talk but not listen.");

    startupAnnouncement();

    std::vector<Turn> history;
    while (S().running) {
        Trigger trigger = listenForWakeWord();
        if (trigger == Trigger::None || !S().running) break;

        if (trigger == Trigger::Wifi) { doWifiScan(); continue; }

        speak(wakeGreeting());
        S().setState("listening");
        logf("Listening for command...");

        MicClip clip = recordCommand(RECORD_SECONDS);
        if (!clip.ok) { speak("I didn't catch that."); continue; }
        if (!clip.heardSpeech || peakLevel(clip.samples) <= RMS_THRESHOLD) {
            speak("I couldn't hear you clearly.");
            continue;
        }

        std::string userText = sttTranscribe(clip.samples, clip.rate);
        if (userText.empty()) { speak("I didn't catch that."); continue; }

        logf("You: %s", userText.c_str());
        S().setState("idle");
        touchInteraction();
        if (!handleCommand(userText, history)) break;
    }
}
