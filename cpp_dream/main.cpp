// dream.exe - DREAM as one program: the C++ version of scripts/dream.py.
//
//   dream.exe                 full screen avatar, "Hey DREAM" wake word, sensors
//   dream.exe --windowed      a normal window instead of borderless fullscreen
//   dream.exe --shot out.bmp  save a screenshot of the window a few seconds in
//   dream.exe --no-voice      display only (no microphone, Whisper or Ollama)
//   dream.exe --no-sensors    don't look for the sensor board
//   dream.exe --selftest      check each part works (no window, mic or speakers needed)
//
// Esc or Q closes it. Needs, next to the repo: config.json, videos/, voices/*.onnx,
// venv311/Scripts/piper.exe (or piper.exe beside this exe), Ollama running with the
// model from config, and models/ggml-tiny.en.bin.

#include "audio.h"
#include "brain.h"
#include "common.h"
#include "config.h"
#include "display.h"
#include "llm.h"
#include "memory.h"
#include "sensors.h"
#include "state.h"
#include "stt.h"
#include "sysinfo.h"
#include "tts.h"
#include "video.h"
#include "wav.h"

#include <windows.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>

namespace fs = std::filesystem;
using dream::logf;

namespace {

BOOL WINAPI onCtrlC(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT) {
        S().running = false;
        stopPlayback();
        return TRUE;
    }
    return FALSE;
}

int selfTest(bool withWifi) {
    int failures = 0;
    auto check = [&](bool ok, const std::string& what) {
        logf("%s  %s", ok ? "PASS" : "FAIL", what.c_str());
        if (!ok) failures++;
    };
    auto note = [&](const std::string& what) { logf("INFO  %s", what.c_str()); };

    // Keep the test's memories out of the real memories/ folder.
    std::string mem = (fs::path(dream::tempDir()) / "selftest_memories").string();
    std::error_code ec;
    fs::remove_all(mem, ec);
    setMemoriesDir(mem);

    logf("== Files");
    check(dream::dirExists(cfg().videosDir), "videos/ found: " + cfg().videosDir);
    check(!cfg().voiceModel.empty(), "voice model: " + cfg().voiceModel);
    check(dream::fileExists(cfg().piperExe), "piper: " + cfg().piperExe);
    check(dream::fileExists(cfg().whisperModel), "whisper model: " + cfg().whisperModel);
    check(!cfg().systemPrompt.empty() && cfg().systemPrompt.find("{name}") == std::string::npos, "config.json system prompt loaded");

    logf("== Memory");
    auto facts = extractMemories("my name is Sam and I have a dog named Rex");
    check(facts.size() == 2, "extracts a name and a pet");
    check(remember("name", "Human's name is Sam.") == Remembered::Milestone, "first name is a milestone");
    check(remember("name", "Human's name is Sam.") == Remembered::Duplicate, "repeat is ignored");
    check(remember("name", "Human's name is Alex.") == Remembered::Memory, "new name is a plain memory");
    std::string block = memoryPromptBlock();
    check(block.find("Alex") != std::string::npos && block.find("Sam") == std::string::npos, "prompt keeps only the latest name");
    check(block.find("(learned just now)") != std::string::npos, "prompt says how long ago");

    logf("== Speech: Piper -> Whisper");
    std::string wav = dream::tempPath("selftest", ".wav");
    bool spoke = ttsSynthesize("Hey dream, what is the time?", wav);
    check(spoke, "Piper made a WAV");
    if (spoke) {
        Wav w;
        bool readable = readWav(wav, w) && w.samples.size() > 1000;
        check(readable, "WAV readable (" + std::to_string(w.sampleRate) + " Hz)");
        check(sttLoad(cfg().whisperModel), "Whisper model loaded");
        std::string heard = sttTranscribe(w.samples, w.sampleRate);
        note("Whisper heard: \"" + heard + "\"");
        check(dream::toLower(heard).find("dream") != std::string::npos, "Whisper recognised the wake word");
    }
    fs::remove(wav, ec);

    logf("== Ollama");
    std::vector<std::string> models;
    bool up = ollamaAvailable(models);
    check(up, "Ollama answers");
    if (up) {
        std::vector<Turn> history;
        std::string reply = askLlm("Reply with just the word: ready", history);
        note("model replied: \"" + reply + "\"");
        check(!reply.empty() && reply.rfind("Error", 0) != 0 && reply.rfind("Sorry", 0) != 0, "model gave a reply");
    }

    logf("== Video");
    std::string idle = (fs::path(cfg().videosDir) / "idle0.mp4").string();
    std::string bmp = (fs::path(dream::tempDir()) / "selftest_frame.bmp").string();
    check(dumpVideoFrame(idle, bmp, 30), "decoded idle0.mp4 -> " + bmp);
    VideoPools pools = buildVideoPools();
    check(!pools["idle"].empty() && !pools["sleeping"].empty(), "video pools built (idle + sleeping present)");

    logf("== System");
    std::string stats = buildStatsSummary();
    note(stats);
    check(stats.find("CPU is at") != std::string::npos && stats.find("RAM usage") != std::string::npos, "stats summary");
    if (withWifi) {
        std::string scan = runNetworkScan();
        note(scan.substr(0, 300));
        check(scan.find("I found") == 0, "network scan");
    }

    logf("== Sensors");
    auto ports = listComPorts();
    note(std::to_string(ports.size()) + " serial port(s) found");
    for (const auto& [port, desc] : ports) note("  " + port + " - " + desc);
    std::string board = findDreamBoard();
    note(board.empty() ? "no board answered \"I am Dream\" (fine if it isn't plugged in / flashed yet)"
                       : "board answered on " + board);

    logf("== Microphone");
    note(std::to_string(micCount()) + " recording device(s)");
    if (micCount() > 0) {
        MicClip clip = recordFixed(0.4);
        check(clip.ok && !clip.samples.empty(), "opened the mic and recorded " + std::to_string(clip.samples.size()) +
                                                    " samples at " + std::to_string(clip.rate) + " Hz (peak " +
                                                    std::to_string(peakLevel(clip.samples)) + ")");
    }

    logf(failures == 0 ? "ALL CHECKS PASSED" : "%d CHECK(S) FAILED", failures);
    return failures == 0 ? 0 : 1;
}

void usage() {
    std::printf(
        "dream.exe - DREAM as one program\n\n"
        "  --windowed        normal window instead of borderless fullscreen\n"
        "  --shot FILE.bmp   save a screenshot of the window a few seconds in\n"
        "  --no-voice        display only (no microphone, Whisper or Ollama)\n"
        "  --no-sensors      don't look for the sensor board\n"
        "  --mute            don't play sound (still shows the talking video)\n"
        "  --mic-file F.wav  use a WAV as the microphone (for testing without one)\n"
        "  --flirt-after S   flirt clip after S idle seconds (default 600)\n"
        "  --sleep-after S   fall asleep after S idle seconds (default 900)\n"
        "  --selftest        check each part works, then exit (add --wifi to include the network scan)\n"
        "\nEsc or Q closes the window.\n");
}

} // namespace

int main(int argc, char** argv) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCtrlHandler(onCtrlC, TRUE);

    DisplayOptions display;
    bool noVoice = false, noSensors = false, selftest = false, wifi = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--windowed") display.windowed = true;
        else if (a == "--shot" && i + 1 < argc) display.shotPath = argv[++i];
        else if (a == "--shot-after" && i + 1 < argc) display.shotAfterSec = std::atof(argv[++i]);
        else if (a == "--no-voice") noVoice = true;
        else if (a == "--mute") setMuted(true);
        else if (a == "--mic-file" && i + 1 < argc) {
            if (!setMicFile(argv[++i])) { std::printf("Could not read %s\n", argv[i]); return 2; }
        }
        else if (a == "--flirt-after" && i + 1 < argc) FLIRT_IDLE_TIMEOUT = std::atof(argv[++i]);
        else if (a == "--sleep-after" && i + 1 < argc) SLEEP_IDLE_TIMEOUT = std::atof(argv[++i]);
        else if (a == "--no-sensors") noSensors = true;
        else if (a == "--selftest") selftest = true;
        else if (a == "--wifi") wifi = true;
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else { std::printf("Unknown option: %s\n\n", a.c_str()); usage(); return 2; }
    }

    logf("DREAM (C++) - %s", cfg().charName.c_str());
    logf("Root: %s", cfg().root.c_str());
    if (selftest) return selfTest(wifi);

    touchInteraction();
    std::thread(timersLoop).detach();
    if (!noSensors) sensorsStart(handleSensorLine);
    if (!noVoice) std::thread(voiceLoop).detach();

    runDisplay(display); // the main thread lives here until the window closes

    S().running = false;
    stopPlayback();
    logf("Bye.");
    std::fflush(stdout);
    return 0; // detached threads end with the process
}
