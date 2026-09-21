#include "stt.h"

#include "common.h"
#include "wav.h"

#include <whisper.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>

namespace {

whisper_context* g_ctx = nullptr;
std::mutex g_mutex;               // one transcription at a time
std::atomic<bool> g_ready{false};

// Whisper marks non-speech as "[BLANK_AUDIO]", "(silence)", "[Music]" and so on.
bool isNonSpeechMarker(const std::string& t) {
    if (t.size() < 2) return false;
    char a = t.front(), b = t.back();
    return (a == '[' && b == ']') || (a == '(' && b == ')') || (a == '*' && b == '*');
}

} // namespace

bool sttLoad(const std::string& modelPath) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_ctx) return true;
    if (!dream::fileExists(modelPath)) {
        dream::logf("Whisper model not found: %s", modelPath.c_str());
        return false;
    }
    whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = false;
    g_ctx = whisper_init_from_file_with_params(modelPath.c_str(), cp);
    if (!g_ctx) {
        dream::logf("Could not load the Whisper model.");
        return false;
    }
    g_ready = true;
    return true;
}

bool sttReady() { return g_ready; }

std::string sttTranscribe(const std::vector<int16_t>& samples, int rate) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_ctx) return "";

    std::vector<float> pcm = toWhisperInput(samples, rate);
    if (pcm.size() < 16000 * 12 / 10) pcm.resize(16000 * 12 / 10, 0.0f); // Whisper wants at least ~1 s

    whisper_full_params p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    p.language = "en";
    p.translate = false;
    p.no_context = true;
    p.single_segment = false;
    p.print_progress = p.print_realtime = p.print_timestamps = p.print_special = false;
    p.n_threads = int(std::clamp(std::thread::hardware_concurrency() / 2, 2u, 8u));

    if (whisper_full(g_ctx, p, pcm.data(), int(pcm.size())) != 0) return "";

    std::string text;
    for (int i = 0; i < whisper_full_n_segments(g_ctx); i++) text += whisper_full_get_segment_text(g_ctx, i);
    text = dream::trim(text);
    return isNonSpeechMarker(text) ? "" : text;
}
