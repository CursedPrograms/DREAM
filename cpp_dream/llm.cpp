#include "llm.h"

#include "common.h"
#include "config.h"
#include "http.h"
#include "memory.h"
#include "state.h"

#include <atomic>
#include <nlohmann/json.hpp>

namespace {

// dream.py asks Ollama to put 20 layers on the GPU. If that crashes (an Ollama
// CUDA build the graphics driver can't run, or too little VRAM) we fall back to
// the CPU once and stay there, rather than failing every reply.
std::atomic<bool> g_gpuBroken{false};

std::string dumpJson(const nlohmann::json& j) {
    return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace); // never throw on stray bytes
}

} // namespace

std::string askLlm(const std::string& prompt, const std::vector<Turn>& history) {
    S().setState("thinking");
    dream::logf("Thinking...");

    std::string context;
    size_t start = history.size() > 6 ? history.size() - 6 : 0;
    for (size_t i = start; i < history.size(); i++) {
        context += std::string(history[i].assistant ? "You" : "Human") + ": " + history[i].text + "\n";
    }
    std::string full = "System: " + cfg().systemPrompt + memoryPromptBlock() + "\n\n" + context +
                       "Human: " + prompt + "\nYou:";

    auto ask = [&](int numGpu) {
        nlohmann::json payload = {
            {"model", cfg().model},
            {"prompt", full},
            {"stream", false},
            {"options", {{"temperature", 0.7}, {"num_predict", 150}, {"num_gpu", numGpu}}},
        };
        return httpPost(cfg().ollamaHost, cfg().ollamaPort, "/api/generate", dumpJson(payload), 120000);
    };

    HttpResult r = ask(g_gpuBroken ? 0 : 20);
    if (r.status == 500 && !g_gpuBroken && r.body.find("CUDA") != std::string::npos) {
        dream::logf("Ollama's GPU mode failed (%s) - using the CPU from now on", r.body.substr(0, 120).c_str());
        g_gpuBroken = true;
        r = ask(0);
    }
    if (r.timedOut) return "That took too long. Please try again.";
    if (!r.error.empty()) return "Error: " + r.error;
    if (r.status != 200) {
        dream::logf("Ollama answered %d: %s", r.status, r.body.substr(0, 200).c_str());
        return "Sorry, I encountered an error.";
    }

    std::string response;
    try {
        response = dream::trim(nlohmann::json::parse(r.body).value("response", ""));
    } catch (const std::exception& e) {
        return std::string("Error: ") + e.what();
    }

    size_t think = response.find("<think>");
    if (think != std::string::npos) {
        size_t end = response.find("</think>");
        if (end != std::string::npos) response = dream::trim(response.substr(end + 8));
    }
    if (dream::toLower(response.substr(0, 6)) == "dream:") response = dream::trim(response.substr(6));
    return response.empty() ? "I didn't catch that." : response;
}

bool ollamaAvailable(std::vector<std::string>& models) {
    HttpResult r = httpGet(cfg().ollamaHost, cfg().ollamaPort, "/api/tags", 3000);
    if (r.status != 200) return false;
    try {
        for (const auto& m : nlohmann::json::parse(r.body).value("models", nlohmann::json::array())) {
            models.push_back(m.value("name", ""));
        }
    } catch (...) {
    }
    return true;
}
