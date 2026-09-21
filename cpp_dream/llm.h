// llm.h - asks the local Ollama model, the way dream.py's ask_llm() does.
#pragma once

#include <string>
#include <vector>

struct Turn {
    bool assistant = false;
    std::string text;
};

// One reply. Sets the shared state to "thinking" while it works. Failures come
// back as a sentence DREAM can say ("That took too long. Please try again.").
std::string askLlm(const std::string& prompt, const std::vector<Turn>& history);

// True if Ollama answers; `models` gets the names it has pulled.
bool ollamaAvailable(std::vector<std::string>& models);
