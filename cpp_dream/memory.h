// memory.h - DREAM's long-term memory. Same files and line format as
// scripts/dream_memory.py, so the Python and C++ versions share memories.
#pragma once

#include <string>
#include <utility>
#include <vector>

// Pull simple durable facts (name, pet, home, job, birthday) out of what the
// user said. Returns {kind, sentence} pairs.
std::vector<std::pair<std::string, std::string>> extractMemories(const std::string& userText);

enum class Remembered { Duplicate, Memory, Milestone };

// Saves a fact to memories/memories.txt. The first fact of each kind is also
// logged to mymilestones.txt (Milestone); a repeat of something already known
// is ignored (Duplicate).
Remembered remember(const std::string& kind, const std::string& text);

// Text to append to the system prompt ("Things you remember about the human: ...",
// each with how long ago it was learned), or "" if nothing is remembered.
std::string memoryPromptBlock(size_t limit = 6);
