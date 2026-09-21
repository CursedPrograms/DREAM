// brain.h - what DREAM does: the wake-word and voice loop, the idle timers
// (flirt, sleep), reacting to the sensors, and speaking. This is dream.py's
// logic; the display, audio, speech and sensor parts it uses are separate.
#pragma once

#include <string>

// Speaks `text` (Piper -> speakers) and shows the "talking" video while it does.
void speak(const std::string& text);

void enterSleep();
void exitSleep();

// One line from the sensor board: PRESENT / ABSENT / RANGE x m ...
void handleSensorLine(const std::string& line);

// Background loops (run each on its own thread).
void timersLoop();  // flirt clip after FLIRT_IDLE_TIMEOUT, sleep after SLEEP_IDLE_TIMEOUT
void voiceLoop();   // loads Whisper, announces startup, then waits for the wake word
