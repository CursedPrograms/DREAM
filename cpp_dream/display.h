// display.h - the avatar window (dream.py's run_display()).
#pragma once

#include <string>

struct DisplayOptions {
    bool windowed = false;      // a normal window instead of borderless fullscreen
    std::string shotPath;       // save a screenshot (BMP) of the window here...
    double shotAfterSec = 4.0;  // ...this long after it opens, then keep running
};

// Opens the window and shows the avatar until S().running goes false (Esc/Q
// or closing the window sets it). Runs on the calling thread.
void runDisplay(const DisplayOptions& opts);

// Decodes `path` for a moment and writes one frame to a BMP - a check that
// video decoding works, with no window. Returns false on failure.
bool dumpVideoFrame(const std::string& path, const std::string& bmpPath, int frames = 30);
