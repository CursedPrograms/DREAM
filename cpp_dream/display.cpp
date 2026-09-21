#include "display.h"

#include "common.h"
#include "state.h"
#include "video.h"

#include <windows.h>
#include <mfapi.h>

#include <algorithm>
#include <cstdio>
#include <fstream>

namespace {

constexpr uint32_t BLANK_COLOUR = 0x000A0C14; // (10, 12, 20), like dream.py's blank frame

LRESULT CALLBACK wndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_KEYDOWN:
            if (wp == VK_ESCAPE || wp == 'Q') {
                dream::logf("Key exit");
                S().running = false;
            }
            return 0;
        case WM_CLOSE:
            S().running = false;
            return 0;
        case WM_SETCURSOR:
            SetCursor(nullptr); // no pointer over the avatar, like dream.py
            return TRUE;
        case WM_ERASEBKGND:
            return 1;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

bool writeBmp(const std::string& path, const uint32_t* px, int w, int h) { // px: top-down BGRA
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    BITMAPFILEHEADER fh = {};
    BITMAPINFOHEADER ih = {};
    ih.biSize = sizeof(ih);
    ih.biWidth = w;
    ih.biHeight = -h; // top-down
    ih.biPlanes = 1;
    ih.biBitCount = 32;
    ih.biCompression = BI_RGB;
    ih.biSizeImage = DWORD(w) * DWORD(h) * 4;
    fh.bfType = 0x4D42;
    fh.bfOffBits = sizeof(fh) + sizeof(ih);
    fh.bfSize = fh.bfOffBits + ih.biSizeImage;
    f.write(reinterpret_cast<const char*>(&fh), sizeof(fh));
    f.write(reinterpret_cast<const char*>(&ih), sizeof(ih));
    f.write(reinterpret_cast<const char*>(px), ih.biSizeImage);
    return bool(f);
}

bool saveWindowShot(HWND hwnd, const std::string& path) {
    RECT rc;
    GetClientRect(hwnd, &rc);
    int w = rc.right - rc.left, h = rc.bottom - rc.top;
    if (w <= 0 || h <= 0) return false;
    HDC dc = GetDC(hwnd);
    HDC mem = CreateCompatibleDC(dc);
    HBITMAP bmp = CreateCompatibleBitmap(dc, w, h);
    HGDIOBJ old = SelectObject(mem, bmp);
    BitBlt(mem, 0, 0, w, h, dc, 0, 0, SRCCOPY);
    BITMAPINFO bi = {};
    bi.bmiHeader.biSize = sizeof(bi.bmiHeader);
    bi.bmiHeader.biWidth = w;
    bi.bmiHeader.biHeight = -h;
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    std::vector<uint32_t> px(size_t(w) * size_t(h));
    GetDIBits(mem, bmp, 0, UINT(h), px.data(), &bi, DIB_RGB_COLORS);
    SelectObject(mem, old);
    DeleteObject(bmp);
    DeleteDC(mem);
    ReleaseDC(hwnd, dc);
    return writeBmp(path, px.data(), w, h);
}

// The part of `f` that fills a window of `aspect` (width/height) with no bars:
// a centred crop, like dream.py's "scale to cover".
const Frame& coverCrop(const Frame& f, double aspect, Frame& scratch) {
    double fa = double(f.w) / double(f.h);
    if (std::abs(fa - aspect) < 0.01) return f;
    int cw = f.w, ch = f.h;
    if (fa > aspect) cw = std::max(1, int(f.h * aspect));
    else ch = std::max(1, int(f.w / aspect));
    int x0 = (f.w - cw) / 2, y0 = (f.h - ch) / 2;
    scratch.w = cw;
    scratch.h = ch;
    scratch.px.resize(size_t(cw) * size_t(ch));
    for (int y = 0; y < ch; y++) {
        std::copy_n(&f.px[size_t(y0 + y) * size_t(f.w) + size_t(x0)], cw, &scratch.px[size_t(y) * size_t(cw)]);
    }
    return scratch;
}

} // namespace

void runDisplay(const DisplayOptions& opts) {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(MFStartup(MF_VERSION))) {
        dream::logf("Media Foundation failed to start - no video playback.");
        S().running = false;
        return;
    }

    HINSTANCE inst = GetModuleHandleW(nullptr);
    WNDCLASSW wc = {};
    wc.lpfnWndProc = wndProc;
    wc.hInstance = inst;
    wc.lpszClassName = L"DreamCppWindow";
    wc.hCursor = nullptr;
    RegisterClassW(&wc);

    int sw = GetSystemMetrics(SM_CXSCREEN), sh = GetSystemMetrics(SM_CYSCREEN);
    HWND hwnd;
    if (opts.windowed) {
        int w = std::min(1280, sw - 100), h = w * 9 / 16;
        RECT r = {0, 0, w, h};
        AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
        hwnd = CreateWindowExW(0, wc.lpszClassName, L"DREAM", WS_OVERLAPPEDWINDOW | WS_VISIBLE, 60, 40,
                               r.right - r.left, r.bottom - r.top, nullptr, nullptr, inst, nullptr);
    } else {
        // Borderless "windowed fullscreen", as dream.py does (real fullscreen mode
        // crashes on some Windows/driver setups).
        hwnd = CreateWindowExW(0, wc.lpszClassName, L"DREAM", WS_POPUP | WS_VISIBLE, 0, 0, sw, sh, nullptr, nullptr,
                               inst, nullptr);
    }
    if (!hwnd) {
        dream::logf("Could not create the window (error %lu)", GetLastError());
        S().running = false;
        return;
    }
    ShowCursor(FALSE);
    SetForegroundWindow(hwnd);
    dream::logf("Window open - Esc or Q to quit.");

    VideoStateManager vsm(buildVideoPools());
    std::vector<uint32_t> blank;
    Frame scratch;
    double opened = dream::nowSec();
    bool shotTaken = opts.shotPath.empty();

    while (S().running) {
        MSG m;
        while (PeekMessageW(&m, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&m);
            DispatchMessageW(&m);
        }
        if (!S().running) break;

        double frameStart = dream::nowMs();
        RECT rc;
        GetClientRect(hwnd, &rc);
        int cw = rc.right - rc.left, ch = rc.bottom - rc.top;
        if (cw > 0 && ch > 0) {
            const Frame* f = vsm.getFrame(frameStart, S().state());
            HDC dc = GetDC(hwnd);
            SetStretchBltMode(dc, COLORONCOLOR);
            BITMAPINFO bi = {};
            bi.bmiHeader.biSize = sizeof(bi.bmiHeader);
            bi.bmiHeader.biPlanes = 1;
            bi.bmiHeader.biBitCount = 32;
            bi.bmiHeader.biCompression = BI_RGB;
            if (f && f->w > 0) {
                const Frame& src = coverCrop(*f, double(cw) / double(ch), scratch);
                bi.bmiHeader.biWidth = src.w;
                bi.bmiHeader.biHeight = -src.h;
                StretchDIBits(dc, 0, 0, cw, ch, 0, 0, src.w, src.h, src.px.data(), &bi, DIB_RGB_COLORS, SRCCOPY);
            } else {
                blank.assign(64 * 36, BLANK_COLOUR);
                bi.bmiHeader.biWidth = 64;
                bi.bmiHeader.biHeight = -36;
                StretchDIBits(dc, 0, 0, cw, ch, 0, 0, 64, 36, blank.data(), &bi, DIB_RGB_COLORS, SRCCOPY);
            }
            if (!shotTaken && dream::nowSec() - opened >= opts.shotAfterSec) {
                shotTaken = true;
                dream::logf("Screenshot %s: %s", saveWindowShot(hwnd, opts.shotPath) ? "saved" : "FAILED", opts.shotPath.c_str());
            }
            ReleaseDC(hwnd, dc);
        }

        double spent = dream::nowMs() - frameStart;
        if (spent < 33.0) dream::sleepMs(int(33.0 - spent)); // ~30 fps, like dream.py
    }

    DestroyWindow(hwnd);
    ShowCursor(TRUE);
    MFShutdown();
    CoUninitialize();
}

bool dumpVideoFrame(const std::string& path, const std::string& bmpPath, int frames) {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(MFStartup(MF_VERSION))) return false;
    bool ok = false;
    {
        VideoPlayer p;
        if (p.open(path, true)) {
            double t = dream::nowMs();
            for (int i = 0; i < frames; i++) p.update(t + i * (1000.0 / 24));
            if (p.hasFrame()) ok = writeBmp(bmpPath, p.frame().px.data(), p.frame().w, p.frame().h);
        }
    }
    MFShutdown();
    return ok;
}
