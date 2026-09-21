#include "audio.h"

#include "common.h"
#include "state.h"
#include "wav.h"

#include <windows.h>
#include <mmsystem.h>

#include <algorithm>
#include <cstdlib>
#include <functional>

namespace {

constexpr int SPEECH_PEAK = 600;  // per-50ms peak (of 32768) that counts as speech

Wav g_micFile;           // --mic-file: what the "microphone" hears
size_t g_micPos = 0;     // how far through it we are (persists across recordings)
bool g_useMicFile = false;
bool g_muted = false;

// Opens the default mic at the first rate Windows accepts (it converts from
// whatever the device natively does).
bool openMic(HWAVEIN& h, int& rate) {
    for (int r : {16000, 44100, 48000, 22050, 8000}) {
        WAVEFORMATEX fmt = {};
        fmt.wFormatTag = WAVE_FORMAT_PCM;
        fmt.nChannels = 1;
        fmt.nSamplesPerSec = DWORD(r);
        fmt.wBitsPerSample = 16;
        fmt.nBlockAlign = 2;
        fmt.nAvgBytesPerSec = DWORD(r) * 2;
        if (waveInOpen(&h, WAVE_MAPPER, &fmt, 0, 0, CALLBACK_NULL) == MMSYSERR_NOERROR) {
            rate = r;
            return true;
        }
    }
    return false;
}

// Records until onChunk(samples, count, rate) returns false. Chunks are 50 ms.
MicClip record(const std::function<bool(const int16_t*, size_t, int)>& onChunk) {
    MicClip clip;
    if (g_useMicFile) { // pretend the file is a live microphone
        clip.ok = true;
        clip.rate = g_micFile.sampleRate;
        const size_t chunk = size_t(clip.rate / 20);
        std::vector<int16_t> buf(chunk);
        for (bool go = true; go && S().running;) {
            for (size_t i = 0; i < chunk; i++) buf[i] = g_micPos + i < g_micFile.samples.size() ? g_micFile.samples[g_micPos + i] : 0;
            g_micPos += chunk;
            clip.samples.insert(clip.samples.end(), buf.begin(), buf.end());
            go = onChunk(buf.data(), chunk, clip.rate);
            dream::sleepMs(50); // real time, so the timeouts behave as with a real mic
        }
        return clip;
    }
    HWAVEIN h = nullptr;
    if (!openMic(h, clip.rate)) return clip;

    const size_t chunk = size_t(clip.rate / 20);
    constexpr int kBuffers = 6;
    std::vector<std::vector<int16_t>> data(kBuffers, std::vector<int16_t>(chunk));
    WAVEHDR hdr[kBuffers] = {};
    for (int i = 0; i < kBuffers; i++) {
        hdr[i].lpData = reinterpret_cast<LPSTR>(data[size_t(i)].data());
        hdr[i].dwBufferLength = DWORD(chunk * 2);
        waveInPrepareHeader(h, &hdr[i], sizeof(WAVEHDR));
        waveInAddBuffer(h, &hdr[i], sizeof(WAVEHDR));
    }
    waveInStart(h);
    clip.ok = true;

    bool go = true;
    int next = 0;
    while (go && S().running) {
        if (!(hdr[next].dwFlags & WHDR_DONE)) { dream::sleepMs(10); continue; }
        size_t n = hdr[next].dwBytesRecorded / 2;
        const int16_t* p = data[size_t(next)].data();
        clip.samples.insert(clip.samples.end(), p, p + n);
        go = onChunk(p, n, clip.rate);
        hdr[next].dwFlags &= ~DWORD(WHDR_DONE);
        waveInAddBuffer(h, &hdr[next], sizeof(WAVEHDR));
        next = (next + 1) % kBuffers;
    }

    waveInStop(h);
    waveInReset(h);
    for (int i = 0; i < kBuffers; i++) waveInUnprepareHeader(h, &hdr[i], sizeof(WAVEHDR));
    waveInClose(h);
    return clip;
}

int peakOf(const int16_t* p, size_t n) {
    int peak = 0;
    for (size_t i = 0; i < n; i++) peak = std::max(peak, std::abs(int(p[i])));
    return peak;
}

} // namespace

int micCount() { return g_useMicFile ? 1 : int(waveInGetNumDevs()); }

MicClip recordFixed(double seconds) {
    size_t total = 0;
    return record([&](const int16_t*, size_t n, int rate) {
        total += n;
        return total < size_t(seconds * rate);
    });
}

MicClip recordCommand(double maxSeconds, double silenceSec, double noSpeechSec) {
    double t0 = dream::nowSec();
    double lastVoice = t0;
    bool heard = false;
    MicClip clip = record([&](const int16_t* p, size_t n, int) {
        double now = dream::nowSec();
        if (peakOf(p, n) > SPEECH_PEAK) { heard = true; lastVoice = now; }
        if (now - t0 >= maxSeconds) return false;
        if (heard && now - lastVoice > silenceSec) return false;
        if (!heard && now - t0 > noSpeechSec) return false;
        return true;
    });
    clip.heardSpeech = heard;
    return clip;
}

bool setMicFile(const std::string& path) {
    g_useMicFile = readWav(path, g_micFile);
    g_micPos = 0;
    return g_useMicFile;
}

void setMuted(bool muted) { g_muted = muted; }

bool playWavFile(const std::string& path) {
    if (g_muted) { dream::sleepMs(400); return true; } // long enough for the "talking" video to show
    return PlaySoundA(path.c_str(), nullptr, SND_FILENAME | SND_SYNC | SND_NODEFAULT) != FALSE;
}

void stopPlayback() { PlaySoundA(nullptr, nullptr, SND_PURGE); }
