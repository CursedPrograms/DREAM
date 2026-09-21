#include "wav.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace {

uint32_t rd32(const unsigned char* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (uint32_t(p[3]) << 24); }
uint16_t rd16(const unsigned char* p) { return uint16_t(p[0] | (p[1] << 8)); }

void wr32(std::ofstream& f, uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); }
void wr16(std::ofstream& f, uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); }

} // namespace

bool readWav(const std::string& path, Wav& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<unsigned char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (data.size() < 44 || std::memcmp(data.data(), "RIFF", 4) != 0 || std::memcmp(data.data() + 8, "WAVE", 4) != 0) return false;

    int channels = 1, bits = 16;
    size_t pos = 12;
    const unsigned char* pcm = nullptr;
    size_t pcmBytes = 0;
    while (pos + 8 <= data.size()) {
        uint32_t size = rd32(&data[pos + 4]);
        const unsigned char* body = &data[pos + 8];
        if (std::memcmp(&data[pos], "fmt ", 4) == 0 && size >= 16 && pos + 8 + size <= data.size()) {
            channels = rd16(body + 2);
            out.sampleRate = int(rd32(body + 4));
            bits = rd16(body + 14);
        } else if (std::memcmp(&data[pos], "data", 4) == 0) {
            pcm = body;
            pcmBytes = std::min<size_t>(size, data.size() - (pos + 8)); // streamed WAVs may claim more than exists
            break;
        }
        pos += 8 + size + (size & 1);
    }
    if (!pcm || bits != 16 || channels < 1 || out.sampleRate <= 0) return false;

    size_t frames = pcmBytes / (2 * size_t(channels));
    out.samples.resize(frames);
    for (size_t i = 0; i < frames; i++) {
        int sum = 0;
        for (int c = 0; c < channels; c++) sum += int16_t(rd16(pcm + (i * channels + c) * 2));
        out.samples[i] = int16_t(sum / channels);
    }
    return true;
}

bool writeWav(const std::string& path, const std::vector<int16_t>& samples, int sampleRate) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t bytes = uint32_t(samples.size() * 2);
    f.write("RIFF", 4); wr32(f, 36 + bytes); f.write("WAVE", 4);
    f.write("fmt ", 4); wr32(f, 16); wr16(f, 1); wr16(f, 1);
    wr32(f, uint32_t(sampleRate)); wr32(f, uint32_t(sampleRate) * 2); wr16(f, 2); wr16(f, 16);
    f.write("data", 4); wr32(f, bytes);
    f.write(reinterpret_cast<const char*>(samples.data()), bytes);
    return bool(f);
}

std::vector<float> toWhisperInput(const std::vector<int16_t>& samples, int rate) {
    const int target = 16000;
    std::vector<float> out;
    if (samples.empty() || rate <= 0) return out;
    if (rate == target) {
        out.reserve(samples.size());
        for (int16_t s : samples) out.push_back(s / 32768.0f);
        return out;
    }
    size_t n = size_t(double(samples.size()) * target / rate);
    out.resize(n);
    double step = double(rate) / target;
    for (size_t i = 0; i < n; i++) {
        double src = i * step;
        size_t i0 = size_t(src);
        size_t i1 = std::min(i0 + 1, samples.size() - 1);
        double frac = src - double(i0);
        out[i] = float((samples[i0] * (1.0 - frac) + samples[i1] * frac) / 32768.0);
    }
    return out;
}

int peakLevel(const std::vector<int16_t>& samples) {
    int peak = 0;
    for (int16_t s : samples) peak = std::max(peak, std::abs(int(s)));
    return peak;
}
