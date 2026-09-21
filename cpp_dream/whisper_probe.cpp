// whisper_probe.cpp - transcribe a WAV file with whisper.cpp. Used to check the
// Whisper build and model work: whisper_probe <model.bin> <file.wav>
#include "wav.h"

#include <whisper.h>

#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: whisper_probe <model.bin> <file.wav>\n"); return 2; }
    Wav wav;
    if (!readWav(argv[2], wav)) { std::printf("could not read %s\n", argv[2]); return 1; }
    std::vector<float> pcm = toWhisperInput(wav.samples, wav.sampleRate);

    whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = false;
    whisper_context* ctx = whisper_init_from_file_with_params(argv[1], cp);
    if (!ctx) { std::printf("could not load model\n"); return 1; }

    whisper_full_params p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    p.language = "en";
    p.print_progress = p.print_realtime = p.print_timestamps = p.print_special = false;
    p.n_threads = 4;
    if (whisper_full(ctx, p, pcm.data(), int(pcm.size())) != 0) { std::printf("transcribe failed\n"); return 1; }

    std::string text;
    for (int i = 0; i < whisper_full_n_segments(ctx); i++) text += whisper_full_get_segment_text(ctx, i);
    std::printf("TEXT:%s\n", text.c_str());
    whisper_free(ctx);
    return 0;
}
