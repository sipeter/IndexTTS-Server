#pragma once
// Mel spectrogram extraction from audio waveform

#include <memory>
#include <string>
#include <vector>

namespace indextts {

/// 2D float buffer stored as [rows × cols] in row-major order
struct MelResult {
    std::vector<float> data;   // row-major: [n_mels × n_frames]
    int n_mels  = 0;
    int n_frames = 0;

    /// Access element at (mel_bin, frame)
    float& at(int mel, int frame) { return data[mel * n_frames + frame]; }
    float  at(int mel, int frame) const { return data[mel * n_frames + frame]; }
};

class MelExtractor {
public:
    MelExtractor();
    ~MelExtractor();

    /// Initialize: build mel filterbank, Hann window.
    void init();

    /// Load audio from WAV file, resample to 24kHz mono.
    /// @param path       WAV file path
    /// @param out_pcm    Output float PCM [-1, 1]
    /// @param out_sr     Output sample rate (always 24000 after resample)
    /// @return true on success
    bool load_audio(const std::string& path,
                    std::vector<float>& out_pcm,
                    int& out_sr) const;

    /// Compute log-mel spectrogram from PCM audio.
    /// Output shape: [n_mels=100, T] stored in MelResult.
    /// T = ceil(wav_samples / hop_length) + 1
    MelResult compute(const std::vector<float>& pcm) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace indextts
