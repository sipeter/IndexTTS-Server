// mel_extractor.cpp — Mel spectrogram from PCM audio
// Implements: load WAV/MP3 → resample to 24kHz mono → STFT → mel filterbank → log
//
// Uses dr_wav (header-only) for WAV I/O.
// Uses dr_mp3 (header-only) for MP3 decoding.
// STFT is implemented directly (no FFTW dependency) using a small radix-2 FFT.

#include "mel_extractor.h"
#include "constants.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#define DR_MP3_IMPLEMENTATION
#include "dr_mp3.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace indextts {

// ============================================================================
// Simple radix-2 FFT (enough for n_fft=1024)
// ============================================================================

static void fft(std::vector<std::complex<float>>& x) {
    const size_t N = x.size();
    if (N <= 1) return;

    // Bit-reversal permutation
    for (size_t i = 1, j = 0; i < N; ++i) {
        size_t bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }

    // Cooley-Tukey
    for (size_t len = 2; len <= N; len <<= 1) {
        float angle = -2.0f * static_cast<float>(M_PI) / static_cast<float>(len);
        std::complex<float> wlen(std::cos(angle), std::sin(angle));
        for (size_t i = 0; i < N; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (size_t j = 0; j < len / 2; ++j) {
                auto u = x[i + j];
                auto v = x[i + j + len / 2] * w;
                x[i + j]           = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// ============================================================================
// Mel filterbank
// ============================================================================

static float hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

/// Build mel filterbank matrix [n_mels × (n_fft/2+1)]
static std::vector<std::vector<float>> build_mel_filterbank(
    int n_mels, int n_fft, int sample_rate, float fmin, float fmax)
{
    int n_freqs = n_fft / 2 + 1;
    float mel_low  = hz_to_mel(fmin);
    float mel_high = hz_to_mel(fmax);

    // n_mels + 2 equally spaced points in mel scale
    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_points[i] = mel_low + (mel_high - mel_low) * i / (n_mels + 1);
    }

    // Convert to Hz then to FFT bin indices
    std::vector<float> hz_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }

    std::vector<float> bin_freqs(n_freqs);
    for (int i = 0; i < n_freqs; ++i) {
        bin_freqs[i] = static_cast<float>(i) * sample_rate / n_fft;
    }

    std::vector<std::vector<float>> filterbank(n_mels, std::vector<float>(n_freqs, 0.0f));
    for (int m = 0; m < n_mels; ++m) {
        float left   = hz_points[m];
        float center = hz_points[m + 1];
        float right  = hz_points[m + 2];
        for (int k = 0; k < n_freqs; ++k) {
            float freq = bin_freqs[k];
            if (freq >= left && freq <= center && center > left) {
                filterbank[m][k] = (freq - left) / (center - left);
            } else if (freq > center && freq <= right && right > center) {
                filterbank[m][k] = (right - freq) / (right - center);
            }
        }
    }
    return filterbank;
}

// ============================================================================
// Simple linear resampler
// ============================================================================

static std::vector<float> resample(const std::vector<float>& input, int in_sr, int out_sr) {
    if (in_sr == out_sr) return input;
    double ratio = static_cast<double>(out_sr) / in_sr;
    size_t out_len = static_cast<size_t>(input.size() * ratio);
    std::vector<float> output(out_len);
    for (size_t i = 0; i < out_len; ++i) {
        double src_idx = i / ratio;
        size_t idx = static_cast<size_t>(src_idx);
        float frac = static_cast<float>(src_idx - idx);
        if (idx + 1 < input.size()) {
            output[i] = input[idx] * (1.0f - frac) + input[idx + 1] * frac;
        } else if (idx < input.size()) {
            output[i] = input[idx];
        }
    }
    return output;
}

// ============================================================================
// Impl
// ============================================================================

struct MelExtractor::Impl {
    std::vector<float> hann_window;             // [WIN_LENGTH]
    std::vector<std::vector<float>> filterbank;  // [N_MELS × (N_FFT/2+1)]
    bool initialized = false;
};

MelExtractor::MelExtractor() : impl_(std::make_unique<Impl>()) {}
MelExtractor::~MelExtractor() = default;

void MelExtractor::init() {
    auto& d = *impl_;

    // Build Hann window
    d.hann_window.resize(WIN_LENGTH);
    for (int i = 0; i < WIN_LENGTH; ++i) {
        d.hann_window[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / WIN_LENGTH));
    }

    // Build mel filterbank
    d.filterbank = build_mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE, MEL_FMIN, MEL_FMAX);

    d.initialized = true;
}

bool MelExtractor::load_audio(const std::string& path,
                               std::vector<float>& out_pcm,
                               int& out_sr) const
{
    std::vector<float> raw;
    uint32_t channels = 0;
    uint32_t sr = 0;
    uint64_t total_frames = 0;

    // Check file extension for format detection
    std::string lower_path = path;
    std::transform(lower_path.begin(), lower_path.end(), lower_path.begin(), ::tolower);
    bool is_mp3 = (lower_path.size() >= 4 && lower_path.substr(lower_path.size() - 4) == ".mp3");

    if (is_mp3) {
        // ── MP3 via dr_mp3 ──
        drmp3 mp3;
        if (!drmp3_init_file(&mp3, path.c_str(), nullptr)) {
            return false;
        }
        total_frames = drmp3_get_pcm_frame_count(&mp3);
        channels = mp3.channels;
        sr = mp3.sampleRate;

        raw.resize(total_frames * channels);
        uint64_t read = drmp3_read_pcm_frames_f32(&mp3, total_frames, raw.data());
        drmp3_uninit(&mp3);
        if (read == 0) return false;
        total_frames = read;
    } else {
        // ── WAV via dr_wav (default) ──
        drwav wav;
        if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
            return false;
        }
        total_frames = wav.totalPCMFrameCount;
        channels = wav.channels;
        sr = wav.sampleRate;

        raw.resize(total_frames * channels);
        uint64_t read = drwav_read_pcm_frames_f32(&wav, total_frames, raw.data());
        drwav_uninit(&wav);
        if (read == 0) return false;
        total_frames = read;
    }

    // Mix to mono
    std::vector<float> mono(total_frames);
    if (channels == 1) {
        mono.assign(raw.begin(), raw.begin() + total_frames);
    } else {
        for (uint64_t i = 0; i < total_frames; ++i) {
            float sum = 0.0f;
            for (uint32_t c = 0; c < channels; ++c) {
                sum += raw[i * channels + c];
            }
            mono[i] = sum / channels;
        }
    }

    // Resample to 24kHz
    if (static_cast<int>(sr) != SAMPLE_RATE) {
        out_pcm = resample(mono, sr, SAMPLE_RATE);
    } else {
        out_pcm = std::move(mono);
    }
    out_sr = SAMPLE_RATE;
    return true;
}

MelResult MelExtractor::compute(const std::vector<float>& pcm) const {
    if (!impl_->initialized) {
        throw std::runtime_error("MelExtractor not initialized");
    }

    const auto& win = impl_->hann_window;
    const auto& fb  = impl_->filterbank;
    const int n_freqs = N_FFT / 2 + 1;

    // Center padding (reflect)
    int pad = N_FFT / 2;
    std::vector<float> padded(pcm.size() + 2 * pad);
    // Reflect left
    for (int i = 0; i < pad; ++i) {
        int src = pad - i;
        if (src >= static_cast<int>(pcm.size())) src = static_cast<int>(pcm.size()) - 1;
        padded[i] = pcm[src];
    }
    // Center
    std::memcpy(padded.data() + pad, pcm.data(), pcm.size() * sizeof(float));
    // Reflect right
    for (int i = 0; i < pad; ++i) {
        int src = static_cast<int>(pcm.size()) - 2 - i;
        if (src < 0) src = 0;
        padded[pad + pcm.size() + i] = pcm[src];
    }

    // Number of STFT frames
    int n_frames = static_cast<int>((padded.size() - N_FFT) / HOP_LENGTH) + 1;

    // STFT + Mel + Log
    MelResult result;
    result.n_mels = N_MELS;
    result.n_frames = n_frames;
    result.data.resize(N_MELS * n_frames);

    std::vector<std::complex<float>> frame_buf(N_FFT);

    for (int t = 0; t < n_frames; ++t) {
        int offset = t * HOP_LENGTH;

        // Apply window
        for (int i = 0; i < N_FFT; ++i) {
            frame_buf[i] = std::complex<float>(padded[offset + i] * win[i], 0.0f);
        }

        // FFT
        fft(frame_buf);

        // Magnitude spectrum (power=1 → magnitude, not power)
        std::vector<float> mag(n_freqs);
        for (int k = 0; k < n_freqs; ++k) {
            mag[k] = std::abs(frame_buf[k]);
        }

        // Apply mel filterbank + log
        for (int m = 0; m < N_MELS; ++m) {
            float val = 0.0f;
            for (int k = 0; k < n_freqs; ++k) {
                val += fb[m][k] * mag[k];
            }
            val = std::log(std::max(val, MEL_LOG_CLIP));
            result.at(m, t) = val;
        }
    }

    return result;
}

} // namespace indextts
