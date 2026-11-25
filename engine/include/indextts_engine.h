#pragma once
// IndexTTS ONNX Inference Engine — main orchestrator

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace indextts {

/// Generation parameters (all optional — defaults from constants.h)
struct GenerateParams {
    float temperature       = 1.0f;
    float top_p             = 0.8f;
    int   top_k             = 30;
    float repetition_penalty= 10.0f;
    int   max_generate_length = 600;
    bool  enable_fast_mode  = false;
    bool  enable_preterminal_collapse_stop = false;
    int   early_stop_trailing_silence = 0;
    int   early_stop_min_generated_codes = 0;
    int   pause_trim_trigger_total_silent = 30;
    int   pause_trim_keep_silence = 10;
};

struct TimingEntry {
    std::string label;
    double ms = 0.0;
    int steps = 0;
};

struct TimingReport {
    std::vector<TimingEntry> stages;
    double pipeline_ms = 0.0;
    double write_wav_ms = 0.0;
    double total_ms = 0.0;
    int total_decode_steps = 0;
    int audio_samples = 0;
    int sample_rate = 0;
};

/// Progress callback: (current_sentence, total_sentences, tokens_generated)
using ProgressCallback = std::function<void(int, int, int)>;

class IndexTTSEngine {
public:
    IndexTTSEngine();
    ~IndexTTSEngine();

    /// Initialize the engine.
    /// @param model_dir    Directory containing: cond_encoder.onnx, bigvgan.onnx,
    ///                     gpt_prefill.onnx, gpt_decode.onnx, gpt_latent.onnx,
    ///                     gpt_config.json, embeddings/
    /// @param bpe_model    Path to bpe.model
    /// @param use_cuda     Enable CUDA EP (requires CUDA-enabled ONNXRuntime)
    /// @param num_threads  Number of intra-op threads (0 = auto)
    /// @return true on success
    bool init(const std::string& model_dir,
              const std::string& bpe_model,
              bool use_cuda = false,
              bool use_tensorrt = false,
              const std::string& trt_cache_dir = "",
              int num_threads = 0);

    /// Synthesize speech from text using a reference audio prompt.
    /// @param ref_audio_path   Path to reference WAV file
    /// @param text             Input text (UTF-8)
    /// @param output_path      Output WAV file path
    /// @param params           Generation parameters
    /// @param callback         Optional progress callback
    /// @return true on success
    bool synthesize(const std::string& ref_audio_path,
                    const std::string& text,
                    const std::string& output_path,
                    const GenerateParams& params = {},
                    ProgressCallback callback = nullptr);

    /// Synthesize and return raw PCM instead of saving to file.
    /// @return float PCM at 24kHz, mono
    std::vector<float> synthesize_pcm(const std::string& ref_audio_path,
                                       const std::string& text,
                                       const GenerateParams& params = {},
                                       ProgressCallback callback = nullptr);

    TimingReport get_last_timing() const;

    /// Check if engine is initialized.
    bool is_ready() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace indextts
