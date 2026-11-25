// indextts_engine.cpp — Main ONNX inference pipeline for IndexTTS
//
// Pipeline:
//   1. Load ref audio → mel spectrogram → cond_encoder → cond [1,32,1280]
//   2. Text → normalize → CJK → BPE → split sentences
//   3. Per sentence: build embeddings → GPT prefill → GPT decode loop → mel codes
//   4. Silence removal
//   5. Latent extraction via gpt_prefill hidden_states → latent [1,T,1280]
//   6. BigVGAN vocoder → wav
//   7. Concatenate all sentence wavs → output WAV

#include "indextts_engine.h"
#include "constants.h"
#include "mel_extractor.h"
#include "npy_reader.h"
#include "text_processor.h"

#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

// dr_wav for writing output WAV
#include "dr_wav.h"

#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace indextts {

// ============================================================================
// ONNX helper: create tensor from vector
// ============================================================================

static Ort::Value create_tensor_float(Ort::MemoryInfo& mem,
                                       std::vector<float>& data,
                                       const std::vector<int64_t>& shape) {
    return Ort::Value::CreateTensor<float>(mem, data.data(), data.size(),
                                            shape.data(), shape.size());
}

static Ort::Value create_tensor_int64(Ort::MemoryInfo& mem,
                                       std::vector<int64_t>& data,
                                       const std::vector<int64_t>& shape) {
    return Ort::Value::CreateTensor<int64_t>(mem, data.data(), data.size(),
                                              shape.data(), shape.size());
}

// ============================================================================
// Sampling utilities
// ============================================================================

/// Apply repetition penalty to logits in-place
static void apply_repetition_penalty(float* logits, int vocab_size,
                                      const std::vector<int64_t>& generated_ids,
                                      float penalty) {
    for (auto id : generated_ids) {
        if (id >= 0 && id < vocab_size) {
            if (logits[id] > 0) {
                logits[id] /= penalty;
            } else {
                logits[id] *= penalty;
            }
        }
    }
}

/// Apply temperature
static void apply_temperature(float* logits, int vocab_size, float temperature) {
    if (temperature <= 0.0f || temperature == 1.0f) return;
    for (int i = 0; i < vocab_size; ++i) {
        logits[i] /= temperature;
    }
}

/// Top-K + Top-P sampling. Returns sampled token id.
static int sample_top_k_top_p(float* logits, int vocab_size,
                                int top_k, float top_p,
                                std::mt19937& rng) {
    // Find top-K indices
    std::vector<std::pair<float, int>> scored(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        scored[i] = {logits[i], i};
    }

    // Partial sort for top-K
    if (top_k > 0 && top_k < vocab_size) {
        std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
        scored.resize(top_k);
    } else {
        std::sort(scored.begin(), scored.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });
    }

    // Softmax over candidates
    float max_val = scored[0].first;
    std::vector<float> probs(scored.size());
    float sum = 0.0f;
    for (size_t i = 0; i < scored.size(); ++i) {
        probs[i] = std::exp(scored[i].first - max_val);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;

    // Top-P (nucleus) filtering
    float cumsum = 0.0f;
    int cutoff = static_cast<int>(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i];
        if (cumsum >= top_p) {
            cutoff = static_cast<int>(i) + 1;
            break;
        }
    }

    // Renormalize
    float re_sum = 0.0f;
    for (int i = 0; i < cutoff; ++i) re_sum += probs[i];
    for (int i = 0; i < cutoff; ++i) probs[i] /= re_sum;

    // Sample
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float acc = 0.0f;
    for (int i = 0; i < cutoff; ++i) {
        acc += probs[i];
        if (r <= acc) {
            return scored[i].second;
        }
    }
    return scored[cutoff - 1].second;
}

// ============================================================================
// Silence removal
// ============================================================================

struct SilenceRun {
    int start = 0;
    int length = 0;
};

struct TokenCandidate {
    int id = -1;
    float prob = 0.0f;
};

struct TokenDistributionSnapshot {
    int after_generated_codes = 0;
    int sampled_token = -1;
    int top_token = -1;
    float top_prob = 0.0f;
    int top2_token = -1;
    float top2_prob = 0.0f;
    float top12_margin = 0.0f;
    int stop_rank = -1;
    float stop_prob = 0.0f;
    int silent_rank = -1;
    float silent_prob = 0.0f;
    float entropy = 0.0f;
    std::vector<TokenCandidate> top_candidates;
};

struct LookaheadProbeArm {
    bool active = false;
    int after_generated_codes = -1;
    int sampled_token = -1;
    float top_prob = 0.0f;
    float top12_margin = 0.0f;
    float entropy = 0.0f;
};

static void log_engine_message(const std::string& message) {
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "IndexTTS", "%s", message.c_str());
#endif
    std::cerr << message << std::endl;
}

static std::string format_silence_runs(const std::vector<SilenceRun>& silence_runs) {
    if (silence_runs.empty()) {
        return "none";
    }

    std::ostringstream oss;
    for (size_t i = 0; i < silence_runs.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << silence_runs[i].start << "+" << silence_runs[i].length;
    }
    return oss.str();
}

static TokenDistributionSnapshot inspect_token_distribution(
    const float* logits,
    int vocab_size,
    int after_generated_codes,
    int top_n = 3) {
    TokenDistributionSnapshot snapshot;
    snapshot.after_generated_codes = after_generated_codes;

    if (vocab_size <= 0) {
        return snapshot;
    }

    float max_logit = logits[0];
    int top_token = 0;
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            top_token = i;
        }
    }

    float exp_sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        exp_sum += std::exp(logits[i] - max_logit);
    }
    if (exp_sum <= 0.0f) {
        exp_sum = 1.0f;
    }

    snapshot.top_token = top_token;
    snapshot.top_prob = std::exp(logits[top_token] - max_logit) / exp_sum;

    auto inspect_target = [&](int token_id, int* rank_out, float* prob_out) {
        if (token_id < 0 || token_id >= vocab_size) {
            *rank_out = -1;
            *prob_out = 0.0f;
            return;
        }

        int rank = 1;
        float token_logit = logits[token_id];
        for (int i = 0; i < vocab_size; ++i) {
            if (logits[i] > token_logit) {
                ++rank;
            }
        }

        *rank_out = rank;
        *prob_out = std::exp(token_logit - max_logit) / exp_sum;
    };

    inspect_target(STOP_MEL_TOKEN, &snapshot.stop_rank, &snapshot.stop_prob);
    inspect_target(SILENT_TOKEN, &snapshot.silent_rank, &snapshot.silent_prob);

    std::vector<std::pair<float, int>> scored(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        scored[i] = {logits[i], i};
    }

    int candidate_count = std::min(top_n, vocab_size);
    if (candidate_count > 0) {
        std::partial_sort(
            scored.begin(),
            scored.begin() + candidate_count,
            scored.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        snapshot.top_candidates.reserve(candidate_count);
        for (int i = 0; i < candidate_count; ++i) {
            float prob = std::exp(scored[i].first - max_logit) / exp_sum;
            snapshot.top_candidates.push_back({scored[i].second, prob});
            if (i == 1) {
                snapshot.top2_token = scored[i].second;
                snapshot.top2_prob = prob;
            }
        }
    }

    if (snapshot.top2_token >= 0) {
        snapshot.top12_margin = snapshot.top_prob - snapshot.top2_prob;
    }

    for (int i = 0; i < vocab_size; ++i) {
        float prob = std::exp(logits[i] - max_logit) / exp_sum;
        if (prob > 0.0f) {
            snapshot.entropy -= prob * std::log(prob);
        }
    }

    return snapshot;
}

static std::string format_token_candidates(const std::vector<TokenCandidate>& candidates) {
    if (candidates.empty()) {
        return "none";
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (i != 0) {
            oss << "|";
        }
        oss << candidates[i].id << ":" << candidates[i].prob;
    }
    return oss.str();
}

static std::string format_distribution_snapshot(const TokenDistributionSnapshot& snapshot) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "after" << snapshot.after_generated_codes
        << "/sampled=" << snapshot.sampled_token
        << "/top1=" << snapshot.top_token << ":" << snapshot.top_prob
        << "/top2=" << snapshot.top2_token << ":" << snapshot.top2_prob
        << "/margin=" << snapshot.top12_margin
        << "/stop=" << snapshot.stop_prob << "#" << snapshot.stop_rank
        << "/entropy=" << snapshot.entropy;
    return oss.str();
}

static bool is_preterminal_collapse_candidate(
    const TokenDistributionSnapshot& snapshot,
    int sampled_token,
    int min_generated_codes) {
    constexpr float kLookaheadProbeMinTopProb = 0.999f;
    constexpr float kLookaheadProbeMinMargin = 0.999f;
    constexpr float kLookaheadProbeMaxEntropy = 0.001f;

    return min_generated_codes > 0 &&
           snapshot.after_generated_codes >= min_generated_codes &&
           sampled_token == snapshot.top_token &&
           sampled_token != STOP_MEL_TOKEN &&
           sampled_token != SILENT_TOKEN &&
           snapshot.top_prob >= kLookaheadProbeMinTopProb &&
           snapshot.top12_margin >= kLookaheadProbeMinMargin &&
           snapshot.entropy <= kLookaheadProbeMaxEntropy;
}

static std::vector<int64_t> remove_long_silence(
    const std::vector<int64_t>& codes,
    int trim_trigger_total_silent = MAX_CONSECUTIVE_SILENCE,
    int keep_silence = KEEP_SILENCE) {
    if (trim_trigger_total_silent <= 0) {
        trim_trigger_total_silent = MAX_CONSECUTIVE_SILENCE;
    }
    if (keep_silence < 0) {
        keep_silence = KEEP_SILENCE;
    }

    // Find effective length (up to stop_mel_token)
    size_t eff_len = codes.size();
    for (size_t i = 0; i < codes.size(); ++i) {
        if (codes[i] == STOP_MEL_TOKEN) {
            eff_len = i;
            break;
        }
    }

    // Count consecutive silent tokens
    int total_silent = 0;
    for (size_t i = 0; i < eff_len; ++i) {
        if (codes[i] == SILENT_TOKEN) ++total_silent;
    }

    if (total_silent <= trim_trigger_total_silent) {
        return std::vector<int64_t>(codes.begin(), codes.begin() + eff_len);
    }

    // Rebuild: keep non-silent, allow max KEEP_SILENCE consecutive silent
    std::vector<int64_t> result;
    result.reserve(eff_len);
    int consecutive = 0;
    for (size_t i = 0; i < eff_len; ++i) {
        if (codes[i] != SILENT_TOKEN) {
            result.push_back(codes[i]);
            consecutive = 0;
        } else if (consecutive < keep_silence) {
            result.push_back(codes[i]);
            ++consecutive;
        }
        // else: skip excessive silence
    }
    return result;
}

// ============================================================================
// WAV writer helper
// ============================================================================

static bool write_wav(const std::string& path, const std::vector<float>& pcm, int sample_rate) {
    drwav wav;
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_PCM;
    format.channels = 1;
    format.sampleRate = static_cast<drwav_uint32>(sample_rate);
    format.bitsPerSample = 16;

    if (!drwav_init_file_write(&wav, path.c_str(), &format, nullptr)) {
        return false;
    }

    // Convert float to int16
    std::vector<int16_t> pcm16(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i) {
        float v = pcm[i] * WAV_SCALE;
        v = std::max(-WAV_SCALE, std::min(WAV_SCALE, v));
        pcm16[i] = static_cast<int16_t>(v);
    }

    drwav_write_pcm_frames(&wav, pcm.size(), pcm16.data());
    drwav_uninit(&wav);
    return true;
}

// ============================================================================
// Engine Implementation
// ============================================================================

struct IndexTTSEngine::Impl {
    // ONNX Runtime
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "indextts"};
    Ort::SessionOptions session_opts;
    std::unique_ptr<Ort::Session> sess_cond_encoder;
    std::unique_ptr<Ort::Session> sess_gpt_prefill;
    std::unique_ptr<Ort::Session> sess_gpt_decode;
    std::unique_ptr<Ort::Session> sess_bigvgan;
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Embedding tables
    NpyArray text_emb;       // [12001, 1280]
    NpyArray mel_emb;        // [8194, 1280]
    NpyArray text_pos_emb;   // [602, 1280]
    NpyArray mel_pos_emb;    // [803, 1280]

    // Sub-modules
    TextProcessor text_proc;
    MelExtractor mel_ext;

    // RNG
    std::mt19937 rng{42};

    // KV-cache format: true = combined [2,B,H,S,Dh], false = separate key/value
    bool combined_kv = false;

    TimingReport last_timing;

    bool ready = false;
};

IndexTTSEngine::IndexTTSEngine() : impl_(std::make_unique<Impl>()) {}
IndexTTSEngine::~IndexTTSEngine() = default;

bool IndexTTSEngine::is_ready() const { return impl_->ready; }

TimingReport IndexTTSEngine::get_last_timing() const { return impl_->last_timing; }

bool IndexTTSEngine::init(const std::string& model_dir,
                           const std::string& bpe_model,
                           bool use_cuda,
                           bool use_tensorrt,
                           const std::string& trt_cache_dir,
                           int num_threads) {
    auto& d = *impl_;

    // Session options
    if (num_threads > 0) {
        d.session_opts.SetIntraOpNumThreads(num_threads);
    }

#ifdef INDEXTTS_USE_CUDA
    // GPU path: enable all optimizations including Gelu fusion (GPU supports FP16 Gelu)
    d.session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#else
    // CPU-only path: use BASIC optimization to avoid Gelu fusion with FP16 tensors
    // (CPU EP doesn't support com.microsoft.Gelu with float16)
    d.session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
#endif

#ifdef INDEXTTS_USE_CUDA
    if (use_tensorrt) {
        // TensorRT EP (preferred) — falls back to CUDA EP for unsupported ops
        OrtTensorRTProviderOptions trt_opts{};
        trt_opts.trt_fp16_enable = 1;
        trt_opts.trt_engine_cache_enable = 1;
        trt_opts.trt_max_workspace_size = (size_t)2 << 30;  // 2GB
        std::string cache_dir = trt_cache_dir.empty() ? model_dir + "/trt_cache" : trt_cache_dir;
        std::filesystem::create_directories(cache_dir);
        trt_opts.trt_engine_cache_path = cache_dir.c_str();
        std::cerr << "[IndexTTS] TensorRT EP enabled (FP16, cache: " << cache_dir << ")" << std::endl;
        d.session_opts.AppendExecutionProvider_TensorRT(trt_opts);
        // Also add CUDA EP as fallback
        OrtCUDAProviderOptions cuda_opts{};
        d.session_opts.AppendExecutionProvider_CUDA(cuda_opts);
    } else if (use_cuda) {
        OrtCUDAProviderOptions cuda_opts{};
        d.session_opts.AppendExecutionProvider_CUDA(cuda_opts);
        std::cerr << "[IndexTTS] CUDA EP enabled" << std::endl;
    }
#else
    // 编译期未开 USE_CUDA 时 --cuda/--tensorrt 只能落到 CPU：必须显式告警，
    // 否则表现为"参数带了却慢一个数量级"的静默降级（20260712 便携包事故）
    if (use_cuda || use_tensorrt) {
        std::cerr << "[IndexTTS] WARNING: --cuda/--tensorrt requested but this binary "
                     "was built WITHOUT CUDA support (USE_CUDA=OFF). Running on CPU!" << std::endl;
    }
    (void)use_cuda;
    (void)use_tensorrt;
    (void)trt_cache_dir;
#endif

    try {
        // Load ONNX models
        auto load_model = [&](const std::string& name) -> std::unique_ptr<Ort::Session> {
            std::string path = model_dir + "/" + name;
#ifdef _WIN32
            // Convert to wstring for Windows
            std::wstring wpath(path.begin(), path.end());
            return std::make_unique<Ort::Session>(d.env, wpath.c_str(), d.session_opts);
#else
            return std::make_unique<Ort::Session>(d.env, path.c_str(), d.session_opts);
#endif
        };

        std::cerr << "[IndexTTS] Loading cond_encoder.onnx..." << std::endl;
        d.sess_cond_encoder = load_model("cond_encoder.onnx");

        std::cerr << "[IndexTTS] Loading gpt_prefill.onnx..." << std::endl;
        d.sess_gpt_prefill = load_model("gpt_prefill.onnx");

        // Prefer gpt_decode_opt.onnx (ORT-optimized with Attention fusion) if available
        std::string decode_model_name = "gpt_decode.onnx";
        if (std::filesystem::exists(model_dir + "/gpt_decode_opt.onnx")) {
            decode_model_name = "gpt_decode_opt.onnx";
        }
        std::cerr << "[IndexTTS] Loading " << decode_model_name << "..." << std::endl;
        d.sess_gpt_decode = load_model(decode_model_name);

        // Auto-detect KV-cache format by checking decode model input names
        {
            Ort::AllocatorWithDefaultOptions alloc;
            size_t num_inputs = d.sess_gpt_decode->GetInputCount();
            for (size_t i = 0; i < num_inputs; ++i) {
                auto name = d.sess_gpt_decode->GetInputNameAllocated(i, alloc);
                if (std::string(name.get()) == "past_0") {
                    d.combined_kv = true;
                    break;
                }
            }
            std::cerr << "[IndexTTS] KV-cache format: "
                      << (d.combined_kv ? "combined [2,B,H,S,Dh]" : "separate key/value")
                      << std::endl;
        }

        std::cerr << "[IndexTTS] Loading bigvgan.onnx..." << std::endl;
        d.sess_bigvgan = load_model("bigvgan.onnx");

        // Load embeddings
        std::string emb_dir = model_dir + "/embeddings/";
        std::cerr << "[IndexTTS] Loading embeddings..." << std::endl;
        d.text_emb     = load_npy_float32(emb_dir + "text_embedding.npy");
        d.mel_emb      = load_npy_float32(emb_dir + "mel_embedding.npy");
        d.text_pos_emb = load_npy_float32(emb_dir + "text_pos_embedding.npy");
        d.mel_pos_emb  = load_npy_float32(emb_dir + "mel_pos_embedding.npy");

        // Load BPE model
        std::cerr << "[IndexTTS] Loading BPE model..." << std::endl;
        if (!d.text_proc.load(bpe_model)) {
            std::cerr << "[IndexTTS] Failed to load BPE model: " << bpe_model << std::endl;
            return false;
        }

        // Initialize mel extractor
        d.mel_ext.init();

        d.ready = true;
        std::cerr << "[IndexTTS] Engine ready!" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[IndexTTS] Init error: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> IndexTTSEngine::synthesize_pcm(
    const std::string& ref_audio_path,
    const std::string& text,
    const GenerateParams& params,
    ProgressCallback callback)
{
    auto& d = *impl_;
    if (!d.ready) {
        throw std::runtime_error("Engine not initialized");
    }

    d.last_timing = TimingReport{};
    auto& timing = d.last_timing;
    timing.sample_rate = SAMPLE_RATE;

    auto t_total_start = std::chrono::steady_clock::now();
    auto t_stage = t_total_start;
    auto ms_since = [](auto start) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    };
    auto stage_label = [](const char* base, int sentence_index, int sentence_count) {
        if (sentence_count <= 1) {
            return std::string(base);
        }
        std::ostringstream oss;
        oss << "sentence_" << sentence_index << "." << base;
        return oss.str();
    };
    auto record_stage = [&](const std::string& label, auto start, int steps = 0) {
        const double elapsed_ms = ms_since(start);
        timing.stages.push_back(TimingEntry{label, elapsed_ms, steps});
        if (steps > 0) {
            timing.total_decode_steps += steps;
        }
        std::cerr << "[TIMING] " << label;
        if (steps > 0) {
            std::cerr << " (" << steps << " steps)";
        }
        std::cerr << ": " << elapsed_ms << "ms" << std::endl;
    };

    // ================================================================
    // Step 1: Reference audio → condition latents
    // ================================================================
    std::vector<float> ref_pcm;
    int ref_sr;
    if (!d.mel_ext.load_audio(ref_audio_path, ref_pcm, ref_sr)) {
        throw std::runtime_error("Failed to load reference audio: " + ref_audio_path);
    }

    MelResult ref_mel = d.mel_ext.compute(ref_pcm);
    // ref_mel: [n_mels=100, T_ref] row-major → need [1, 100, T_ref] for cond_encoder

    std::vector<float> mel_input(ref_mel.data);  // already [100, T_ref]
    std::vector<int64_t> mel_shape = {1, N_MELS, ref_mel.n_frames};

    // Run cond_encoder
    std::vector<const char*> cond_enc_in_names = {"mel"};
    std::vector<const char*> cond_enc_out_names = {"cond"};

    auto mel_tensor = create_tensor_float(d.mem_info, mel_input, mel_shape);
    auto cond_out = d.sess_cond_encoder->Run(
        Ort::RunOptions{nullptr},
        cond_enc_in_names.data(), &mel_tensor, 1,
        cond_enc_out_names.data(), 1);

    // cond: [1, 32, 1280]
    float* cond_data = cond_out[0].GetTensorMutableData<float>();
    size_t cond_size = COND_NUM_LATENTS * MODEL_DIM;
    std::vector<float> cond(cond_data, cond_data + cond_size);

    record_stage("cond_encoder", t_stage);
    t_stage = std::chrono::steady_clock::now();

    // Also prepare mel_ref for BigVGAN: [1, T_ref, 100] (transposed from [100, T_ref])
    std::vector<float> mel_ref_t(ref_mel.n_frames * N_MELS);
    for (int t = 0; t < ref_mel.n_frames; ++t) {
        for (int m = 0; m < N_MELS; ++m) {
            mel_ref_t[t * N_MELS + m] = ref_mel.at(m, t);
        }
    }

    // ================================================================  
    // Step 2: Text processing
    // ================================================================
    auto sentences = d.text_proc.process(text, 100);
    if (sentences.empty()) {
        throw std::runtime_error("No text to synthesize");
    }

    record_stage("text_processing", t_stage);
    t_stage = std::chrono::steady_clock::now();

    std::cerr << "[IndexTTS] " << sentences.size() << " sentence(s) to synthesize" << std::endl;

    // ================================================================
    // Step 3-5: Per-sentence generation → latent extraction
    // Collect all latents first, then batch through BigVGAN (like Python).
    // ================================================================
    struct SentenceLatent {
        std::vector<float> data;  // [T, MODEL_DIM] flattened
        int T;                    // number of latent frames
    };
    std::vector<SentenceLatent> all_latents;

    for (int si = 0; si < static_cast<int>(sentences.size()); ++si) {
        const auto& token_ids = sentences[si];
        std::cerr << "[IndexTTS] Sentence " << (si + 1) << "/" << sentences.size()
                  << " (" << token_ids.size() << " tokens)" << std::endl;

        // ── Build prefill input embeddings ────────────────────────
        // For autoregressive generation: text = [start_text, tokens..., stop_text]
        // (1 stop — different from latent extraction which uses 2 stops)
        std::vector<int32_t> text_input;
        text_input.push_back(START_TEXT_TOKEN);
        text_input.insert(text_input.end(), token_ids.begin(), token_ids.end());
        text_input.push_back(STOP_TEXT_TOKEN);

        int text_len = static_cast<int>(text_input.size());
        int prefill_seq = COND_NUM_LATENTS + text_len + 1;  // +1 for start_mel

        // Build inputs_embeds: [1, prefill_seq, 1280]
        std::vector<float> inputs_embeds(prefill_seq * MODEL_DIM, 0.0f);

        // Position 0..31: cond (already [32, 1280])
        std::memcpy(inputs_embeds.data(), cond.data(), cond_size * sizeof(float));

        // Position 32..32+text_len-1: text_embedding[id] + text_pos_embedding[pos]
        for (int i = 0; i < text_len; ++i) {
            int token = text_input[i];
            int out_pos = COND_NUM_LATENTS + i;
            const float* te = d.text_emb.row_ptr(token);
            const float* tp = d.text_pos_emb.row_ptr(i);
            float* dst = inputs_embeds.data() + out_pos * MODEL_DIM;
            for (int d_i = 0; d_i < MODEL_DIM; ++d_i) {
                dst[d_i] = te[d_i] + tp[d_i];
            }
        }

        // Last position: mel_embedding[start_mel] + mel_pos_embedding[0]
        {
            int out_pos = COND_NUM_LATENTS + text_len;
            const float* me = d.mel_emb.row_ptr(START_MEL_TOKEN);
            const float* mp = d.mel_pos_emb.row_ptr(0);
            float* dst = inputs_embeds.data() + out_pos * MODEL_DIM;
            for (int d_i = 0; d_i < MODEL_DIM; ++d_i) {
                dst[d_i] = me[d_i] + mp[d_i];
            }
        }

        std::vector<int64_t> embeds_shape = {1, prefill_seq, MODEL_DIM};

        // ── GPT Prefill ──────────────────────────────────────────
        std::vector<const char*> prefill_in_names = {"inputs_embeds"};

        // Build output names: logits + hidden_states + 24*2 KV
        std::vector<std::string> prefill_out_name_strs;
        prefill_out_name_strs.push_back("logits");
        prefill_out_name_strs.push_back("hidden_states");
        for (int l = 0; l < NUM_LAYERS; ++l) {
            prefill_out_name_strs.push_back("present_key_" + std::to_string(l));
            prefill_out_name_strs.push_back("present_value_" + std::to_string(l));
        }
        std::vector<const char*> prefill_out_names;
        for (auto& s : prefill_out_name_strs) prefill_out_names.push_back(s.c_str());

        auto embeds_tensor = create_tensor_float(d.mem_info, inputs_embeds, embeds_shape);

        auto prefill_out = d.sess_gpt_prefill->Run(
            Ort::RunOptions{nullptr},
            prefill_in_names.data(), &embeds_tensor, 1,
            prefill_out_names.data(), prefill_out_names.size());

        record_stage(stage_label("gpt_prefill", si + 1, static_cast<int>(sentences.size())), t_stage);
        t_stage = std::chrono::steady_clock::now();

        // Extract last-position logits for first decode step
        float* prefill_logits = prefill_out[0].GetTensorMutableData<float>();
        // logits shape: [1, prefill_seq, VOCAB_MEL]
        // We need logits at position prefill_seq - 1
        float* last_logits = prefill_logits + (prefill_seq - 1) * VOCAB_MEL;

        // KV-cache: prefill_out[2..49] are [1, 20, prefill_seq, 64]
        // (index 0 = logits, index 1 = hidden_states, index 2+ = KV-cache)

        // ── GPT Decode Loop ─────────────────────────────────────
        std::vector<int64_t> generated_codes;
        int max_gen = params.max_generate_length;

        // Sample first token from prefill logits
        std::vector<float> logits_buf(VOCAB_MEL);
        std::memcpy(logits_buf.data(), last_logits, VOCAB_MEL * sizeof(float));
        std::vector<SilenceRun> silence_runs;
        int trailing_silence = 0;
        int current_silence_run_start = -1;
        int max_silence_run = 0;
        int max_eligible_trailing_silence = 0;
        std::string stop_reason = "max_generate_length";
        std::vector<TokenDistributionSnapshot> tail_trace;
        LookaheadProbeArm lookahead_probe_arm;
        int lookahead_probe_candidate_count = 0;
        int lookahead_probe_next_step_stop_count = 0;
        TokenDistributionSnapshot last_distribution;
        bool have_last_distribution = false;
        TokenDistributionSnapshot best_eligible_stop_distribution;
        bool have_best_eligible_stop_distribution = false;
        TokenDistributionSnapshot best_preterminal_stop_distribution;
        bool have_best_preterminal_stop_distribution = false;
        TokenDistributionSnapshot best_eligible_silent_distribution;
        bool have_best_eligible_silent_distribution = false;
        TokenDistributionSnapshot last_non_stop_distribution;
        bool have_last_non_stop_distribution = false;
        float best_eligible_stop_prob = -1.0f;
        int best_eligible_stop_rank = std::numeric_limits<int>::max();
        float best_preterminal_stop_prob = -1.0f;
        int best_preterminal_stop_rank = std::numeric_limits<int>::max();
        float best_eligible_silent_prob = -1.0f;
        int best_eligible_silent_rank = std::numeric_limits<int>::max();

        // Build decode I/O names (depends on KV-cache format)
        std::vector<std::string> decode_in_name_strs;
        decode_in_name_strs.push_back("input_ids");
        decode_in_name_strs.push_back("position_ids");
        if (d.combined_kv) {
            for (int l = 0; l < NUM_LAYERS; ++l) {
                decode_in_name_strs.push_back("past_" + std::to_string(l));
            }
        } else {
            for (int l = 0; l < NUM_LAYERS; ++l) {
                decode_in_name_strs.push_back("past_key_" + std::to_string(l));
                decode_in_name_strs.push_back("past_value_" + std::to_string(l));
            }
        }
        std::vector<const char*> decode_in_names;
        for (auto& s : decode_in_name_strs) decode_in_names.push_back(s.c_str());

        std::vector<std::string> decode_out_name_strs;
        decode_out_name_strs.push_back("logits");
        if (d.combined_kv) {
            for (int l = 0; l < NUM_LAYERS; ++l) {
                decode_out_name_strs.push_back("present_" + std::to_string(l));
            }
        } else {
            for (int l = 0; l < NUM_LAYERS; ++l) {
                decode_out_name_strs.push_back("present_key_" + std::to_string(l));
                decode_out_name_strs.push_back("present_value_" + std::to_string(l));
            }
        }
        std::vector<const char*> decode_out_names;
        for (auto& s : decode_out_name_strs) decode_out_names.push_back(s.c_str());

        // Initialize KV-cache from prefill output (skip logits[0] and hidden_states[1])
        // Prefill always outputs separate present_key_N [B,H,S,Dh] + present_value_N [B,H,S,Dh]
        std::vector<Ort::Value> kv_cache;
        // Backing storage for combined KV tensors (must outlive kv_cache until first Run())
        std::vector<std::vector<float>> kv_combined_bufs;
        if (d.combined_kv) {
            // Combine each key/value pair into [2, B, H, S, Dh] for fused decode model
            size_t kv_per_layer = NUM_HEADS * prefill_seq * HEAD_DIM;
            kv_combined_bufs.resize(NUM_LAYERS);
            for (int l = 0; l < NUM_LAYERS; ++l) {
                size_t key_idx = 2 + l * 2;
                size_t val_idx = 2 + l * 2 + 1;
                float* key_data = prefill_out[key_idx].GetTensorMutableData<float>();
                float* val_data = prefill_out[val_idx].GetTensorMutableData<float>();

                // Allocate combined buffer [2, 1, H, S, Dh]
                kv_combined_bufs[l].resize(2 * kv_per_layer);
                std::memcpy(kv_combined_bufs[l].data(), key_data, kv_per_layer * sizeof(float));
                std::memcpy(kv_combined_bufs[l].data() + kv_per_layer, val_data, kv_per_layer * sizeof(float));

                std::vector<int64_t> shape = {2, 1, NUM_HEADS, static_cast<int64_t>(prefill_seq), HEAD_DIM};
                kv_cache.push_back(Ort::Value::CreateTensor<float>(
                    d.mem_info, kv_combined_bufs[l].data(), 2 * kv_per_layer,
                    shape.data(), shape.size()));
            }
        } else {
            for (size_t i = 2; i < prefill_out.size(); ++i) {
                kv_cache.push_back(std::move(prefill_out[i]));
            }
        }

        // mel_pos tracks the mel-relative position for mel_pos_embedding.
        // Python sequence: prefill start_mel uses mel_pos=0, then decode steps use 2, 3, 4, ...
        // (position 1 is skipped because HF generate appends 1 to attention_mask for the
        // prefill's start_mel token, making first decode index = (mel_len+1+1) - mel_len = 2)
        int64_t mel_pos = 2;  // first decode step starts at mel position 2

        // NOTE: IoBinding was tested but caused severe GPU memory explosion (~15.6G)
        // because the ORT arena accumulates fragmented KV-cache blocks: each step's
        // KV output is 1 position larger than the previous, so freed blocks can never
        // be reused. Session::Run() avoids this because all GPU memory is transient
        // (allocated during Run, freed after), allowing the arena to fully reuse blocks.

        // Reusable buffers for input_ids and position_ids (1 element each)
        std::vector<int64_t> input_ids_data(1);
        std::vector<int64_t> input_ids_shape = {1, 1};
        std::vector<int64_t> position_ids_data(1);
        std::vector<int64_t> position_ids_shape = {1, 1};

        for (int step = 0; step < max_gen; ++step) {
            // Apply sampling to logits_buf
            apply_repetition_penalty(logits_buf.data(), VOCAB_MEL,
                                     generated_codes, params.repetition_penalty);
            apply_temperature(logits_buf.data(), VOCAB_MEL, params.temperature);

            if (params.enable_fast_mode) {
                TokenDistributionSnapshot current_distribution = inspect_token_distribution(
                    logits_buf.data(),
                    VOCAB_MEL,
                    static_cast<int>(generated_codes.size()));
                last_distribution = current_distribution;
                have_last_distribution = true;

                tail_trace.push_back(current_distribution);
                if (tail_trace.size() > 16) {
                    tail_trace.erase(tail_trace.begin());
                }

                if (params.early_stop_min_generated_codes > 0 &&
                    static_cast<int>(generated_codes.size()) >= params.early_stop_min_generated_codes) {
                    if (!have_best_eligible_stop_distribution ||
                        current_distribution.stop_prob > best_eligible_stop_prob ||
                        (current_distribution.stop_prob == best_eligible_stop_prob &&
                         current_distribution.stop_rank < best_eligible_stop_rank)) {
                        best_eligible_stop_prob = current_distribution.stop_prob;
                        best_eligible_stop_rank = current_distribution.stop_rank;
                        best_eligible_stop_distribution = current_distribution;
                        have_best_eligible_stop_distribution = true;
                    }

                    if (!have_best_eligible_silent_distribution ||
                        current_distribution.silent_prob > best_eligible_silent_prob ||
                        (current_distribution.silent_prob == best_eligible_silent_prob &&
                         current_distribution.silent_rank < best_eligible_silent_rank)) {
                        best_eligible_silent_prob = current_distribution.silent_prob;
                        best_eligible_silent_rank = current_distribution.silent_rank;
                        best_eligible_silent_distribution = current_distribution;
                        have_best_eligible_silent_distribution = true;
                    }
                }

                if (params.enable_preterminal_collapse_stop &&
                    is_preterminal_collapse_candidate(
                        current_distribution,
                        current_distribution.top_token,
                        params.early_stop_min_generated_codes)) {
                    stop_reason = "fast_mode_preterminal_collapse";
                    std::ostringstream collapse_oss;
                    collapse_oss << std::fixed << std::setprecision(4);
                    collapse_oss << "[IndexTTS] Fast mode collapse early-stop after"
                                 << current_distribution.after_generated_codes
                                 << "/topToken=" << current_distribution.top_token
                                 << "/top=" << current_distribution.top_prob
                                 << "/margin=" << current_distribution.top12_margin
                                 << "/entropy=" << current_distribution.entropy;
                    log_engine_message(collapse_oss.str());
                    break;
                }
            }

            int token = sample_top_k_top_p(logits_buf.data(), VOCAB_MEL,
                                            params.top_k, params.top_p, d.rng);
            if (have_last_distribution) {
                last_distribution.sampled_token = token;
                if (!tail_trace.empty() &&
                    tail_trace.back().after_generated_codes == last_distribution.after_generated_codes &&
                    tail_trace.back().sampled_token < 0) {
                    tail_trace.back().sampled_token = token;
                }
            }
            if (have_best_eligible_stop_distribution &&
                best_eligible_stop_distribution.after_generated_codes == static_cast<int>(generated_codes.size()) &&
                best_eligible_stop_distribution.sampled_token < 0) {
                best_eligible_stop_distribution.sampled_token = token;
            }
            if (have_best_eligible_silent_distribution &&
                best_eligible_silent_distribution.after_generated_codes == static_cast<int>(generated_codes.size()) &&
                best_eligible_silent_distribution.sampled_token < 0) {
                best_eligible_silent_distribution.sampled_token = token;
            }

            bool resolving_lookahead_probe =
                lookahead_probe_arm.active &&
                have_last_distribution &&
                last_distribution.after_generated_codes == lookahead_probe_arm.after_generated_codes + 1;

            bool eligible_for_stop_probe =
                params.enable_fast_mode &&
                params.early_stop_min_generated_codes > 0 &&
                static_cast<int>(generated_codes.size()) >= params.early_stop_min_generated_codes;

            if (resolving_lookahead_probe) {
                bool next_step_stop = (token == STOP_MEL_TOKEN);
                if (next_step_stop) {
                    ++lookahead_probe_next_step_stop_count;
                }

                std::ostringstream probe_oss;
                probe_oss << std::fixed << std::setprecision(4);
                probe_oss << "[IndexTTS] Lookahead probe result armedAfter"
                          << lookahead_probe_arm.after_generated_codes
                          << "/armedToken=" << lookahead_probe_arm.sampled_token
                          << "/armedTop=" << lookahead_probe_arm.top_prob
                          << "/armedMargin=" << lookahead_probe_arm.top12_margin
                          << "/armedEntropy=" << lookahead_probe_arm.entropy
                          << "/nextAfter" << last_distribution.after_generated_codes
                          << "/nextSampled=" << token
                          << "/nextStop=" << last_distribution.stop_prob << "#" << last_distribution.stop_rank
                          << "/result=" << (next_step_stop ? "next_step_stop" : "next_step_non_stop");
                log_engine_message(probe_oss.str());
                lookahead_probe_arm.active = false;
            }

            if (eligible_for_stop_probe &&
                have_last_distribution &&
                is_preterminal_collapse_candidate(
                    last_distribution,
                    token,
                    params.early_stop_min_generated_codes)) {
                lookahead_probe_arm.active = true;
                lookahead_probe_arm.after_generated_codes = last_distribution.after_generated_codes;
                lookahead_probe_arm.sampled_token = token;
                lookahead_probe_arm.top_prob = last_distribution.top_prob;
                lookahead_probe_arm.top12_margin = last_distribution.top12_margin;
                lookahead_probe_arm.entropy = last_distribution.entropy;
                ++lookahead_probe_candidate_count;

                std::ostringstream probe_oss;
                probe_oss << std::fixed << std::setprecision(4);
                probe_oss << "[IndexTTS] Lookahead probe armed after"
                          << lookahead_probe_arm.after_generated_codes
                          << "/sampled=" << lookahead_probe_arm.sampled_token
                          << "/top=" << lookahead_probe_arm.top_prob
                          << "/margin=" << lookahead_probe_arm.top12_margin
                          << "/entropy=" << lookahead_probe_arm.entropy;
                log_engine_message(probe_oss.str());
            }

            if (have_last_distribution && token != STOP_MEL_TOKEN) {
                last_non_stop_distribution = last_distribution;
                have_last_non_stop_distribution = true;

                if (eligible_for_stop_probe &&
                    (!have_best_preterminal_stop_distribution ||
                     last_distribution.stop_prob > best_preterminal_stop_prob ||
                     (last_distribution.stop_prob == best_preterminal_stop_prob &&
                      last_distribution.stop_rank < best_preterminal_stop_rank))) {
                    best_preterminal_stop_prob = last_distribution.stop_prob;
                    best_preterminal_stop_rank = last_distribution.stop_rank;
                    best_preterminal_stop_distribution = last_distribution;
                    have_best_preterminal_stop_distribution = true;
                }
            }

            // Check for stop token
            if (token == STOP_MEL_TOKEN) {
                stop_reason = "stop_mel_token";
                break;
            }

            generated_codes.push_back(token);
            int generated_index = static_cast<int>(generated_codes.size()) - 1;
            if (token == SILENT_TOKEN) {
                if (current_silence_run_start < 0) {
                    current_silence_run_start = generated_index;
                }
                ++trailing_silence;
                max_silence_run = std::max(max_silence_run, trailing_silence);
            } else {
                if (current_silence_run_start >= 0) {
                    silence_runs.push_back({current_silence_run_start, trailing_silence});
                    current_silence_run_start = -1;
                }
                trailing_silence = 0;
            }

            if (params.enable_fast_mode &&
                params.early_stop_min_generated_codes > 0 &&
                static_cast<int>(generated_codes.size()) >= params.early_stop_min_generated_codes) {
                max_eligible_trailing_silence = std::max(
                    max_eligible_trailing_silence,
                    trailing_silence);
            }

            if (params.enable_fast_mode &&
                params.early_stop_trailing_silence > 0 &&
                params.early_stop_min_generated_codes > 0 &&
                static_cast<int>(generated_codes.size()) >= params.early_stop_min_generated_codes &&
                trailing_silence >= params.early_stop_trailing_silence) {
                stop_reason = "fast_mode_trailing_silence";
                std::ostringstream oss;
                oss << "[IndexTTS] Fast mode early-stop after trailing silence run="
                    << trailing_silence << " and " << generated_codes.size()
                    << " generated codes";
                log_engine_message(oss.str());
                break;
            }

            if (callback) {
                callback(si + 1, static_cast<int>(sentences.size()), step + 1);
            }

            // Prepare decode inputs
            input_ids_data[0] = static_cast<int64_t>(token);
            position_ids_data[0] = mel_pos;
            auto input_ids_tensor = create_tensor_int64(d.mem_info, input_ids_data, input_ids_shape);
            auto position_ids_tensor = create_tensor_int64(d.mem_info, position_ids_data, position_ids_shape);

            // Build input tensor array
            std::vector<Ort::Value> decode_inputs;
            decode_inputs.push_back(std::move(input_ids_tensor));
            decode_inputs.push_back(std::move(position_ids_tensor));
            for (auto& kv : kv_cache) {
                decode_inputs.push_back(std::move(kv));
            }
            kv_cache.clear();

            // Run decode
            auto decode_out = d.sess_gpt_decode->Run(
                Ort::RunOptions{nullptr},
                decode_in_names.data(),
                decode_inputs.data(),
                decode_inputs.size(),
                decode_out_names.data(),
                decode_out_names.size());

            // Extract logits: [1, 1, VOCAB_MEL]
            float* decode_logits = decode_out[0].GetTensorMutableData<float>();
            std::memcpy(logits_buf.data(), decode_logits, VOCAB_MEL * sizeof(float));

            // Update KV-cache
            for (size_t i = 1; i < decode_out.size(); ++i) {
                kv_cache.push_back(std::move(decode_out[i]));
            }

            ++mel_pos;
        }

        if (current_silence_run_start >= 0) {
            silence_runs.push_back({current_silence_run_start, trailing_silence});
        }

        int total_silence_tokens = 0;
        for (const auto& run : silence_runs) {
            total_silence_tokens += run.length;
        }

        if (params.enable_fast_mode) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(4);
            oss << "[IndexTTS] Fast mode diagnostics: stopReason=" << stop_reason
                << ", generatedCodes=" << generated_codes.size()
                << ", silenceTokens=" << total_silence_tokens
                << ", silenceRuns=" << format_silence_runs(silence_runs)
                << ", maxSilenceRun=" << max_silence_run
                << ", maxEligibleTrailingSilence=" << max_eligible_trailing_silence
                << ", finalTrailingSilence=" << trailing_silence
                << ", preterminalCollapseStop=" << (params.enable_preterminal_collapse_stop ? 1 : 0)
                << ", earlyStopMinCodes=" << params.early_stop_min_generated_codes
                << ", earlyStopTrailingSilence=" << params.early_stop_trailing_silence;
            if (have_last_distribution) {
                oss << ", terminalProbe=after" << last_distribution.after_generated_codes
                    << "/sampled=" << last_distribution.sampled_token
                    << "/top=" << last_distribution.top_token << ":" << last_distribution.top_prob
                    << "/stop=" << last_distribution.stop_prob << "#" << last_distribution.stop_rank
                    << "/silent=" << last_distribution.silent_prob << "#" << last_distribution.silent_rank
                    << "/top3=" << format_token_candidates(last_distribution.top_candidates);
            }
            if (have_best_eligible_stop_distribution) {
                oss << ", bestEligibleStop=after" << best_eligible_stop_distribution.after_generated_codes
                    << "/sampled=" << best_eligible_stop_distribution.sampled_token
                    << "/p=" << best_eligible_stop_distribution.stop_prob
                    << "#" << best_eligible_stop_distribution.stop_rank
                    << "/top3=" << format_token_candidates(best_eligible_stop_distribution.top_candidates);
            }
            if (have_last_non_stop_distribution) {
                oss << ", lastNonStopProbe=after" << last_non_stop_distribution.after_generated_codes
                    << "/sampled=" << last_non_stop_distribution.sampled_token
                    << "/stop=" << last_non_stop_distribution.stop_prob
                    << "#" << last_non_stop_distribution.stop_rank
                    << "/top3=" << format_token_candidates(last_non_stop_distribution.top_candidates);
            }
            if (have_best_preterminal_stop_distribution) {
                oss << ", bestEligiblePreStop=after" << best_preterminal_stop_distribution.after_generated_codes
                    << "/sampled=" << best_preterminal_stop_distribution.sampled_token
                    << "/p=" << best_preterminal_stop_distribution.stop_prob
                    << "#" << best_preterminal_stop_distribution.stop_rank
                    << "/top3=" << format_token_candidates(best_preterminal_stop_distribution.top_candidates);
            }
            if (have_best_eligible_silent_distribution) {
                oss << ", bestEligibleSilent=after" << best_eligible_silent_distribution.after_generated_codes
                    << "/sampled=" << best_eligible_silent_distribution.sampled_token
                    << "/p=" << best_eligible_silent_distribution.silent_prob
                    << "#" << best_eligible_silent_distribution.silent_rank
                    << "/top3=" << format_token_candidates(best_eligible_silent_distribution.top_candidates);
            }
            log_engine_message(oss.str());

            {
                std::ostringstream probe_oss;
                probe_oss << "[IndexTTS] Lookahead probe summary: candidates="
                          << lookahead_probe_candidate_count
                          << ", nextStepStopHits=" << lookahead_probe_next_step_stop_count;
                if (lookahead_probe_arm.active) {
                    probe_oss << ", pendingAfter=" << lookahead_probe_arm.after_generated_codes
                              << ", pendingToken=" << lookahead_probe_arm.sampled_token;
                }
                log_engine_message(probe_oss.str());
            }

            for (const auto& snapshot : tail_trace) {
                std::ostringstream tail_oss;
                tail_oss << "[IndexTTS] Tail trace " << format_distribution_snapshot(snapshot);
                log_engine_message(tail_oss.str());
            }
        }

        {
            std::ostringstream oss;
            oss << "[IndexTTS] Generated " << generated_codes.size() << " mel codes";
            log_engine_message(oss.str());
        }
        record_stage(
            stage_label("gpt_decode_loop", si + 1, static_cast<int>(sentences.size())),
            t_stage,
            static_cast<int>(generated_codes.size()));
        t_stage = std::chrono::steady_clock::now();

        if (generated_codes.empty()) continue;

        // ── Silence Removal ──────────────────────────────────────
        auto codes = remove_long_silence(
            generated_codes,
            params.pause_trim_trigger_total_silent,
            params.pause_trim_keep_silence);
        {
            std::ostringstream oss;
            oss << "[IndexTTS] After silence removal: " << codes.size()
                << " codes (from " << generated_codes.size() << ")";
            log_engine_message(oss.str());
        }

        // ── Latent Extraction (via gpt_prefill hidden_states) ──
        // Reuses gpt_prefill to extract latent, eliminating gpt_latent.onnx
        // (~950MB GPU memory savings in FP16).
        //
        // Python forward() token flow:
        //   1. codes input: [c1,...,cN] (N elements, silence-removed, no stop)
        //   2. F.pad(codes, (0,1), stop) → [c1,...,cN, stop] (N+1)
        //   3. build_aligned prepend start → [start, c1,...,cN, stop] (N+2)
        //   4. mel_emb positions: 0, 1, ..., N+1
        //   5. return hidden_states[mel_offset:][:, :-2] → N latent frames
        //
        // text_inputs: F.pad(stop) then build_aligned(start, stop)
        //   → [start, t1,...,tK, stop] (K+2 tokens)
        std::vector<int64_t> text_tokens_i64;
        text_tokens_i64.push_back(START_TEXT_TOKEN);
        for (auto id : token_ids) text_tokens_i64.push_back(static_cast<int64_t>(id));
        text_tokens_i64.push_back(STOP_TEXT_TOKEN);

        std::vector<int64_t> mel_codes_i64;
        mel_codes_i64.push_back(START_MEL_TOKEN);
        mel_codes_i64.insert(mel_codes_i64.end(), codes.begin(), codes.end());
        mel_codes_i64.push_back(STOP_MEL_TOKEN);
        // Python forward(): F.pad(codes,(0,1),stop) then build_aligned prepend start
        //   → mel = [start, c1..cN, stop] (N+2), latent = hidden[:, :-2] = N frames

        int lat_text_len = static_cast<int>(text_tokens_i64.size());
        int lat_mel_len = static_cast<int>(mel_codes_i64.size());
        int lat_total_seq = COND_NUM_LATENTS + lat_text_len + lat_mel_len;

        // Build inputs_embeds: [1, total_seq, 1280]
        std::vector<float> lat_embeds(lat_total_seq * MODEL_DIM, 0.0f);

        // Position 0..31: cond
        std::memcpy(lat_embeds.data(), cond.data(), cond_size * sizeof(float));

        // Position 32..32+text_len-1: text embeddings
        for (int i = 0; i < lat_text_len; ++i) {
            int token = static_cast<int>(text_tokens_i64[i]);
            int out_pos = COND_NUM_LATENTS + i;
            const float* te = d.text_emb.row_ptr(token);
            const float* tp = d.text_pos_emb.row_ptr(i);
            float* dst = lat_embeds.data() + out_pos * MODEL_DIM;
            for (int d_i = 0; d_i < MODEL_DIM; ++d_i) {
                dst[d_i] = te[d_i] + tp[d_i];
            }
        }

        // Position 32+text_len..end: mel embeddings
        for (int i = 0; i < lat_mel_len; ++i) {
            int code = static_cast<int>(mel_codes_i64[i]);
            int out_pos = COND_NUM_LATENTS + lat_text_len + i;
            const float* me = d.mel_emb.row_ptr(code);
            const float* mp = d.mel_pos_emb.row_ptr(i);
            float* dst = lat_embeds.data() + out_pos * MODEL_DIM;
            for (int d_i = 0; d_i < MODEL_DIM; ++d_i) {
                dst[d_i] = me[d_i] + mp[d_i];
            }
        }

        std::vector<int64_t> lat_embeds_shape = {1, lat_total_seq, MODEL_DIM};

        // Run gpt_prefill requesting ONLY hidden_states (skip logits + KV-cache)
        std::vector<const char*> lat_in_names = {"inputs_embeds"};
        std::vector<const char*> lat_out_names = {"hidden_states"};

        auto lat_embeds_tensor = create_tensor_float(d.mem_info, lat_embeds, lat_embeds_shape);

        auto lat_out = d.sess_gpt_prefill->Run(
            Ort::RunOptions{nullptr},
            lat_in_names.data(), &lat_embeds_tensor, 1,
            lat_out_names.data(), 1);

        // hidden_states: [1, total_seq, 1280] (post-final_norm)
        // Extract mel portion [offset .. offset+mel_len-2] (Python's :-2)
        float* hidden_data = lat_out[0].GetTensorMutableData<float>();
        int mel_offset = COND_NUM_LATENTS + lat_text_len;
        int latent_T = lat_mel_len - 2;
        float* latent_start = hidden_data + mel_offset * MODEL_DIM;
        std::vector<float> latent(latent_start, latent_start + latent_T * MODEL_DIM);

        all_latents.push_back({std::move(latent), latent_T});

        std::cerr << "[IndexTTS] Sentence " << (si + 1) << " latent: "
                  << latent_T << " frames" << std::endl;
        record_stage(stage_label("latent_extraction", si + 1, static_cast<int>(sentences.size())), t_stage);
        t_stage = std::chrono::steady_clock::now();
    }

    // ================================================================
    // Step 6: BigVGAN Vocoder — chunk latents (chunk_size=2, like Python)
    // Python concatenates adjacent sentences' latents so BigVGAN generates
    // smooth audio across sentence boundaries.
    // ================================================================
    std::vector<float> all_wav;
    const int CHUNK_SIZE = 2;

    std::vector<const char*> bv_in_names = {"latent", "mel_ref"};
    std::vector<const char*> bv_out_names = {"wav"};

    for (int ci = 0; ci < static_cast<int>(all_latents.size()); ci += CHUNK_SIZE) {
        int chunk_end = std::min(ci + CHUNK_SIZE, static_cast<int>(all_latents.size()));

        // Concatenate latents in this chunk
        int total_T = 0;
        for (int j = ci; j < chunk_end; ++j) total_T += all_latents[j].T;

        std::vector<float> chunk_latent;
        chunk_latent.reserve(total_T * MODEL_DIM);
        for (int j = ci; j < chunk_end; ++j) {
            chunk_latent.insert(chunk_latent.end(),
                                all_latents[j].data.begin(),
                                all_latents[j].data.end());
        }

        std::vector<int64_t> latent_bv_shape = {1, total_T, MODEL_DIM};
        std::vector<int64_t> mel_ref_shape = {1, ref_mel.n_frames, N_MELS};
        std::vector<float> mel_ref_copy(mel_ref_t);

        std::vector<Ort::Value> bv_inputs;
        bv_inputs.push_back(create_tensor_float(d.mem_info, chunk_latent, latent_bv_shape));
        bv_inputs.push_back(create_tensor_float(d.mem_info, mel_ref_copy, mel_ref_shape));

        auto bv_out = d.sess_bigvgan->Run(
            Ort::RunOptions{nullptr},
            bv_in_names.data(),
            bv_inputs.data(), bv_inputs.size(),
            bv_out_names.data(), 1);

        // wav: [1, 1, wav_samples]
        auto wav_info = bv_out[0].GetTensorTypeAndShapeInfo();
        auto wav_shape = wav_info.GetShape();
        int wav_samples = static_cast<int>(wav_shape.back());
        float* wav_data = bv_out[0].GetTensorMutableData<float>();

        // Match Python: no fade processing, just concatenate raw BigVGAN output.
        // Python does: wav = torch.clamp(32767 * wav, -32767, 32767); wavs.append(wav)
        all_wav.insert(all_wav.end(), wav_data, wav_data + wav_samples);

        std::cerr << "[IndexTTS] BigVGAN chunk " << (ci / CHUNK_SIZE + 1)
                  << " (sentences " << (ci + 1) << "-" << chunk_end
                  << "): " << wav_samples << " samples" << std::endl;
    }

    record_stage("bigvgan_total", t_stage);
    timing.audio_samples = static_cast<int>(all_wav.size());
    timing.pipeline_ms = ms_since(t_total_start);
    std::cerr << "[TIMING] total: " << timing.pipeline_ms << "ms" << std::endl;

    // Match Python: no post-processing (no trim, no fade-out).
    // Python does: wav = torch.cat(wavs, dim=1); torchaudio.save(output_path, wav.int16, sr)

    return all_wav;
}

bool IndexTTSEngine::synthesize(const std::string& ref_audio_path,
                                  const std::string& text,
                                  const std::string& output_path,
                                  const GenerateParams& params,
                                  ProgressCallback callback) {
    try {
        auto t_total_start = std::chrono::steady_clock::now();
        auto pcm = synthesize_pcm(ref_audio_path, text, params, callback);
        auto t_write_start = std::chrono::steady_clock::now();
        const bool ok = write_wav(output_path, pcm, SAMPLE_RATE);
        impl_->last_timing.write_wav_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_write_start).count();
        impl_->last_timing.stages.push_back(TimingEntry{"write_wav", impl_->last_timing.write_wav_ms, 0});
        impl_->last_timing.total_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_total_start).count();
        return ok;
    } catch (const std::exception& e) {
        std::cerr << "[IndexTTS] Synthesis error: " << e.what() << std::endl;
        return false;
    }
}

} // namespace indextts
