#pragma once
// IndexTTS C++ Inference Engine — Model & Audio Constants
// Auto-derived from gpt_config.json and export_onnx.py

#include <cstdint>

namespace indextts {

// ── GPT-2 Model ──────────────────────────────────────────────
constexpr int MODEL_DIM       = 1280;
constexpr int NUM_LAYERS      = 24;
constexpr int NUM_HEADS       = 20;
constexpr int HEAD_DIM        = 64;      // MODEL_DIM / NUM_HEADS

constexpr int VOCAB_MEL       = 8194;    // mel codebook size (incl. start/stop)
constexpr int VOCAB_TEXT      = 12000;   // BPE text vocabulary

constexpr int START_MEL_TOKEN = 8192;
constexpr int STOP_MEL_TOKEN  = 8193;
constexpr int START_TEXT_TOKEN= 0;
constexpr int STOP_TEXT_TOKEN = 1;

constexpr int MAX_MEL_TOKENS  = 800;
constexpr int MAX_TEXT_TOKENS  = 600;
constexpr int COND_NUM_LATENTS= 32;
constexpr int MEL_LENGTH_COMPRESSION = 1024;

constexpr int SILENT_TOKEN    = 52;
constexpr int MAX_CONSECUTIVE_SILENCE = 30;
constexpr int KEEP_SILENCE    = 10;      // keep at most 10 consecutive silent

// ── Embedding table shapes (row counts — columns are all MODEL_DIM) ──
constexpr int TEXT_EMB_ROWS   = 12001;   // vocab_text + 1
constexpr int MEL_EMB_ROWS    = 8194;    // vocab_mel
constexpr int TEXT_POS_ROWS   = 602;     // max_text + 2
constexpr int MEL_POS_ROWS    = 803;     // max_mel + 3

// ── Audio / Mel ──────────────────────────────────────────────
constexpr int SAMPLE_RATE     = 24000;
constexpr int N_FFT           = 1024;
constexpr int HOP_LENGTH      = 256;
constexpr int WIN_LENGTH      = 1024;    // same as N_FFT
constexpr int N_MELS          = 100;
constexpr float MEL_FMIN      = 0.0f;
constexpr float MEL_FMAX      = 12000.0f;  // sample_rate / 2
constexpr float MEL_LOG_CLIP  = 1e-7f;

// ── GPT Generation defaults ─────────────────────────────────
constexpr float TEMPERATURE       = 1.0f;
constexpr float TOP_P             = 0.8f;
constexpr int   TOP_K             = 30;
constexpr float REPETITION_PENALTY= 10.0f;
constexpr int   MAX_GENERATE_LENGTH = 600;

// ── Wav output ───────────────────────────────────────────────
constexpr float WAV_SCALE       = 32767.0f;

} // namespace indextts
