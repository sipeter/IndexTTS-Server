// main.cpp — CLI entry point for IndexTTS C++ inference engine
// Modes: (1) server mode with --server, (2) single-shot synthesis

#include "api_server.h"
#include "indextts_engine.h"
#include "constants.h"

#include <iostream>
#include <string>
#include <cstring>
#include <csignal>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace indextts;

static ApiServer* g_server = nullptr;

static void signal_handler(int) {
    if (g_server) g_server->stop();
}

static void print_usage(const char* prog) {
    std::cerr
        << "IndexTTS C++ Inference Engine\n\n"
        << "Usage:\n"
        << "  " << prog << " --server [options]          Start HTTP API server\n"
        << "  " << prog << " --text \"...\" [options]      Single-shot synthesis\n"
        << "\nRequired:\n"
        << "  --model-dir <path>    Directory with ONNX models & embeddings\n"
        << "  --bpe-model <path>    Path to bpe.model\n"
        << "\nServer mode:\n"
        << "  --server              Start HTTP server\n"
        << "  --host <addr>         Bind address (default: 0.0.0.0)\n"
        << "  --port <port>         Bind port (default: 9966)\n"
        << "  --ref-audio <path>    Default reference audio for server\n"
        << "  --auth-token <token>  Require this token on every request (empty = no auth)\n"
        << "\nSingle-shot mode:\n"
        << "  --text <string>       Text to synthesize\n"
        << "  --ref-audio <path>    Reference audio (required)\n"
        << "  --output <path>       Output WAV path (default: output.wav)\n"
        << "\nOptional:\n"
        << "  --cuda                Use CUDA execution provider\n"
        << "  --tensorrt            Use TensorRT EP (with CUDA fallback, fastest)\n"
        << "  --trt-cache <dir>     TensorRT engine cache directory\n"
        << "  --threads <n>         Number of threads (0=auto)\n"
        << "  --temperature <f>     Sampling temperature (default: 1.0)\n"
        << "  --top-p <f>           Top-p sampling (default: 0.8)\n"
        << "  --top-k <n>           Top-k sampling (default: 30)\n"
        << "  --max-length <n>      Max mel tokens (default: 600)\n"
        << "  --help                Show this help\n";
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    // Log output is UTF-8; without this the console decodes it as the ANSI
    // code page (GBK on zh-CN) and Chinese voice names turn into mojibake.
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Parse arguments
    std::string model_dir, bpe_model, ref_audio, text, output = "output.wav";
    std::string host = "0.0.0.0";
    int port = 9966;
    bool server_mode = false;
    bool use_cuda = false;
    bool use_tensorrt = false;
    std::string trt_cache_dir;
    int num_threads = 0;
    std::string auth_token;
    GenerateParams params;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 < argc) return argv[++i];
            std::cerr << "Missing value for " << arg << "\n";
            std::exit(1);
            return "";
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--model-dir")    { model_dir = next(); }
        else if (arg == "--bpe-model")      { bpe_model = next(); }
        else if (arg == "--ref-audio")      { ref_audio = next(); }
        else if (arg == "--text")           { text = next(); }
        else if (arg == "--output")         { output = next(); }
        else if (arg == "--host")           { host = next(); }
        else if (arg == "--port")           { port = std::stoi(next()); }
        else if (arg == "--server")         { server_mode = true; }
        else if (arg == "--cuda")           { use_cuda = true; }
        else if (arg == "--tensorrt")       { use_tensorrt = true; use_cuda = true; }
        else if (arg == "--trt-cache")      { trt_cache_dir = next(); }
        else if (arg == "--threads")        { num_threads = std::stoi(next()); }
        else if (arg == "--auth-token")     { auth_token = next(); }
        else if (arg == "--temperature")    { params.temperature = std::stof(next()); }
        else if (arg == "--top-p")          { params.top_p = std::stof(next()); }
        else if (arg == "--top-k")          { params.top_k = std::stoi(next()); }
        else if (arg == "--max-length")     { params.max_generate_length = std::stoi(next()); }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            return 1;
        }
    }

    // Validate required args
    if (model_dir.empty()) {
        std::cerr << "Error: --model-dir is required\n";
        return 1;
    }
    if (bpe_model.empty()) {
        std::cerr << "Error: --bpe-model is required\n";
        return 1;
    }

    if (server_mode) {
        // ── Server mode ──
        ServerConfig config;
        config.model_dir = model_dir;
        config.bpe_model = bpe_model;
        config.host = host;
        config.port = port;
        config.default_ref_audio = ref_audio;
        config.use_cuda = use_cuda;
        config.use_tensorrt = use_tensorrt;
        config.trt_cache_dir = trt_cache_dir;
        config.num_threads = num_threads;
        config.auth_token = auth_token;

        ApiServer server;
        g_server = &server;

        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        if (!server.init(config)) {
            std::cerr << "Failed to initialize server\n";
            return 1;
        }

        server.run();
        g_server = nullptr;

    } else {
        // ── Single-shot mode ──
        if (text.empty()) {
            std::cerr << "Error: --text is required in single-shot mode\n";
            return 1;
        }
        if (ref_audio.empty()) {
            std::cerr << "Error: --ref-audio is required in single-shot mode\n";
            return 1;
        }

        IndexTTSEngine engine;
        if (!engine.init(model_dir, bpe_model, use_cuda, use_tensorrt, trt_cache_dir, num_threads)) {
            std::cerr << "Failed to initialize engine\n";
            return 1;
        }

        std::cerr << "Synthesizing..." << std::endl;
        auto progress = [](int sent, int total, int tokens) {
            std::cerr << "\r  Sentence " << sent << "/" << total
                      << " (" << tokens << " tokens)" << std::flush;
        };

        bool ok = engine.synthesize(ref_audio, text, output, params, progress);
        std::cerr << std::endl;

        if (!ok) {
            std::cerr << "Synthesis failed\n";
            return 1;
        }
        std::cerr << "Output: " << output << std::endl;
    }

    return 0;
}
