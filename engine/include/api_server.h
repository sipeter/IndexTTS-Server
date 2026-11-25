#pragma once
// HTTP API server for IndexTTS
// Routes: GET /v1/health, GET /v1/models, GET /v1/audio/voices, POST /v1/audio/speech,
//         GET /v1/tts/refs, POST /v1/tts/upload_ref, POST /v1/tts/synthesize,
//         GET /api/legado[/all], GET|POST /legado/tts
// Optional bearer-token auth via ServerConfig::auth_token (see --auth-token).

#include <string>
#include <memory>

namespace indextts {

class IndexTTSEngine;  // forward decl

struct ServerConfig {
    std::string host     = "0.0.0.0";
    int         port     = 9966;
    std::string model_dir;               // required
    std::string bpe_model;               // required
    std::string default_ref_audio;       // optional default reference
    bool        use_cuda = false;
    bool        use_tensorrt = false;
    std::string trt_cache_dir;
    int         num_threads = 0;
    std::string auth_token;              // optional; when non-empty, all requests must present it
};

class ApiServer {
public:
    ApiServer();
    ~ApiServer();

    /// Initialize with config. Loads the engine.
    bool init(const ServerConfig& config);

    /// Run the HTTP server (blocking).
    void run();

    /// Stop the server.
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace indextts
