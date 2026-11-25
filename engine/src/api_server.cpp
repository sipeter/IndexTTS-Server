// api_server.cpp — HTTP API server for IndexTTS
// Uses cpp-httplib for REST endpoints

#include "api_server.h"
#include "indextts_engine.h"
#include "constants.h"
#include "md5.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <algorithm>
#include <unordered_map>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace indextts {

namespace {

static std::string url_encode(const std::string& s) {
    static const char* kHex = "0123456789ABCDEF";
    std::string out;
    out.reserve(s.size() * 3);
    for (unsigned char c : s) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~') {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('%');
            out.push_back(kHex[(c >> 4) & 0x0F]);
            out.push_back(kHex[c & 0x0F]);
        }
    }
    return out;
}

static bool is_audio_file(const fs::path& p) {
    auto ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext == ".wav" || ext == ".mp3" || ext == ".flac" || ext == ".ogg" || ext == ".m4a";
}

// Upper bound for input text per request (UTF-8 bytes). A whole book chapter
// fits well below this; anything larger would hold the synthesis lock for
// many minutes and overflow the uint32 WAV size fields.
constexpr size_t kMaxTextBytes = 16 * 1024;

}  // namespace

struct ApiServer::Impl {
    ServerConfig config;
    IndexTTSEngine engine;
    httplib::Server server;
    std::mutex synth_mutex;  // Serialize synthesis requests
    std::mutex voices_mutex; // Guards `voices` (read by handlers, mutated by upload_ref)
    std::vector<std::pair<std::string, std::string>> voices;  // (voice_name, wav_path)
    bool ref_is_directory = false;
    bool ref_is_file = false;
};

ApiServer::ApiServer() : impl_(std::make_unique<Impl>()) {}
ApiServer::~ApiServer() = default;

bool ApiServer::init(const ServerConfig& config) {
    impl_->config = config;

    // Initialize engine
    if (!impl_->engine.init(config.model_dir, config.bpe_model,
                             config.use_cuda, config.use_tensorrt,
                             config.trt_cache_dir, config.num_threads)) {
        return false;
    }

    // Build voice list from reference audio config.
    impl_->voices.clear();
    impl_->ref_is_directory = false;
    impl_->ref_is_file = false;
    try {
        fs::path ref_path(config.default_ref_audio);
        if (!config.default_ref_audio.empty() && fs::exists(ref_path)) {
            if (fs::is_directory(ref_path)) {
                impl_->ref_is_directory = true;
                for (const auto& entry : fs::directory_iterator(ref_path)) {
                    if (!entry.is_regular_file()) {
                        continue;
                    }
                    const auto& p = entry.path();
                    if (!is_audio_file(p)) {
                        continue;
                    }
                    impl_->voices.emplace_back(p.stem().u8string(), p.string());
                }
                std::sort(impl_->voices.begin(), impl_->voices.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });
            } else if (fs::is_regular_file(ref_path)) {
                impl_->ref_is_file = true;
                impl_->voices.emplace_back(ref_path.stem().u8string(), ref_path.string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[IndexTTS-Server] Failed to enumerate reference voices: " << e.what() << std::endl;
    }

    std::cerr << "[IndexTTS-Server] Reference path: " << config.default_ref_audio << std::endl;
    std::cerr << "[IndexTTS-Server] Reference mode: "
              << (impl_->ref_is_directory ? "directory" : (impl_->ref_is_file ? "single-file" : "none"))
              << ", voices=" << impl_->voices.size() << std::endl;
    for (const auto& v : impl_->voices) {
        // v.second is in the ANSI code page (what ifstream needs on Windows);
        // re-encode to UTF-8 for display only.
        std::cerr << "[IndexTTS-Server] Voice: " << v.first << " -> "
                  << fs::path(v.second).u8string() << std::endl;
    }

    auto& srv = impl_->server;

    // Reject oversized/stalled requests before they reach handlers.
    // 20 MB accommodates reference-audio uploads (/v1/tts/upload_ref); the
    // synthesize text body is separately capped at kMaxTextBytes.
    srv.set_payload_max_length(20 * 1024 * 1024);  // 20 MB
    srv.set_read_timeout(15, 0);
    srv.set_write_timeout(60, 0);

    // ── Optional bearer-token auth (pre-routing gate) ───────────
    // When auth_token is set, every request must present it via
    // `Authorization: Bearer <token>` or `?token=<token>`. Empty = open
    // (backward compatible). /v1/health is intentionally NOT exempt so the
    // client's connection test surfaces auth failures directly.
    if (!impl_->config.auth_token.empty()) {
        srv.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
            std::string provided;
            if (req.has_header("Authorization")) {
                const std::string h = req.get_header_value("Authorization");
                const std::string prefix = "Bearer ";
                if (h.size() > prefix.size() && h.compare(0, prefix.size(), prefix) == 0) {
                    provided = h.substr(prefix.size());
                }
            }
            if (provided.empty() && req.has_param("token")) {
                provided = req.get_param_value("token");
            }
            if (provided != impl_->config.auth_token) {
                res.status = 401;
                res.set_content(R"({"error": "unauthorized"})", "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
    }

    // ── GET /v1/health ──────────────────────────────────────────
    srv.Get("/v1/health", [this](const httplib::Request&, httplib::Response& res) {
        json j = {
            {"status", "ok"},
            {"engine_ready", impl_->engine.is_ready()}
        };
        res.set_content(j.dump(), "application/json");
    });

    // ── GET /v1/models — OpenAI compatible model list ───────────
    srv.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
        json j = {
            {"object", "list"},
            {"data", json::array({
                {
                    {"id", "indextts-1.5"},
                    {"object", "model"},
                    {"owned_by", "clonetts"}
                }
            })}
        };
        res.set_content(j.dump(), "application/json");
    });

    // ── GET /v1/audio/voices — OpenAI-style voice discovery ─────
    srv.Get("/v1/audio/voices", [this](const httplib::Request&, httplib::Response& res) {
        json voices = json::array();
        {
            std::lock_guard<std::mutex> vlock(impl_->voices_mutex);
            for (const auto& v : impl_->voices) {
                voices.push_back({{"id", v.first}, {"name", v.first}});
            }
        }
        json body = {{"voices", voices}};
        res.set_content(body.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
    });

    // ── POST /v1/audio/speech — OpenAI compatible TTS endpoint ──
    srv.Post("/v1/audio/speech", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            if (!req.has_header("Content-Type") ||
                req.get_header_value("Content-Type").find("application/json") == std::string::npos) {
                res.status = 400;
                res.set_content(R"({"error":{"message":"Content-Type must be application/json","type":"invalid_request_error"}})", "application/json");
                return;
            }
            json body = json::parse(req.body);
            std::string text = body.value("input", "");
            if (text.empty()) {
                res.status = 400;
                res.set_content(R"({"error":{"message":"input is required","type":"invalid_request_error"}})", "application/json");
                return;
            }
            if (text.size() > kMaxTextBytes) {
                res.status = 413;
                res.set_content(R"({"error":{"message":"input too long","type":"invalid_request_error"}})", "application/json");
                return;
            }

            std::string voice = body.value("voice", "");
            std::string ref_audio;
            {
                std::lock_guard<std::mutex> vlock(impl_->voices_mutex);
                if (!voice.empty() && voice != "default" && voice != "alloy") {
                    auto it = std::find_if(impl_->voices.begin(), impl_->voices.end(), [&](const auto& v) {
                        return v.first == voice;
                    });
                    if (it == impl_->voices.end()) {
                        res.status = 400;
                        json err = {{"error", {{"message", "unknown voice: " + voice}, {"type", "invalid_request_error"}}}};
                        res.set_content(err.dump(), "application/json");
                        return;
                    }
                    ref_audio = it->second;
                }
                if (ref_audio.empty()) {
                    if (!impl_->voices.empty()) {
                        ref_audio = impl_->voices.front().second;
                    } else if (impl_->ref_is_file) {
                        ref_audio = impl_->config.default_ref_audio;
                    }
                }
            }
            if (ref_audio.empty()) {
                res.status = 400;
                res.set_content(R"({"error":{"message":"no reference voice configured on server","type":"invalid_request_error"}})", "application/json");
                return;
            }

            GenerateParams params;
            auto start = std::chrono::steady_clock::now();
            std::vector<float> pcm;
            {
                std::unique_lock<std::mutex> lock(impl_->synth_mutex, std::try_to_lock);
                if (!lock.owns_lock()) {
                    res.status = 429;
                    res.set_header("Retry-After", "2");
                    res.set_content(R"({"error":{"message":"server busy, try again","type":"rate_limit_error"}})", "application/json");
                    return;
                }
                pcm = impl_->engine.synthesize_pcm(ref_audio, text, params);
            }

            auto elapsed = std::chrono::duration<float>(
                std::chrono::steady_clock::now() - start).count();
            float duration = static_cast<float>(pcm.size()) / SAMPLE_RATE;

            std::string resp_fmt = body.value("response_format", "wav");
            std::transform(resp_fmt.begin(), resp_fmt.end(), resp_fmt.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });

            std::vector<int16_t> pcm16(pcm.size());
            for (size_t i = 0; i < pcm.size(); ++i) {
                float v = pcm[i] * WAV_SCALE;
                v = std::max(-WAV_SCALE, std::min(WAV_SCALE, v));
                pcm16[i] = static_cast<int16_t>(v);
            }
            uint32_t data_size = static_cast<uint32_t>(pcm16.size() * 2);

            if (resp_fmt == "pcm") {
                res.set_content(
                    std::string(reinterpret_cast<const char*>(pcm16.data()), data_size),
                    "audio/pcm");
            } else {
                std::vector<char> wav_buf(44 + data_size);
                char* p = wav_buf.data();
                uint32_t file_size = 36 + data_size;
                uint32_t fmt_size = 16;
                uint16_t audio_fmt = 1;
                uint16_t num_ch = 1;
                uint32_t sr = SAMPLE_RATE;
                uint32_t byte_rate = SAMPLE_RATE * 2;
                uint16_t block_align = 2;
                uint16_t bps = 16;

                std::memcpy(p, "RIFF", 4); p += 4;
                std::memcpy(p, &file_size, 4); p += 4;
                std::memcpy(p, "WAVE", 4); p += 4;
                std::memcpy(p, "fmt ", 4); p += 4;
                std::memcpy(p, &fmt_size, 4); p += 4;
                std::memcpy(p, &audio_fmt, 2); p += 2;
                std::memcpy(p, &num_ch, 2); p += 2;
                std::memcpy(p, &sr, 4); p += 4;
                std::memcpy(p, &byte_rate, 4); p += 4;
                std::memcpy(p, &block_align, 2); p += 2;
                std::memcpy(p, &bps, 2); p += 2;
                std::memcpy(p, "data", 4); p += 4;
                std::memcpy(p, &data_size, 4); p += 4;
                std::memcpy(p, pcm16.data(), data_size);

                res.set_content(std::string(wav_buf.begin(), wav_buf.end()), "audio/wav");
            }
            res.set_header("X-Duration", std::to_string(duration));
            res.set_header("X-Elapsed", std::to_string(elapsed));
            res.set_header("X-RTF", std::to_string(elapsed / std::max(duration, 0.001f)));
        } catch (const std::exception& e) {
            res.status = 500;
            json err = {{"error", {{"message", e.what()}, {"type", "server_error"}}}};
            res.set_content(err.dump(), "application/json");
        }
    });

    // ── GET /v1/tts/refs — CloneTTS voice list ──────────────────
    srv.Get("/v1/tts/refs", [this](const httplib::Request&, httplib::Response& res) {
        json refs = json::array();
        {
            std::lock_guard<std::mutex> vlock(impl_->voices_mutex);
            for (const auto& v : impl_->voices) {
                fs::path p(v.second);
                long long size_bytes = 0;
                try {
                    if (fs::is_regular_file(p)) {
                        size_bytes = static_cast<long long>(fs::file_size(p));
                    }
                } catch (...) {}
                json item = {
                    {"id", v.first},
                    {"filename", p.filename().u8string()},
                    {"size_bytes", size_bytes}
                };
                refs.push_back(item);
            }
        }
        json body = {{"refs", refs}};
        res.set_content(body.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
    });

    // ── POST /v1/tts/upload_ref — register a reference voice ────
    // multipart/form-data with a single file field `audio`. The stored name is
    // {md5}{ext} so a re-upload of identical bytes is idempotent and a hostile
    // original filename can never escape the refs directory (P1 whitelist only
    // guards *reads*; this is the *write* path and validates on its own).
    srv.Post("/v1/tts/upload_ref", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            if (!req.is_multipart_form_data() || !req.has_file("audio")) {
                res.status = 400;
                res.set_content(R"({"error": "multipart form field 'audio' required"})", "application/json");
                return;
            }
            const auto& file = req.get_file_value("audio");
            if (file.content.empty()) {
                res.status = 400;
                res.set_content(R"({"error": "Empty file"})", "application/json");
                return;
            }
            // Only accept known audio extensions; the stored name uses this ext.
            fs::path orig(file.filename);
            if (!is_audio_file(orig)) {
                res.status = 400;
                res.set_content(R"j({"error": "unsupported audio format (wav/mp3/flac/ogg/m4a)"})j", "application/json");
                return;
            }
            std::string ext = orig.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });

            // Uploads need a directory to write into (folder ref mode).
            if (!impl_->ref_is_directory || impl_->config.default_ref_audio.empty()) {
                res.status = 400;
                res.set_content(R"({"error": "server not in directory ref mode; cannot accept uploads"})", "application/json");
                return;
            }

            std::string md5 = md5_hex(file.content);
            std::string filename = md5 + ext;
            fs::path dest = fs::path(impl_->config.default_ref_audio) / fs::u8path(filename);

            // MD5 naming makes writes idempotent: identical bytes → identical file.
            if (!fs::exists(dest)) {
                std::ofstream ofs(dest, std::ios::binary);
                if (!ofs) {
                    res.status = 500;
                    res.set_content(R"({"error": "failed to write reference file"})", "application/json");
                    return;
                }
                ofs.write(file.content.data(), static_cast<std::streamsize>(file.content.size()));
                ofs.close();
            }

            // Register into the live voice list so /v1/tts/refs and synthesize
            // see it immediately (`voices` is otherwise built once at init()).
            {
                std::lock_guard<std::mutex> vlock(impl_->voices_mutex);
                auto it = std::find_if(impl_->voices.begin(), impl_->voices.end(),
                    [&](const auto& v) { return v.first == md5; });
                if (it == impl_->voices.end()) {
                    impl_->voices.emplace_back(md5, dest.string());
                    std::sort(impl_->voices.begin(), impl_->voices.end(),
                        [](const auto& a, const auto& b) { return a.first < b.first; });
                }
            }

            json body = {{"ref_id", md5}, {"filename", filename}};
            res.set_content(body.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            json err = {{"error", e.what()}};
            res.set_content(err.dump(), "application/json");
        }
    });

    // ── POST /v1/tts/synthesize ─────────────────────────────────
    srv.Post("/v1/tts/synthesize", [this](const httplib::Request& req, httplib::Response& res) {
        auto t_request_start = std::chrono::steady_clock::now();
        auto ms_since_req = [&t_request_start]() {
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_request_start).count();
        };
        try {
            json body;
            if (req.has_header("Content-Type") &&
                req.get_header_value("Content-Type").find("application/json") != std::string::npos) {
                body = json::parse(req.body);
            } else {
                res.status = 400;
                res.set_content(R"({"error": "Content-Type must be application/json"})", "application/json");
                return;
            }

            std::string text = body.value("text", "");
            if (text.empty()) {
                res.status = 400;
                res.set_content(R"({"error": "text is required"})", "application/json");
                return;
            }
            if (text.size() > kMaxTextBytes) {
                res.status = 413;
                res.set_content(R"({"error": "text too long"})", "application/json");
                return;
            }

            std::string ref_audio = body.value("ref_audio", "");
            // Resolve voice name to full path. Only whitelisted voice names are
            // accepted — a raw filesystem path from the client must never reach
            // the engine (arbitrary file read). voices_mutex guards against a
            // concurrent upload_ref mutating the vector.
            {
                std::lock_guard<std::mutex> vlock(impl_->voices_mutex);
                if (!ref_audio.empty()) {
                    auto it = std::find_if(impl_->voices.begin(), impl_->voices.end(), [&](const auto& v) {
                        return v.first == ref_audio;
                    });
                    if (it == impl_->voices.end()) {
                        res.status = 400;
                        res.set_content(R"({"error": "unknown ref_audio voice name"})", "application/json");
                        return;
                    }
                    ref_audio = it->second;
                }
                // Fallback to default
                if (ref_audio.empty()) {
                    if (!impl_->voices.empty()) {
                        ref_audio = impl_->voices.front().second;
                    } else if (impl_->ref_is_file) {
                        ref_audio = impl_->config.default_ref_audio;
                    }
                }
            }
            if (ref_audio.empty()) {
                res.status = 400;
                res.set_content(R"({"error": "ref_audio is required"})", "application/json");
                return;
            }

            GenerateParams params;
            if (body.contains("temperature"))
                params.temperature = body["temperature"].get<float>();
            if (body.contains("top_p"))
                params.top_p = body["top_p"].get<float>();
            if (body.contains("top_k"))
                params.top_k = body["top_k"].get<int>();
            if (body.contains("max_length"))
                params.max_generate_length = body["max_length"].get<int>();

            auto start = std::chrono::steady_clock::now();
            std::cerr << "[API-TIMING] pre-synth: " << ms_since_req() << "ms" << std::endl;

            std::vector<float> pcm;
            {
                std::unique_lock<std::mutex> lock(impl_->synth_mutex, std::try_to_lock);
                if (!lock.owns_lock()) {
                    res.status = 429;
                    res.set_header("Retry-After", "2");
                    res.set_content(R"({"error": "server busy, try again"})", "application/json");
                    return;
                }
                pcm = impl_->engine.synthesize_pcm(ref_audio, text, params);
            }

            std::cerr << "[API-TIMING] post-synth: " << ms_since_req() << "ms" << std::endl;

            auto elapsed = std::chrono::duration<float>(
                std::chrono::steady_clock::now() - start).count();

            // Encode as 16-bit PCM WAV in memory
            float duration = static_cast<float>(pcm.size()) / SAMPLE_RATE;

            // Build WAV in memory
            std::vector<char> wav_buf;
            {
                // WAV header (44 bytes) + PCM data
                int16_t* pcm16 = new int16_t[pcm.size()];
                for (size_t i = 0; i < pcm.size(); ++i) {
                    float v = pcm[i] * WAV_SCALE;
                    v = std::max(-WAV_SCALE, std::min(WAV_SCALE, v));
                    pcm16[i] = static_cast<int16_t>(v);
                }
                uint32_t data_size = static_cast<uint32_t>(pcm.size() * 2);
                uint32_t file_size = 36 + data_size;
                wav_buf.resize(44 + data_size);
                char* p = wav_buf.data();

                // RIFF header
                std::memcpy(p, "RIFF", 4); p += 4;
                std::memcpy(p, &file_size, 4); p += 4;
                std::memcpy(p, "WAVE", 4); p += 4;

                // fmt chunk
                std::memcpy(p, "fmt ", 4); p += 4;
                uint32_t fmt_size = 16;
                std::memcpy(p, &fmt_size, 4); p += 4;
                uint16_t audio_fmt = 1; // PCM
                std::memcpy(p, &audio_fmt, 2); p += 2;
                uint16_t num_ch = 1;
                std::memcpy(p, &num_ch, 2); p += 2;
                uint32_t sr = SAMPLE_RATE;
                std::memcpy(p, &sr, 4); p += 4;
                uint32_t byte_rate = SAMPLE_RATE * 2;
                std::memcpy(p, &byte_rate, 4); p += 4;
                uint16_t block_align = 2;
                std::memcpy(p, &block_align, 2); p += 2;
                uint16_t bps = 16;
                std::memcpy(p, &bps, 2); p += 2;

                // data chunk
                std::memcpy(p, "data", 4); p += 4;
                std::memcpy(p, &data_size, 4); p += 4;
                std::memcpy(p, pcm16, data_size);

                delete[] pcm16;
            }

            std::string format = body.value("format", "wav");
            if (format == "json") {
                json resp = {
                    {"duration", duration},
                    {"elapsed", elapsed},
                    {"rtf", elapsed / std::max(duration, 0.001f)},
                    {"sample_rate", SAMPLE_RATE},
                    {"samples", pcm.size()}
                };
                res.set_content(resp.dump(), "application/json");
            } else {
                res.set_content(std::string(wav_buf.begin(), wav_buf.end()), "audio/wav");
                res.set_header("X-Duration", std::to_string(duration));
                res.set_header("X-Elapsed", std::to_string(elapsed));
                res.set_header("X-RTF", std::to_string(elapsed / std::max(duration, 0.001f)));
            }
            std::cerr << "[API-TIMING] response-ready: " << ms_since_req() << "ms" << std::endl;

        } catch (const std::exception& e) {
            res.status = 500;
            json err = {{"error", e.what()}};
            res.set_content(err.dump(), "application/json");
        }
    });

    // ── GET /api/legado/all — MultiTTS import compatible ────────
    srv.Get("/api/legado/all", [this](const httplib::Request& req, httplib::Response& res) {

        // Prefer request Host header so imported URL keeps the real reachable IP.
        std::string base_host;
        if (req.has_header("Host")) {
            base_host = req.get_header_value("Host");
        } else {
            const std::string fallback_host =
                (impl_->config.host == "0.0.0.0") ? "127.0.0.1" : impl_->config.host;
            base_host = fallback_host + ":" + std::to_string(impl_->config.port);
        }

        json arr = json::array();
        long long now_ms = static_cast<long long>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());

        // voices_mutex guards against a concurrent upload_ref mutation
        // (an empty voice list simply yields an empty array).
        {
            std::lock_guard<std::mutex> vlock(impl_->voices_mutex);
            int id = 1;
            for (const auto& v : impl_->voices) {
                json item = {
                    {"id", id++},
                    {"name", v.first},
                    {"url", "http://" + base_host + "/legado/tts?voice=" + url_encode(v.first) + "&text={{java.encodeURI(speakText)}}"},
                    {"contentType", "audio/wav"},
                    {"concurrentRate", "0"},
                    {"header", ""},
                    {"lastUpdateTime", now_ms}
                };
                arr.push_back(item);
            }
        }

        // Be tolerant to any unexpected non-UTF8 bytes in voice names on Windows.
        // This avoids blank 500 responses in reader import when a filename contains odd bytes.
        res.set_content(arr.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
    });

    // ── GET /api/legado?voice=xxx — single voice config ─────────
    srv.Get("/api/legado", [this](const httplib::Request& req, httplib::Response& res) {
        std::string voice = req.has_param("voice") ? req.get_param_value("voice") : "";
        if (voice.empty()) {
            std::lock_guard<std::mutex> vlock(impl_->voices_mutex);
            if (!impl_->voices.empty()) {
                voice = impl_->voices.front().first;
            } else {
                res.status = 400;
                res.set_content(R"({"error":"No available voices. Configure Ref Folder/File with wav files."})", "application/json");
                return;
            }
        }

        std::string base_host;
        if (req.has_header("Host")) {
            base_host = req.get_header_value("Host");
        } else {
            const std::string fallback_host =
                (impl_->config.host == "0.0.0.0") ? "127.0.0.1" : impl_->config.host;
            base_host = fallback_host + ":" + std::to_string(impl_->config.port);
        }

        const std::string url =
            "http://" + base_host + "/legado/tts?voice=" + url_encode(voice) + "&text={{java.encodeURI(speakText)}}";

        json item = {
            {"id", 1},
            {"name", voice},
            {"url", url},
            {"contentType", "audio/wav"},
            {"concurrentRate", "0"},
            {"header", ""},
            {"lastUpdateTime", static_cast<long long>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count())}
        };
        res.set_content(item.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
    });

    // ── GET/POST /legado/tts — Legado TTS API compatible ────────
    auto legado_tts_handler = [this](const httplib::Request& req, httplib::Response& res) {
        try {
            // multipart/form-data or URL params
            std::string text = req.has_param("text") ? req.get_param_value("text") : "";
            std::string ref_audio;
            // Both params are voice names resolved against the whitelist; raw
            // paths from the client are rejected (arbitrary file read).
            std::string voice;
            if (req.has_param("ref_audio")) {
                voice = req.get_param_value("ref_audio");
            } else if (req.has_param("voice")) {
                voice = req.get_param_value("voice");
            }
            // voices_mutex guards against a concurrent upload_ref mutation.
            {
                std::lock_guard<std::mutex> vlock(impl_->voices_mutex);
                if (!voice.empty()) {
                    auto it = std::find_if(impl_->voices.begin(), impl_->voices.end(), [&](const auto& v) {
                        return v.first == voice;
                    });
                    if (it == impl_->voices.end()) {
                        res.status = 400;
                        res.set_content("unknown voice name", "text/plain");
                        return;
                    }
                    ref_audio = it->second;
                }

                if (ref_audio.empty()) {
                    if (!impl_->voices.empty()) {
                        ref_audio = impl_->voices.front().second;
                    } else if (impl_->ref_is_file) {
                        ref_audio = impl_->config.default_ref_audio;
                    }
                }
            }

            if (text.empty()) {
                res.status = 400;
                res.set_content("text parameter required", "text/plain");
                return;
            }
            if (text.size() > kMaxTextBytes) {
                res.status = 413;
                res.set_content("text too long", "text/plain");
                return;
            }
            if (ref_audio.empty()) {
                res.status = 400;
                res.set_content("No reference voice available. Set a wav file or a folder with wav files.", "text/plain");
                return;
            }

            std::vector<float> pcm;
            {
                std::unique_lock<std::mutex> lock(impl_->synth_mutex, std::try_to_lock);
                if (!lock.owns_lock()) {
                    res.status = 429;
                    res.set_header("Retry-After", "2");
                    res.set_content("server busy, try again", "text/plain");
                    return;
                }
                pcm = impl_->engine.synthesize_pcm(ref_audio, text);
            }

            // Build WAV response (same as above, simplified)
            std::vector<int16_t> pcm16(pcm.size());
            for (size_t i = 0; i < pcm.size(); ++i) {
                float v = pcm[i] * WAV_SCALE;
                v = std::max(-WAV_SCALE, std::min(WAV_SCALE, v));
                pcm16[i] = static_cast<int16_t>(v);
            }

            uint32_t data_size = static_cast<uint32_t>(pcm.size() * 2);
            uint32_t file_size = 36 + data_size;
            std::string wav_data(44 + data_size, '\0');
            char* p = wav_data.data();

            std::memcpy(p, "RIFF", 4); p += 4;
            std::memcpy(p, &file_size, 4); p += 4;
            std::memcpy(p, "WAVE", 4); p += 4;
            std::memcpy(p, "fmt ", 4); p += 4;
            uint32_t fmt_size = 16;
            std::memcpy(p, &fmt_size, 4); p += 4;
            uint16_t audio_fmt = 1;
            std::memcpy(p, &audio_fmt, 2); p += 2;
            uint16_t num_ch = 1;
            std::memcpy(p, &num_ch, 2); p += 2;
            uint32_t sr = SAMPLE_RATE;
            std::memcpy(p, &sr, 4); p += 4;
            uint32_t byte_rate = SAMPLE_RATE * 2;
            std::memcpy(p, &byte_rate, 4); p += 4;
            uint16_t block_align = 2;
            std::memcpy(p, &block_align, 2); p += 2;
            uint16_t bps = 16;
            std::memcpy(p, &bps, 2); p += 2;
            std::memcpy(p, "data", 4); p += 4;
            std::memcpy(p, &data_size, 4); p += 4;
            std::memcpy(p, pcm16.data(), data_size);

            res.set_content(wav_data, "audio/wav");

        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(std::string("Error: ") + e.what(), "text/plain");
        }
    };
    srv.Get("/legado/tts", legado_tts_handler);
    srv.Post("/legado/tts", legado_tts_handler);

    return true;
}

void ApiServer::run() {
    auto& cfg = impl_->config;
    std::cerr << "[IndexTTS-Server] Listening on " << cfg.host << ":" << cfg.port << std::endl;
    impl_->server.listen(cfg.host, cfg.port);
}

void ApiServer::stop() {
    impl_->server.stop();
}

} // namespace indextts
