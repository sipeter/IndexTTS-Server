<p align="center">
  <img src="desktop/Assets/app_icon.png" width="96" alt="IndexTTS Server Icon"/>
</p>

<h1 align="center">IndexTTS-Server</h1>

<p align="center">
  Pure C++ ONNX Runtime (CUDA / TensorRT / CPU) inference server &amp; native Windows WPF desktop console for <strong>IndexTTS 1.5</strong> voice cloning
</p>

<p align="center">
  <strong>English</strong> | <a href="README_ZH.md">简体中文</a>
</p>

---

## Highlights

- **Pure C++ & Zero Python Runtime (`engine/`)**
  - Powered by ONNX Runtime with CUDA, TensorRT, and CPU execution providers.
  - Supports **INT4+FP16** lightweight ONNX weights (~923 MB zip / 1.11 GB extracted — over 80% smaller than raw FP32 while preserving FP16 BigVGAN vocoder fidelity) as well as FP16 / FP32 ONNX weights.
  - **No reference transcript required**: drop any `.wav` or `.mp3` reference clip into `refs/` and synthesize immediately.
- **OpenAI-Compatible & Native HTTP API**
  - Standard `/v1/audio/speech`, `/v1/models`, `/v1/audio/voices` plus native `/v1/tts/*` endpoints and optional `--auth-token` Bearer protection.
  - Works out of the box with **Open WebUI**, **SillyTavern**, **Legado**, **CloneTTS**, and any OpenAI TTS client.
- **Modern Native Windows WPF Console (`desktop/`)**
  - One-click CUDA / TensorRT / CPU start & stop, optional Auth Token, LAN URL & Model ID copy, quick synthesis preview, live logs, deadlock-free system tray, and extensible external JSON language packs (`locales/en.json`, `locales/zh.json`).

## Repository Layout

- `engine/` — C++17 ONNX Runtime inference engine & HTTP server (`CMakeLists.txt`, `include/`, `src/`)
- `desktop/` — Native Windows WPF (.NET 8) desktop console (`IndexTTS-Desktop.csproj`, `locales/`)

## Build from Source

### 1. C++ Engine (`engine/`)
Requires CMake 3.20+, MSVC / C++17 compiler, and ONNX Runtime (CUDA or CPU):
```powershell
cmake -S engine -B engine/build -DUSE_CUDA=ON -DONNXRUNTIME_DIR=<path_to_onnxruntime>
cmake --build engine/build --config Release
```

### 2. Windows Desktop Console (`desktop/`)
Requires [.NET 8 SDK](https://dotnet.microsoft.com/):
```powershell
dotnet build desktop/IndexTTS-Desktop.csproj -c Release
```

## Credits & License

- Source code in `engine/` and `desktop/` is licensed under the **Apache License 2.0** (see [LICENSE](LICENSE)).
- Upstream IndexTTS 1.5 model architecture & weights: [index-tts/index-tts](https://github.com/index-tts/index-tts) (model weights are not bundled in this repo and follow their upstream license).
