<p align="center">
  <img src="desktop/Assets/app_icon.png" width="96" alt="IndexTTS Server Icon"/>
</p>

<h1 align="center">IndexTTS-Server</h1>

<p align="center">
  纯 C++ ONNX Runtime（CUDA / TensorRT / CPU）<strong>IndexTTS 1.5</strong> 高速声音克隆服务端与原生 Windows WPF 桌面控制台
</p>

<p align="center">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

---

## 核心特性

- **纯 C++、零 Python 依赖（`engine/`）**
  - 基于 ONNX Runtime，支持 CUDA、TensorRT 与 CPU 推理。
  - 原生支持 **INT4+FP16 轻量化权重**（压缩包约 **923 MB**，相比原始 FP32 缩减 80% 以上，声码器 BigVGAN 保持 FP16 高保真），同时兼容 FP16 / FP32 权重。
  - **免填参考文本**：直接将 `.wav` 或 `.mp3` 参考音频放入 `refs/` 目录即可克隆音色。
- **标准 OpenAI 兼容 API + 原生接口**
  - 提供 `/v1/audio/speech`、`/v1/models`、`/v1/audio/voices` 与原生 `/v1/tts/*` 接口，支持可选 `--auth-token` Bearer 鉴权。
  - 可直接接入 **Open WebUI**、**SillyTavern**、**开源阅读（Legado）**、**CloneTTS** 等任意支持 OpenAI TTS 的客户端。
- **现代原生 Windows WPF 桌面控制台（`desktop/`）**
  - 一键切换 CUDA / TensorRT / CPU 启停服务、配置端口与访问令牌、一键复制局域网地址与模型 ID、快捷合成试听、暗色实时日志、系统托盘后台常驻，以及外置 JSON 多语言包（`locales/zh.json`、`locales/en.json`）。

## 目录结构

- `engine/` — C++17 ONNX Runtime 推理引擎与 HTTP 服务端（`CMakeLists.txt`、`include/`、`src/`）
- `desktop/` — 原生 Windows WPF (.NET 8) 桌面控制台（`IndexTTS-Desktop.csproj`、`locales/`）

## 从源码编译

### 1. C++ 推理服务端（`engine/`）
需要 CMake 3.20+、MSVC / C++17 编译器与 ONNX Runtime（CUDA 或 CPU 版）：
```powershell
cmake -S engine -B engine/build -DUSE_CUDA=ON -DONNXRUNTIME_DIR=<onnxruntime路径>
cmake --build engine/build --config Release
```

### 2. Windows 桌面控制台（`desktop/`）
需要 [.NET 8 SDK](https://dotnet.microsoft.com/)：
```powershell
dotnet build desktop/IndexTTS-Desktop.csproj -c Release
```

## 致谢与开源协议

- `engine/` 与 `desktop/` 源码采用 **Apache License 2.0** 开源协议（详见 [LICENSE](LICENSE)）。
- 上游模型与算法致谢：[index-tts/index-tts](https://github.com/index-tts/index-tts)（本仓库不含任何模型权重或个人音频，模型权重遵循上游许可协议）。
