using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using IndexTtsDesktop.Models;

namespace IndexTtsDesktop.Services;

public sealed class EngineManager : IDisposable
{
    private readonly HttpClient _http = new() { Timeout = TimeSpan.FromSeconds(120) };
    private Process? _process;
    private readonly object _lock = new();

    public event Action<string>? LogReceived;
    public event Action<bool>? RunningStateChanged;

    public bool IsRunning
    {
        get
        {
            lock (_lock)
            {
                return _process is { HasExited: false };
            }
        }
    }

    public static string? DetectRuntimeRoot(string? savedRoot)
    {
        var candidates = new List<string>();
        if (!string.IsNullOrWhiteSpace(savedRoot)) candidates.Add(savedRoot);
        candidates.Add(AppContext.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        candidates.Add(Environment.CurrentDirectory);

        foreach (var dir in candidates.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (IsValidRuntimeRoot(dir)) return Path.GetFullPath(dir);
        }
        return null;
    }

    public static bool IsValidRuntimeRoot(string? dir)
    {
        if (string.IsNullOrWhiteSpace(dir) || !Directory.Exists(dir)) return false;
        return File.Exists(Path.Combine(dir, "engine", "indextts.exe"));
    }

    public static string ResolveModelsDir(string runtimeRoot, string? savedModelsDir)
    {
        if (!string.IsNullOrWhiteSpace(savedModelsDir) && Directory.Exists(savedModelsDir))
            return Path.GetFullPath(savedModelsDir);
        return Path.GetFullPath(Path.Combine(runtimeRoot, "models"));
    }

    public static string ResolveRefsDir(string runtimeRoot, string? savedRefsDir)
    {
        if (!string.IsNullOrWhiteSpace(savedRefsDir) && Directory.Exists(savedRefsDir))
            return Path.GetFullPath(savedRefsDir);
        var dir = Path.GetFullPath(Path.Combine(runtimeRoot, "refs"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    public static (bool Ready, string Summary) InspectModelsDir(string modelsDir)
    {
        if (!Directory.Exists(modelsDir)) return (false, "models/ not found");
        var required = new[] { "bpe.model", "cond_encoder.onnx", "bigvgan.onnx", "gpt_prefill.onnx" };
        var missing = required.Where(f => !File.Exists(Path.Combine(modelsDir, f))).ToList();
        var hasDecode = File.Exists(Path.Combine(modelsDir, "gpt_decode_opt.onnx"))
                        || File.Exists(Path.Combine(modelsDir, "gpt_decode.onnx"));
        if (!hasDecode) missing.Add("gpt_decode(_opt).onnx");

        if (missing.Count > 0) return (false, $"Missing: {string.Join(", ", missing)}");

        var decodeFile = File.Exists(Path.Combine(modelsDir, "gpt_decode_opt.onnx"))
            ? new FileInfo(Path.Combine(modelsDir, "gpt_decode_opt.onnx"))
            : new FileInfo(Path.Combine(modelsDir, "gpt_decode.onnx"));
        var precision = decodeFile.Length < 600_000_000 ? "INT4+FP16 (~1.0 GB)"
            : (decodeFile.Length < 1_200_000_000 ? "FP16 (~2.2 GB)" : "FP32 (~6.3 GB)");
        return (true, $"IndexTTS 1.5 ONNX [{precision}]");
    }

    public static string GetLanIpAddress()
    {
        try
        {
            foreach (var ni in NetworkInterface.GetAllNetworkInterfaces())
            {
                if (ni.OperationalStatus != OperationalStatus.Up) continue;
                var name = ni.Name;
                if (name.Contains("Loopback", StringComparison.OrdinalIgnoreCase)
                    || name.Contains("vEthernet", StringComparison.OrdinalIgnoreCase)
                    || name.Contains("WSL", StringComparison.OrdinalIgnoreCase)
                    || name.Contains("VMware", StringComparison.OrdinalIgnoreCase)
                    || name.Contains("Meta", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }
                foreach (var ua in ni.GetIPProperties().UnicastAddresses)
                {
                    if (ua.Address.AddressFamily != AddressFamily.InterNetwork) continue;
                    var ip = ua.Address.ToString();
                    if (!ip.StartsWith("169.254.") && !IPAddress.IsLoopback(ua.Address)) return ip;
                }
            }
        }
        catch { }
        return "127.0.0.1";
    }

    public async Task<bool> StartAsync(string runtimeRoot, string modelsDir, string refsDir, DesktopSettings settings, CancellationToken ct = default)
    {
        if (IsRunning) return true;

        var engineExe = Path.Combine(runtimeRoot, "engine", "indextts.exe");
        var bpeModel = Path.Combine(modelsDir, "bpe.model");
        if (!File.Exists(engineExe))
        {
            LogReceived?.Invoke($"[ERROR] Engine not found: {engineExe}");
            return false;
        }
        var (ready, summary) = InspectModelsDir(modelsDir);
        if (!ready)
        {
            LogReceived?.Invoke($"[ERROR] Models incomplete in {modelsDir}: {summary}");
            return false;
        }

        Directory.CreateDirectory(refsDir);
        var args = new StringBuilder();
        args.Append("--server ");
        args.Append($"--model-dir \"{modelsDir}\" ");
        args.Append($"--bpe-model \"{bpeModel}\" ");
        args.Append("--host 0.0.0.0 ");
        args.Append($"--port {settings.Port} ");
        args.Append($"--ref-audio \"{refsDir}\" ");
        if (string.Equals(settings.Backend, "tensorrt", StringComparison.OrdinalIgnoreCase))
        {
            args.Append("--tensorrt ");
            var trtCache = Path.Combine(runtimeRoot, "trt_cache");
            Directory.CreateDirectory(trtCache);
            args.Append($"--trt-cache \"{trtCache}\" ");
        }
        else if (string.Equals(settings.Backend, "cuda", StringComparison.OrdinalIgnoreCase))
        {
            args.Append("--cuda ");
        }
        if (settings.Threads > 0) args.Append($"--threads {settings.Threads} ");
        if (!string.IsNullOrWhiteSpace(settings.AuthToken))
        {
            args.Append($"--auth-token \"{settings.AuthToken.Trim()}\" ");
        }

        var engineDir = Path.Combine(runtimeRoot, "engine");
        var psi = new ProcessStartInfo
        {
            FileName = engineExe,
            Arguments = args.ToString(),
            WorkingDirectory = runtimeRoot,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8
        };
        var existingPath = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
        psi.Environment["PATH"] = $"{engineDir};{existingPath}";

        LogReceived?.Invoke($"[INFO] Starting indextts.exe ({settings.Backend.ToUpperInvariant()}, port={settings.Port}, {summary})");
        var proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
        proc.OutputDataReceived += (_, e) => { if (!string.IsNullOrWhiteSpace(e.Data)) LogReceived?.Invoke(e.Data); };
        proc.ErrorDataReceived += (_, e) => { if (!string.IsNullOrWhiteSpace(e.Data)) LogReceived?.Invoke(e.Data); };
        proc.Exited += (_, _) =>
        {
            LogReceived?.Invoke($"[INFO] indextts.exe exited (code={proc.ExitCode})");
            RunningStateChanged?.Invoke(false);
        };

        lock (_lock) { _process = proc; }

        if (!proc.Start())
        {
            LogReceived?.Invoke("[ERROR] Failed to launch indextts.exe");
            RunningStateChanged?.Invoke(false);
            return false;
        }

        proc.BeginOutputReadLine();
        proc.BeginErrorReadLine();
        RunningStateChanged?.Invoke(true);

        // Wait up to 30s for model loading and /v1/health readiness
        var probeUrl = $"http://127.0.0.1:{settings.Port}/v1/health";
        for (var i = 0; i < 60; i++)
        {
            if (ct.IsCancellationRequested || proc.HasExited) return false;
            try
            {
                using var req = new HttpRequestMessage(HttpMethod.Get, probeUrl);
                if (!string.IsNullOrWhiteSpace(settings.AuthToken))
                    req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", settings.AuthToken.Trim());
                using var resp = await _http.SendAsync(req, ct);
                if ((int)resp.StatusCode < 500)
                {
                    LogReceived?.Invoke($"[READY] Server ready at http://{GetLanIpAddress()}:{settings.Port}");
                    return true;
                }
            }
            catch { }
            await Task.Delay(500, ct);
        }

        LogReceived?.Invoke("[WARN] Engine process is running, still loading models — watch the logs below.");
        return !proc.HasExited;
    }

    public void Stop(bool silent = false)
    {
        Process? proc;
        lock (_lock)
        {
            proc = _process;
            _process = null;
        }
        if (proc is null) return;
        try
        {
            proc.EnableRaisingEvents = false;
            if (!proc.HasExited)
            {
                if (!silent) LogReceived?.Invoke("[INFO] Stopping indextts.exe and releasing VRAM...");
                proc.Kill(entireProcessTree: true);
                proc.WaitForExit(1500);
            }
        }
        catch (Exception ex)
        {
            if (!silent) LogReceived?.Invoke($"[WARN] Stop error: {ex.Message}");
        }
        finally
        {
            try { proc.Dispose(); } catch { }
            if (!silent) RunningStateChanged?.Invoke(false);
        }
    }

    public async Task<byte[]> SynthesizeTestWavAsync(int port, string voiceName, string text, string? authToken, CancellationToken ct = default)
    {
        var payload = new Dictionary<string, object>
        {
            ["model"] = "indextts-1.5",
            ["input"] = text,
            ["voice"] = voiceName,
            ["response_format"] = "wav"
        };
        using var req = new HttpRequestMessage(HttpMethod.Post, $"http://127.0.0.1:{port}/v1/audio/speech")
        {
            Content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json")
        };
        if (!string.IsNullOrWhiteSpace(authToken))
            req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", authToken.Trim());

        using var resp = await _http.SendAsync(req, ct);
        if (!resp.IsSuccessStatusCode)
        {
            var err = await resp.Content.ReadAsStringAsync(ct);
            throw new InvalidOperationException($"HTTP {(int)resp.StatusCode}: {err}");
        }
        return await resp.Content.ReadAsByteArrayAsync(ct);
    }

    public void Dispose()
    {
        Stop(silent: true);
        _http.Dispose();
    }
}
