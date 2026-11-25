using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;

namespace IndexTtsDesktop.Services;

public sealed class ModelDownloader
{
    public const string ModelScopePackageUrl =
        "https://www.modelscope.cn/models/ziansoft/indextts-1.5-onnx-int4/resolve/master/indextts-1.5-onnx-int4.zip";

    private static readonly HttpClient Http = new(new HttpClientHandler
    {
        AllowAutoRedirect = true,
        AutomaticDecompression = DecompressionMethods.None
    })
    {
        Timeout = Timeout.InfiniteTimeSpan
    };

    public async Task DownloadAndExtractPackageAsync(
        string modelsDir,
        Action<double, string> onProgress,
        CancellationToken ct)
    {
        Directory.CreateDirectory(modelsDir);
        var partPath = Path.Combine(modelsDir, ".indextts-1.5-onnx-int4.zip.part");
        long existing = File.Exists(partPath) ? new FileInfo(partPath).Length : 0;

        using var req = new HttpRequestMessage(HttpMethod.Get, ModelScopePackageUrl);
        req.Headers.UserAgent.ParseAdd("IndexTTS-Desktop/0.1");
        if (existing > 0) req.Headers.Range = new RangeHeaderValue(existing, null);

        using var resp = await Http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
        if (resp.StatusCode == HttpStatusCode.RequestedRangeNotSatisfiable)
        {
            existing = 0;
        }
        else if (!resp.IsSuccessStatusCode)
        {
            throw new HttpRequestException($"HTTP {(int)resp.StatusCode} ({resp.ReasonPhrase})");
        }

        bool isPartial = resp.StatusCode == HttpStatusCode.PartialContent;
        long total = isPartial && resp.Content.Headers.ContentRange?.Length is long fullLen
            ? fullLen
            : (resp.Content.Headers.ContentLength ?? -1);
        if (!isPartial) existing = 0;

        var mode = isPartial ? FileMode.Append : FileMode.Create;
        await using (var src = await resp.Content.ReadAsStreamAsync(ct))
        await using (var dst = new FileStream(partPath, mode, FileAccess.Write, FileShare.None, 256 * 1024, useAsync: true))
        {
            var buf = new byte[256 * 1024];
            long downloaded = existing;
            long sessionBytes = 0;
            var sw = Stopwatch.StartNew();
            var lastUi = TimeSpan.Zero;

            while (true)
            {
                int read = await src.ReadAsync(buf, ct);
                if (read == 0) break;
                await dst.WriteAsync(buf.AsMemory(0, read), ct);
                downloaded += read;
                sessionBytes += read;

                if (sw.Elapsed - lastUi >= TimeSpan.FromMilliseconds(200))
                {
                    lastUi = sw.Elapsed;
                    double speedMb = sessionBytes / 1024.0 / 1024.0 / Math.Max(sw.Elapsed.TotalSeconds, 0.1);
                    double pct = total > 0 ? Math.Min(95.0, downloaded * 95.0 / total) : 0;
                    string status = total > 0
                        ? $"{downloaded / 1024.0 / 1024.0:F0} / {total / 1024.0 / 1024.0:F0} MB ({speedMb:F1} MB/s)"
                        : $"{downloaded / 1024.0 / 1024.0:F1} MB ({speedMb:F1} MB/s)";
                    onProgress(pct, status);
                }
            }
        }

        onProgress(96, L10n.Get("models.status_extracting"));
        await Task.Run(() => ExtractZipToModelsDir(partPath, modelsDir), ct);
        try { File.Delete(partPath); } catch { }

        var (ready, summary) = EngineManager.InspectModelsDir(modelsDir);
        if (!ready) throw new InvalidDataException($"Model verification failed after extraction: {summary}");
        onProgress(100, L10n.Get("models.status_installed"));
    }

    public static void ExtractZipToModelsDir(string zipPath, string modelsDir)
    {
        Directory.CreateDirectory(modelsDir);
        using var archive = ZipFile.OpenRead(zipPath);
        foreach (var entry in archive.Entries)
        {
            if (string.IsNullOrEmpty(entry.Name)) continue;
            var rel = entry.FullName.Replace('\\', '/').TrimStart('/');
            if (rel.StartsWith("models/", StringComparison.OrdinalIgnoreCase))
                rel = rel["models/".Length..];
            if (string.IsNullOrWhiteSpace(rel) || rel.Contains("..")) continue;

            var dst = Path.Combine(modelsDir, rel.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(dst)!);
            entry.ExtractToFile(dst, overwrite: true);
        }
    }
}
