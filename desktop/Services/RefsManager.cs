using System.IO;
using IndexTtsDesktop.Models;

namespace IndexTtsDesktop.Services;

public static class RefsManager
{
    private static readonly HashSet<string> SupportedExts = new(StringComparer.OrdinalIgnoreCase) { ".wav", ".mp3" };

    public static List<RefVoiceItem> LoadRefs(string refsDir)
    {
        Directory.CreateDirectory(refsDir);
        var list = new List<RefVoiceItem>();
        foreach (var file in Directory.GetFiles(refsDir, "*.*", SearchOption.TopDirectoryOnly).OrderBy(f => f, StringComparer.OrdinalIgnoreCase))
        {
            var ext = Path.GetExtension(file);
            if (!SupportedExts.Contains(ext)) continue;
            var info = new FileInfo(file);
            list.Add(new RefVoiceItem
            {
                Name = Path.GetFileNameWithoutExtension(file),
                FileName = info.Name,
                FilePath = file,
                Format = ext.TrimStart('.').ToUpperInvariant(),
                SizeText = info.Length >= 1024 * 1024
                    ? $"{info.Length / 1024.0 / 1024.0:F2} MB"
                    : $"{info.Length / 1024.0:F0} KB"
            });
        }
        return list;
    }

    public static string ImportRef(string refsDir, string sourceFile)
    {
        Directory.CreateDirectory(refsDir);
        var ext = Path.GetExtension(sourceFile);
        if (!SupportedExts.Contains(ext))
            throw new InvalidOperationException($"Unsupported audio format '{ext}'. Please select a .wav or .mp3 file.");
        var dst = Path.Combine(refsDir, Path.GetFileName(sourceFile));
        File.Copy(sourceFile, dst, overwrite: true);
        return dst;
    }
}
