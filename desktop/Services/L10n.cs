using System.Globalization;
using System.IO;
using System.Reflection;
using System.Text.Json;

namespace IndexTtsDesktop.Services;

public sealed record LocaleOption(string Code, string DisplayName);

public static class L10n
{
    private static readonly Dictionary<string, Dictionary<string, string>> Packs = new(StringComparer.OrdinalIgnoreCase);
    private static readonly Dictionary<string, string> DisplayNames = new(StringComparer.OrdinalIgnoreCase);

    public static string CurrentLang { get; private set; } = "en";

    static L10n()
    {
        ReloadLocales();
        var sys = CultureInfo.CurrentUICulture.TwoLetterISOLanguageName.ToLowerInvariant();
        CurrentLang = Packs.ContainsKey(sys) ? sys : (Packs.ContainsKey("en") ? "en" : Packs.Keys.FirstOrDefault() ?? "en");
    }

    public static IReadOnlyList<LocaleOption> AvailableLanguages =>
        DisplayNames
            .OrderBy(kv => kv.Key == "zh" ? 0 : (kv.Key == "en" ? 1 : 2))
            .ThenBy(kv => kv.Value, StringComparer.OrdinalIgnoreCase)
            .Select(kv => new LocaleOption(kv.Key, kv.Value))
            .ToList();

    public static void ReloadLocales()
    {
        Packs.Clear();
        DisplayNames.Clear();

        var asm = Assembly.GetExecutingAssembly();
        foreach (var resName in asm.GetManifestResourceNames())
        {
            if (!resName.EndsWith(".json", StringComparison.OrdinalIgnoreCase) || !resName.Contains(".locales.", StringComparison.OrdinalIgnoreCase))
                continue;
            var fileName = resName[(resName.IndexOf(".locales.", StringComparison.OrdinalIgnoreCase) + 9)..];
            var code = Path.GetFileNameWithoutExtension(fileName).ToLowerInvariant();
            using var stream = asm.GetManifestResourceStream(resName);
            if (stream is null) continue;
            using var reader = new StreamReader(stream);
            LoadPackJson(code, reader.ReadToEnd());
        }

        var externalDir = Path.Combine(AppContext.BaseDirectory, "locales");
        if (Directory.Exists(externalDir))
        {
            foreach (var file in Directory.GetFiles(externalDir, "*.json", SearchOption.TopDirectoryOnly))
            {
                try
                {
                    var code = Path.GetFileNameWithoutExtension(file).ToLowerInvariant();
                    LoadPackJson(code, File.ReadAllText(file));
                }
                catch { }
            }
        }
    }

    private static void LoadPackJson(string code, string json)
    {
        var dict = JsonSerializer.Deserialize<Dictionary<string, string>>(json);
        if (dict is null || dict.Count == 0) return;

        if (!Packs.TryGetValue(code, out var target))
        {
            target = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            Packs[code] = target;
        }
        foreach (var kv in dict) target[kv.Key] = kv.Value;

        DisplayNames[code] = dict.TryGetValue("_lang_name", out var name) && !string.IsNullOrWhiteSpace(name)
            ? name
            : code.ToUpperInvariant();
    }

    public static void SetLanguage(string? code)
    {
        if (!string.IsNullOrWhiteSpace(code) && Packs.ContainsKey(code))
            CurrentLang = code.ToLowerInvariant();
    }

    public static string Get(string key)
    {
        if (Packs.TryGetValue(CurrentLang, out var cur) && cur.TryGetValue(key, out var val)) return val;
        if (Packs.TryGetValue("en", out var en) && en.TryGetValue(key, out var enVal)) return enVal;
        return key;
    }

    public static string Format(string key, params object[] args) => string.Format(Get(key), args);
}
