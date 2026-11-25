using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Media;
using System.Reflection;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using IndexTtsDesktop.Models;
using IndexTtsDesktop.Services;
using Forms = System.Windows.Forms;

namespace IndexTtsDesktop;

public partial class MainWindow : Window
{
    private readonly string _settingsPath = Path.Combine(AppContext.BaseDirectory, "desktop_settings.json");
    private readonly EngineManager _engine = new();
    private readonly ModelDownloader _downloader = new();
    private readonly ObservableCollection<RefVoiceItem> _voices = new();

    private CancellationTokenSource? _downloadCts;
    private DesktopSettings _settings = new();
    private Forms.NotifyIcon? _trayIcon;
    private Forms.ToolStripMenuItem? _trayShowItem;
    private Forms.ToolStripMenuItem? _trayExitItem;
    private bool _forceExit;
    private bool _isShuttingDown;
    private bool _initializing = true;
    private string _lastDefaultTestText = string.Empty;

    public MainWindow()
    {
        InitializeComponent();
        VoicesGrid.ItemsSource = _voices;

        _engine.LogReceived += line =>
        {
            if (_isShuttingDown) return;
            Dispatcher.BeginInvoke(() => { if (!_isShuttingDown) AppendLog(line); });
        };
        _engine.RunningStateChanged += running =>
        {
            if (_isShuttingDown) return;
            Dispatcher.BeginInvoke(() => { if (!_isShuttingDown) UpdateRunningUi(running); });
        };

        LoadSettings();
        L10n.SetLanguage(_settings.Language);
        InitTrayIcon();
        ApplySettingsToUi();
        ApplyLanguage();
        _initializing = false;

        RefreshWorkspaceState();
        AppendLog(L10n.Get("msg.ready"));
    }

    private static string AppVersionString
    {
        get
        {
            var v = Assembly.GetExecutingAssembly().GetName().Version;
            return v is null ? "v0.1.0" : $"v{v.Major}.{v.Minor}.{v.Build}";
        }
    }

    private void InitTrayIcon()
    {
        var menu = new Forms.ContextMenuStrip();
        _trayShowItem = new Forms.ToolStripMenuItem("", null, (_, _) => Dispatcher.BeginInvoke(RestoreFromTray));
        _trayExitItem = new Forms.ToolStripMenuItem("", null, (_, _) => Dispatcher.BeginInvoke(ExitCompletely));
        menu.Items.Add(_trayShowItem);
        menu.Items.Add(new Forms.ToolStripSeparator());
        menu.Items.Add(_trayExitItem);

        Icon trayIcon = SystemIcons.Application;
        try
        {
            if (!string.IsNullOrEmpty(Environment.ProcessPath) && File.Exists(Environment.ProcessPath))
                trayIcon = System.Drawing.Icon.ExtractAssociatedIcon(Environment.ProcessPath) ?? SystemIcons.Application;
        }
        catch { }

        _trayIcon = new Forms.NotifyIcon
        {
            Text = "IndexTTS Server",
            Icon = trayIcon,
            ContextMenuStrip = menu,
            Visible = true
        };
        _trayIcon.DoubleClick += (_, _) => Dispatcher.BeginInvoke(RestoreFromTray);
    }

    private void ApplyLanguage()
    {
        var baseName = L10n.Get("app.name");
        Title = $"{baseName} {AppVersionString}";
        TxtAppTitle.Text = baseName;
        TxtVersionBadge.Text = AppVersionString;
        LblRootDir.Text = L10n.Get("header.runtime_dir");
        BtnBrowseRoot.Content = L10n.Get("header.browse");
        BtnRefreshHeader.Content = L10n.Get("header.refresh");
        BtnStart.Content = L10n.Get("header.start");
        BtnStop.Content = L10n.Get("header.stop");

        TabServer.Header = L10n.Get("tab.server");
        TabModels.Header = L10n.Get("tab.models");
        TabVoices.Header = L10n.Get("tab.voices");
        TabAbout.Header = L10n.Get("tab.about");

        LblModelsHeading.Text = L10n.Get("models.heading");
        LblPkgTitle.Text = L10n.Get("models.pkg_title");
        LblPkgDesc.Text = L10n.Get("models.pkg_desc");
        LblModelSource.Text = L10n.Get("models.source_label");
        BtnImportModelZip.Content = L10n.Get("models.import_zip");
        BtnBrowseModelsTab.Content = L10n.Get("models.change_dir");
        BtnOpenModelsTab.Content = L10n.Get("models.open_dir");
        BtnRefreshModels.Content = L10n.Get("header.refresh");
        UpdateDownloadButtonUi();

        LblServerSettings.Text = L10n.Get("server.settings_title");
        LblBackend.Text = L10n.Get("server.backend");
        CbiCuda.Content = L10n.Get("server.backend_cuda");
        CbiTrt.Content = L10n.Get("server.backend_trt");
        CbiCpu.Content = L10n.Get("server.backend_cpu");
        LblPort.Text = L10n.Get("server.port");
        LblAuthToken.Text = L10n.Get("server.auth_token");
        TxtAuthToken.ToolTip = L10n.Get("server.auth_token_tip");
        ChkMinimizeToTray.Content = L10n.Get("server.minimize_tray");

        LblApiTitle.Text = L10n.Get("api.title");
        LblLanUrl.Text = L10n.Get("api.lan_url");
        BtnCopyLanUrl.Content = L10n.Get("api.copy_url");
        LblModelId.Text = L10n.Get("api.model_id");
        BtnCopyModelId.Content = L10n.Get("api.copy_id");
        BtnBrowseModels.Content = L10n.Get("models.change_dir");
        BtnOpenModels.Content = L10n.Get("models.open_dir");

        LblTestVoice.Text = L10n.Get("test.voice");
        var newDefault = L10n.Get("test.default_text");
        if (string.IsNullOrWhiteSpace(TxtTestInput.Text) || TxtTestInput.Text == _lastDefaultTestText || TxtTestInput.Text == "你好，欢迎使用 IndexTTS 1.5 本地语音合成服务。")
            TxtTestInput.Text = newDefault;
        _lastDefaultTestText = newDefault;
        BtnTestSpeak.Content = L10n.Get("test.speak");

        LblLiveLogs.Text = L10n.Get("logs.title");
        BtnClearLogs.Content = L10n.Get("logs.clear");

        BtnImportVoice.Content = L10n.Get("voices.import");
        BtnBrowseRefsDir.Content = L10n.Get("voices.change_dir");
        BtnOpenRefsDir.Content = L10n.Get("voices.open_dir");
        BtnRefreshVoices.Content = L10n.Get("header.refresh");
        ColVoiceName.Header = L10n.Get("voices.col_name");
        ColVoiceFile.Header = L10n.Get("voices.col_file");
        ColVoiceFormat.Header = L10n.Get("voices.col_format");
        ColVoiceSize.Header = L10n.Get("voices.col_size");
        ColVoicePlay.Header = L10n.Get("voices.col_play");

        AboutTitle.Text = baseName;
        AboutVersionBadge.Text = AppVersionString;
        AboutDesc.Text = L10n.Get("about.desc");
        AboutLinksHeader.Text = L10n.Get("about.links_header");
        LblCardSelf.Text = L10n.Get("about.link_self");
        LblCardUpstream.Text = L10n.Get("about.link_upstream");
        AboutLicenseHeader.Text = L10n.Get("about.license_header");
        AboutLicenseBody.Text = L10n.Get("about.license_body");
        AboutClientsLabel.Text = L10n.Get("about.clients_label");

        if (_trayShowItem is not null) _trayShowItem.Text = L10n.Get("tray.show");
        if (_trayExitItem is not null) _trayExitItem.Text = L10n.Get("tray.exit");
        if (_trayIcon is not null) _trayIcon.Text = baseName;

        UpdateRunningUi(_engine.IsRunning);
        RefreshWorkspaceState();
    }

    private void Language_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_initializing) return;
        if (CmbLanguage.SelectedValue is string lang)
        {
            _settings.Language = lang;
            L10n.SetLanguage(lang);
            SaveSettings();
            ApplyLanguage();
        }
    }

    private void RestoreFromTray()
    {
        Show();
        WindowState = WindowState.Normal;
        Activate();
    }

    private void ExitCompletely()
    {
        if (_isShuttingDown) return;
        _isShuttingDown = true;
        _forceExit = true;
        try { _downloadCts?.Cancel(); } catch { }

        if (_trayIcon is not null)
        {
            _trayIcon.Visible = false;
            _trayIcon.Dispose();
            _trayIcon = null;
        }
        try { _engine.Dispose(); } catch { }
        System.Windows.Application.Current.Shutdown();
    }

    private void Window_Closing(object? sender, CancelEventArgs e)
    {
        if (!_forceExit && _settings.MinimizeToTrayOnClose)
        {
            e.Cancel = true;
            Hide();
            _trayIcon?.ShowBalloonTip(2000, L10n.Get("app.name"), L10n.Get("tray.minimized_tip"), Forms.ToolTipIcon.Info);
            return;
        }
        ExitCompletely();
    }

    private void LoadSettings()
    {
        try
        {
            if (File.Exists(_settingsPath))
            {
                var loaded = JsonSerializer.Deserialize<DesktopSettings>(File.ReadAllText(_settingsPath));
                if (loaded is not null) _settings = loaded;
            }
        }
        catch { _settings = new DesktopSettings(); }

        if (string.IsNullOrWhiteSpace(_settings.Language)) _settings.Language = L10n.CurrentLang;
        var detected = EngineManager.DetectRuntimeRoot(_settings.RuntimeRootDir);
        if (detected is not null) _settings.RuntimeRootDir = detected;
    }

    private void SaveSettings()
    {
        try
        {
            File.WriteAllText(_settingsPath, JsonSerializer.Serialize(_settings, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch { }
    }

    private void ApplySettingsToUi()
    {
        CmbLanguage.ItemsSource = L10n.AvailableLanguages;
        CmbLanguage.SelectedValue = L10n.CurrentLang;
        CmbBackend.SelectedIndex = _settings.Backend.ToLowerInvariant() switch
        {
            "tensorrt" => 1,
            "cpu" => 2,
            _ => 0
        };
        TxtPort.Text = _settings.Port.ToString();
        TxtAuthToken.Text = _settings.AuthToken;
        ChkMinimizeToTray.IsChecked = _settings.MinimizeToTrayOnClose;
        UpdateLanUrlBox();
    }

    private void ReadSettingsFromUi()
    {
        if (_initializing) return;
        if (CmbBackend.SelectedItem is ComboBoxItem bi && bi.Tag is string backend) _settings.Backend = backend;
        if (int.TryParse(TxtPort.Text.Trim(), out var port) && port is >= 1024 and <= 65535) _settings.Port = port;
        _settings.AuthToken = TxtAuthToken.Text.Trim();
        _settings.MinimizeToTrayOnClose = ChkMinimizeToTray.IsChecked == true;
        UpdateLanUrlBox();
        SaveSettings();
    }

    private void UpdateLanUrlBox() => TxtLanUrl.Text = $"http://{EngineManager.GetLanIpAddress()}:{_settings.Port}";

    private string ActiveModelsDir => EngineManager.ResolveModelsDir(_settings.RuntimeRootDir, _settings.ModelsDir);
    private string ActiveRefsDir => EngineManager.ResolveRefsDir(_settings.RuntimeRootDir, _settings.RefsDir);

    private void RefreshWorkspaceState()
    {
        var valid = EngineManager.IsValidRuntimeRoot(_settings.RuntimeRootDir);
        RootPathText.Text = valid ? _settings.RuntimeRootDir : L10n.Get("msg.root_not_found");
        RootPathText.Foreground = valid
            ? (System.Windows.Media.Brush)FindResource("TextPrimary")
            : (System.Windows.Media.Brush)FindResource("AccentRed");

        _voices.Clear();
        if (!valid)
        {
            TxtModelStatus.Text = L10n.Get("msg.select_root_first");
            RefsHintText.Text = L10n.Get("msg.select_root_first");
            return;
        }

        var modelsDir = ActiveModelsDir;
        var refsDir = ActiveRefsDir;
        var (ready, summary) = EngineManager.InspectModelsDir(modelsDir);
        TxtModelStatus.Text = ready ? $"✓ {summary}" : $"⚠ {summary}";
        TxtModelStatus.ToolTip = modelsDir;
        TxtModelStatus.Foreground = (System.Windows.Media.Brush)FindResource(ready ? "AccentGreen" : "AccentAmber");
        LblModelDirInfo.Text = modelsDir;

        if (_downloadCts is null)
        {
            PbModelDownload.Value = ready ? 100 : 0;
            TxtModelDownloadStatus.Text = ready ? L10n.Get("models.status_installed") : L10n.Get("models.status_missing");
            UpdateDownloadButtonUi(ready);
        }

        RefsHintText.Text = L10n.Format("voices.dir_hint", refsDir);
        foreach (var v in RefsManager.LoadRefs(refsDir)) _voices.Add(v);

        var prevVoice = CmbTestVoice.SelectedItem as string;
        CmbTestVoice.ItemsSource = _voices.Select(v => v.Name).ToList();
        if (prevVoice is not null && _voices.Any(v => v.Name == prevVoice))
            CmbTestVoice.SelectedItem = prevVoice;
        else if (_voices.Count > 0)
            CmbTestVoice.SelectedIndex = 0;

        UpdateLanUrlBox();
    }

    private void UpdateRunningUi(bool running)
    {
        BtnStart.IsEnabled = !running;
        BtnStop.IsEnabled = running;
        StatusText.Text = running ? L10n.Get("status.running") : L10n.Get("status.stopped");
        StatusText.Foreground = (System.Windows.Media.Brush)FindResource(running ? "AccentGreen" : "AccentRed");
        StatusBadge.Background = new SolidColorBrush(running ? System.Windows.Media.Color.FromRgb(220, 252, 231) : System.Windows.Media.Color.FromRgb(254, 226, 226));
    }

    private void AppendLog(string line)
    {
        TxtLogs.AppendText($"[{DateTime.Now:HH:mm:ss}] {line}{Environment.NewLine}");
        if (TxtLogs.Text.Length > 80_000) TxtLogs.Text = TxtLogs.Text[^40_000..];
        TxtLogs.ScrollToEnd();
    }

    private void SettingsInput_Changed(object sender, RoutedEventArgs e) => ReadSettingsFromUi();
    private void SettingsInput_LostFocus(object sender, RoutedEventArgs e) => ReadSettingsFromUi();

    private void BrowseRoot_Click(object sender, RoutedEventArgs e)
    {
        using var dlg = new Forms.FolderBrowserDialog { Description = L10n.Get("msg.browse_root"), UseDescriptionForTitle = true };
        if (Directory.Exists(_settings.RuntimeRootDir)) dlg.SelectedPath = _settings.RuntimeRootDir;
        if (dlg.ShowDialog() == Forms.DialogResult.OK)
        {
            if (!EngineManager.IsValidRuntimeRoot(dlg.SelectedPath))
            {
                System.Windows.MessageBox.Show(L10n.Get("msg.invalid_root"), L10n.Get("msg.invalid_dir_title"), MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
            _settings.RuntimeRootDir = dlg.SelectedPath;
            SaveSettings();
            RefreshWorkspaceState();
            AppendLog($"[INFO] Runtime dir: {_settings.RuntimeRootDir}");
        }
    }

    private void RefreshAll_Click(object sender, RoutedEventArgs e)
    {
        L10n.ReloadLocales();
        _initializing = true;
        CmbLanguage.ItemsSource = L10n.AvailableLanguages;
        CmbLanguage.SelectedValue = L10n.CurrentLang;
        _initializing = false;
        RefreshWorkspaceState();
        AppendLog("[INFO] Refreshed.");
    }

    private async void StartServer_Click(object sender, RoutedEventArgs e)
    {
        ReadSettingsFromUi();
        if (!EngineManager.IsValidRuntimeRoot(_settings.RuntimeRootDir))
        {
            System.Windows.MessageBox.Show(L10n.Get("msg.select_root_first"), L10n.Get("msg.notice"), MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        BtnStart.IsEnabled = false;
        StatusText.Text = L10n.Get("status.starting");
        StatusText.Foreground = (System.Windows.Media.Brush)FindResource("AccentAmber");
        StatusBadge.Background = new SolidColorBrush(System.Windows.Media.Color.FromRgb(254, 243, 199));

        var ok = await _engine.StartAsync(_settings.RuntimeRootDir, ActiveModelsDir, ActiveRefsDir, _settings);
        UpdateRunningUi(_engine.IsRunning);
        if (!ok && !_engine.IsRunning)
        {
            System.Windows.MessageBox.Show(L10n.Get("msg.start_failed_body"), L10n.Get("msg.start_failed_title"), MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void StopServer_Click(object sender, RoutedEventArgs e) => _engine.Stop();
    private void ClearLogs_Click(object sender, RoutedEventArgs e) => TxtLogs.Clear();

    private void CopyLanUrl_Click(object sender, RoutedEventArgs e)
    {
        System.Windows.Clipboard.SetText(TxtLanUrl.Text);
        AppendLog($"[INFO] Copied URL: {TxtLanUrl.Text}");
    }

    private void CopyModelId_Click(object sender, RoutedEventArgs e)
    {
        System.Windows.Clipboard.SetText("indextts-1.5");
        AppendLog("[INFO] Copied Model ID: indextts-1.5");
    }

    private async void TestSpeak_Click(object sender, RoutedEventArgs e)
    {
        if (!_engine.IsRunning)
        {
            System.Windows.MessageBox.Show(L10n.Get("msg.start_server_first"), L10n.Get("msg.server_stopped_title"), MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }
        var voice = CmbTestVoice.SelectedItem as string;
        var text = TxtTestInput.Text.Trim();
        if (string.IsNullOrEmpty(voice) || string.IsNullOrEmpty(text))
        {
            System.Windows.MessageBox.Show(L10n.Get("msg.select_voice_text"), L10n.Get("msg.notice"), MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        BtnTestSpeak.IsEnabled = false;
        BtnTestSpeak.Content = L10n.Get("test.synthesizing");
        try
        {
            AppendLog($"[TEST] voice={voice}");
            var wavBytes = await _engine.SynthesizeTestWavAsync(_settings.Port, voice, text, _settings.AuthToken);
            AppendLog($"[TEST] OK ({wavBytes.Length / 1024.0:F1} KB)");
            using var ms = new MemoryStream(wavBytes);
            using var player = new SoundPlayer(ms);
            player.PlaySync();
        }
        catch (Exception ex)
        {
            AppendLog($"[ERROR] {ex.Message}");
            System.Windows.MessageBox.Show(ex.Message, L10n.Get("msg.synth_failed"), MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            BtnTestSpeak.IsEnabled = true;
            BtnTestSpeak.Content = L10n.Get("test.speak");
        }
    }

    private void UpdateDownloadButtonUi(bool? isReadyOverride = null)
    {
        if (_downloadCts is not null)
        {
            BtnDownloadAllModels.Content = L10n.Get("models.cancel_download");
            BtnDownloadAllModels.IsEnabled = true;
            BtnImportModelZip.IsEnabled = false;
            return;
        }
        bool ready = isReadyOverride ?? (EngineManager.IsValidRuntimeRoot(_settings.RuntimeRootDir) && EngineManager.InspectModelsDir(ActiveModelsDir).Ready);
        BtnDownloadAllModels.Content = ready ? L10n.Get("models.all_ready") : L10n.Get("models.download_all");
        BtnDownloadAllModels.IsEnabled = !ready;
        BtnImportModelZip.IsEnabled = true;
    }

    private async void DownloadAllModels_Click(object sender, RoutedEventArgs e)
    {
        if (_downloadCts is not null)
        {
            _downloadCts.Cancel();
            return;
        }
        if (!EngineManager.IsValidRuntimeRoot(_settings.RuntimeRootDir))
        {
            System.Windows.MessageBox.Show(L10n.Get("msg.select_root_first"), L10n.Get("msg.notice"), MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        var modelsDir = ActiveModelsDir;
        _downloadCts = new CancellationTokenSource();
        UpdateDownloadButtonUi(false);
        TxtModelDownloadStatus.Text = L10n.Get("models.status_connecting");
        AppendLog($"[DL] Downloading IndexTTS 1.5 model package -> {modelsDir}");

        try
        {
            await _downloader.DownloadAndExtractPackageAsync(modelsDir, (pct, msg) =>
            {
                Dispatcher.BeginInvoke(() =>
                {
                    PbModelDownload.Value = pct;
                    TxtModelDownloadStatus.Text = msg;
                });
            }, _downloadCts.Token);
            AppendLog("[DL] Model package downloaded, extracted, and verified.");
        }
        catch (OperationCanceledException)
        {
            TxtModelDownloadStatus.Text = L10n.Get("models.status_paused");
            AppendLog("[DL] Download paused (resumable).");
        }
        catch (Exception ex)
        {
            TxtModelDownloadStatus.Text = ex.Message;
            AppendLog($"[ERROR] Model download failed: {ex.Message}");
            System.Windows.MessageBox.Show(ex.Message, L10n.Get("msg.notice"), MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            _downloadCts?.Dispose();
            _downloadCts = null;
            RefreshWorkspaceState();
        }
    }

    private async void ImportModelZip_Click(object sender, RoutedEventArgs e)
    {
        if (!EngineManager.IsValidRuntimeRoot(_settings.RuntimeRootDir))
        {
            System.Windows.MessageBox.Show(L10n.Get("msg.select_root_first"), L10n.Get("msg.notice"), MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        var dlg = new Microsoft.Win32.OpenFileDialog
        {
            Title = L10n.Get("models.import_zip"),
            Filter = "ZIP Archive (*.zip)|*.zip|All Files|*.*"
        };
        if (dlg.ShowDialog() != true) return;

        BtnImportModelZip.IsEnabled = false;
        BtnDownloadAllModels.IsEnabled = false;
        TxtModelDownloadStatus.Text = L10n.Get("models.status_extracting");
        PbModelDownload.Value = 50;
        try
        {
            await Task.Run(() => ModelDownloader.ExtractZipToModelsDir(dlg.FileName, ActiveModelsDir));
            AppendLog($"[MODELS] Extracted local model package into {ActiveModelsDir}");
        }
        catch (Exception ex)
        {
            System.Windows.MessageBox.Show(ex.Message, L10n.Get("msg.import_failed"), MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            RefreshWorkspaceState();
        }
    }

    private void BrowseModelsDir_Click(object sender, RoutedEventArgs e)
    {
        using var dlg = new Forms.FolderBrowserDialog { Description = L10n.Get("msg.browse_models"), UseDescriptionForTitle = true };
        if (Directory.Exists(ActiveModelsDir)) dlg.SelectedPath = ActiveModelsDir;
        if (dlg.ShowDialog() == Forms.DialogResult.OK)
        {
            _settings.ModelsDir = dlg.SelectedPath;
            SaveSettings();
            RefreshWorkspaceState();
        }
    }

    private void OpenModelsDir_Click(object sender, RoutedEventArgs e)
    {
        if (!EngineManager.IsValidRuntimeRoot(_settings.RuntimeRootDir)) return;
        Directory.CreateDirectory(ActiveModelsDir);
        Process.Start(new ProcessStartInfo("explorer.exe", $"\"{ActiveModelsDir}\"") { UseShellExecute = true });
    }

    private void BrowseRefsDir_Click(object sender, RoutedEventArgs e)
    {
        using var dlg = new Forms.FolderBrowserDialog { Description = L10n.Get("msg.browse_refs"), UseDescriptionForTitle = true };
        if (Directory.Exists(ActiveRefsDir)) dlg.SelectedPath = ActiveRefsDir;
        if (dlg.ShowDialog() == Forms.DialogResult.OK)
        {
            _settings.RefsDir = dlg.SelectedPath;
            SaveSettings();
            RefreshWorkspaceState();
        }
    }

    private void OpenRefsDir_Click(object sender, RoutedEventArgs e)
    {
        if (!EngineManager.IsValidRuntimeRoot(_settings.RuntimeRootDir)) return;
        Directory.CreateDirectory(ActiveRefsDir);
        Process.Start(new ProcessStartInfo("explorer.exe", $"\"{ActiveRefsDir}\"") { UseShellExecute = true });
    }

    private void ImportVoice_Click(object sender, RoutedEventArgs e)
    {
        if (!EngineManager.IsValidRuntimeRoot(_settings.RuntimeRootDir))
        {
            System.Windows.MessageBox.Show(L10n.Get("msg.select_root_first"), L10n.Get("msg.notice"), MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        var dlg = new Microsoft.Win32.OpenFileDialog
        {
            Title = L10n.Get("msg.import_title"),
            Filter = "Audio Files (*.wav;*.mp3)|*.wav;*.mp3|All Files|*.*",
            Multiselect = true
        };
        if (dlg.ShowDialog() != true) return;
        foreach (var file in dlg.FileNames)
        {
            try
            {
                var dst = RefsManager.ImportRef(ActiveRefsDir, file);
                AppendLog($"[VOICE] Imported: {Path.GetFileName(dst)}");
            }
            catch (Exception ex)
            {
                System.Windows.MessageBox.Show(ex.Message, L10n.Get("msg.import_failed"), MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        }
        RefreshWorkspaceState();
    }

    private void PlayVoiceRef_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not System.Windows.Controls.Button { Tag: RefVoiceItem item }) return;
        try
        {
            if (item.FilePath.EndsWith(".wav", StringComparison.OrdinalIgnoreCase))
            {
                using var player = new SoundPlayer(item.FilePath);
                player.Play();
            }
            else
            {
                Process.Start(new ProcessStartInfo(item.FilePath) { UseShellExecute = true });
            }
        }
        catch (Exception ex)
        {
            System.Windows.MessageBox.Show(ex.Message, L10n.Get("msg.play_failed"), MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void OpenExternalLink_Click(object sender, RoutedEventArgs e)
    {
        if (sender is System.Windows.Controls.Button { Tag: string url })
            Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
    }

    private void Hyperlink_RequestNavigate(object sender, System.Windows.Navigation.RequestNavigateEventArgs e)
    {
        Process.Start(new ProcessStartInfo(e.Uri.AbsoluteUri) { UseShellExecute = true });
        e.Handled = true;
    }
}
