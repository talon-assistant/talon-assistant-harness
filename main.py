import sys
import os
import json
import signal
import traceback

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows DLL fix
if sys.platform == 'win32':
    os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)))


def _install_exception_hook():
    """Install a global exception hook so unhandled exceptions in Qt slots
    produce a traceback instead of a silent crash / segfault."""
    import faulthandler
    faulthandler.enable()

    _original_hook = sys.excepthook

    def _hook(exc_type, exc_value, exc_tb):
        sys.stderr.write("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        sys.stderr.flush()
        _original_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook


def _load_settings(config_dir="config"):
    """Load settings.json and return the full dict."""
    config_path = os.path.join(config_dir, "settings.json")
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _setup_builtin_server(settings, config_dir="config"):
    """If llm_server mode is 'builtin', create and start the server manager.

    This runs BEFORE QApplication so the server can be starting while
    models load.  Returns the LLMServerManager (or None for external mode).
    """
    server_config = settings.get("llm_server", {})
    mode = server_config.get("mode", "external")

    if mode != "builtin":
        return None

    from core.llm_server import LLMServerManager

    manager = LLMServerManager(server_config)

    # Add bin/ to DLL search path for CUDA DLLs
    bin_abs = os.path.abspath(server_config.get("bin_path", "bin/"))
    if sys.platform == "win32" and os.path.isdir(bin_abs):
        try:
            os.add_dll_directory(bin_abs)
        except OSError:
            pass

    if manager.needs_download():
        print("\n   [LLMServer] llama-server.exe not found.")
        print("   Use File > LLM Server... to download it.\n")
        return manager

    model_path = server_config.get("model_path", "")
    if not model_path or not os.path.isfile(model_path):
        print("\n   [LLMServer] No model file configured or file not found.")
        print("   Use File > LLM Server... to configure it.\n")
        return manager

    # Override LLM settings for builtin mode
    port = server_config.get("port", 8080)
    settings.setdefault("llm", {})
    settings["llm"]["endpoint"] = f"http://localhost:{port}/completion"
    settings["llm"]["api_format"] = "llamacpp"

    # Start the server (non-blocking — health poll runs in background)
    print("\n   [LLMServer] Starting built-in server...")
    manager.start()

    return manager


def main():
    mode = "gui"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    if mode == "gui":
        _install_exception_hook()

        # ── Step 0: Load settings and optionally start built-in LLM server ──
        settings = _load_settings("config")
        server_manager = _setup_builtin_server(settings, "config")

        # ── Step 1: Build TalonAssistant BEFORE QApplication ──────────
        # CTranslate2 (used by faster-whisper) segfaults when WhisperModel
        # is created after QApplication has initialised its platform plugins.
        # Loading all models first avoids the conflict entirely.
        print("Loading Talon (models + services)...")
        print("  This may take 10-30 seconds on first launch.\n")
        assistant = None
        init_error = None
        try:
            from core.assistant import TalonAssistant
            assistant = TalonAssistant(config_dir="config")
        except Exception as e:
            init_error = str(e)
            traceback.print_exc()

        # ── Step 2: Now it's safe to create Qt ─────────────────────────
        from PyQt6.QtWidgets import QApplication
        from gui.main_window import MainWindow
        from gui.assistant_bridge import AssistantBridge
        from gui.output_interceptor import OutputInterceptor

        app = QApplication(sys.argv)

        # Allow Ctrl+C to kill the app from the terminal
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        # Theme manager (replaces inline QSS loading)
        from gui.theme_manager import ThemeManager
        theme_manager = ThemeManager(app, config_dir="config")

        # Output interceptor
        interceptor = OutputInterceptor(sys.stdout)
        sys.stdout = interceptor

        # Create window
        bridge = AssistantBridge(config_dir="config")

        # Hand the server manager to the bridge and assistant (if builtin mode)
        if server_manager is not None:
            bridge.set_server_manager(server_manager)
            if assistant is not None:
                assistant.server_manager = server_manager

        window = MainWindow(bridge, theme_manager=theme_manager,
                            config_dir="config")

        # System tray
        from gui.system_tray import SystemTrayManager
        tray_manager = SystemTrayManager(window)
        window.system_tray = tray_manager
        tray_manager.exit_requested.connect(window._force_exit)

        interceptor.text_written.connect(window.activity_log.append_raw)

        window.show()

        # Hand the pre-built assistant to the bridge (or show error)
        if assistant is not None:
            bridge.set_assistant(assistant)
        else:
            window.chat_view.add_error_message(
                f"Initialization failed: {init_error}"
            )
            window.text_input.set_enabled(False)

        exit_code = app.exec()

        # Clean up stdout interceptor
        sys.stdout = interceptor._original or sys.__stdout__

        # Stop built-in LLM server before exit
        if server_manager and server_manager.is_running():
            server_manager.stop()

        # Force-kill the process to ensure all threads (pynput, sounddevice, etc.) die
        # QThread cleanup already attempted in closeEvent -> bridge.cleanup()
        os._exit(exit_code)

    elif mode in ("voice", "text", "both"):
        from core.assistant import TalonAssistant
        talon = TalonAssistant(config_dir="config")
        talon.run(mode=mode)

    else:
        print(f"Unknown mode: {mode}. Use gui, voice, text, or both.")
        sys.exit(1)


if __name__ == "__main__":
    main()
