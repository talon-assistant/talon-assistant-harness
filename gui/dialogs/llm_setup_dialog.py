"""LLM Server configuration dialog.

Two-tab dialog for configuring either a built-in llama.cpp server
or an external LLM endpoint (KoboldCpp, llama.cpp, OpenAI-compatible).
"""

import os
import json
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QFormLayout, QLineEdit, QSpinBox, QComboBox, QLabel,
    QPushButton, QFileDialog, QProgressBar, QGroupBox,
    QSlider, QSizePolicy
)
from PyQt6.QtCore import pyqtSignal, Qt


class LLMSetupDialog(QDialog):
    """Tabbed dialog for configuring the LLM server backend.

    Tab 1 — Built-in Server:
        Model file picker, GPU layers, context size, threads, port,
        download button + progress, start/stop server, status indicator.

    Tab 2 — External Server:
        Endpoint URL, API format selector, test connection button, status.

    Emits ``settings_saved`` with the full combined config dict when the
    user clicks Save.
    """

    settings_saved = pyqtSignal(dict)  # {"llm": {...}, "llm_server": {...}}

    def __init__(self, llm_config, server_config, server_manager=None,
                 config_path="config/settings.json", parent=None):
        super().__init__(parent)
        self.setWindowTitle("LLM Server Configuration")
        self.setMinimumWidth(550)
        self.resize(600, 520)

        self._llm_config = dict(llm_config)
        self._server_config = dict(server_config)
        self._server_manager = server_manager
        self._config_path = config_path
        self._download_worker = None
        self._start_worker = None

        layout = QVBoxLayout(self)

        # Header
        header = QLabel("LLM Server Configuration")
        header.setObjectName("llm_setup_header")
        layout.addWidget(header)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_builtin_tab(), "Built-in Server")
        self.tabs.addTab(self._build_external_tab(), "External Server")
        layout.addWidget(self.tabs)

        # Select the right tab based on current mode
        if self._server_config.get("mode") == "builtin":
            self.tabs.setCurrentIndex(0)
        else:
            self.tabs.setCurrentIndex(1)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        save_btn = QPushButton("Save")
        save_btn.setObjectName("talent_config_save_btn")
        save_btn.clicked.connect(self._on_save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        # Update UI state
        self._update_server_status()

    # ── Built-in Tab ──────────────────────────────────────────

    def _build_builtin_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        # Model configuration
        model_group = QGroupBox("Model")
        model_form = QFormLayout(model_group)

        # Model file picker
        model_row = QHBoxLayout()
        self._model_path_edit = QLineEdit(
            self._server_config.get("model_path", ""))
        self._model_path_edit.setObjectName("llm_model_path")
        self._model_path_edit.setPlaceholderText("Path to .gguf model file")
        model_row.addWidget(self._model_path_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_model)
        model_row.addWidget(browse_btn)
        model_form.addRow("Model File:", model_row)

        # mmproj file picker (optional — required for vision/multimodal)
        mmproj_row = QHBoxLayout()
        self._mmproj_path_edit = QLineEdit(
            self._server_config.get("mmproj_path", ""))
        self._mmproj_path_edit.setPlaceholderText(
            "Path to mmproj file (optional, required for vision)")
        mmproj_row.addWidget(self._mmproj_path_edit)
        mmproj_browse_btn = QPushButton("Browse...")
        mmproj_browse_btn.clicked.connect(self._browse_mmproj)
        mmproj_row.addWidget(mmproj_browse_btn)
        model_form.addRow("mmproj File:", mmproj_row)

        # LoRA adapter file picker (optional fine-tune)
        lora_row = QHBoxLayout()
        self._lora_path_edit = QLineEdit(self._server_config.get("lora_path", ""))
        self._lora_path_edit.setPlaceholderText("Path to LoRA adapter .gguf (optional)")
        lora_row.addWidget(self._lora_path_edit)
        lora_browse_btn = QPushButton("Browse...")
        lora_browse_btn.clicked.connect(self._browse_lora)
        lora_row.addWidget(lora_browse_btn)
        model_form.addRow("LoRA Adapter:", lora_row)

        layout.addWidget(model_group)

        # Server settings
        settings_group = QGroupBox("Server Settings")
        settings_form = QFormLayout(settings_group)

        # GPU layers
        gpu_row = QHBoxLayout()
        self._gpu_slider = QSlider(Qt.Orientation.Horizontal)
        self._gpu_slider.setRange(-1, 99)
        self._gpu_slider.setValue(
            self._server_config.get("n_gpu_layers", -1))
        self._gpu_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._gpu_slider.setTickInterval(10)
        self._gpu_label = QLabel(
            str(self._server_config.get("n_gpu_layers", -1)))
        self._gpu_label.setMinimumWidth(30)
        self._gpu_slider.valueChanged.connect(
            lambda v: self._gpu_label.setText(
                "All" if v == -1 else str(v)))
        gpu_row.addWidget(self._gpu_slider)
        gpu_row.addWidget(self._gpu_label)
        settings_form.addRow("GPU Layers:", gpu_row)

        # Context size
        self._ctx_combo = QComboBox()
        ctx_options = ["512", "1024", "2048", "4096", "8192",
                       "16384", "32768", "65536", "131072"]
        self._ctx_combo.addItems(ctx_options)
        current_ctx = str(self._server_config.get("ctx_size", 4096))
        idx = self._ctx_combo.findText(current_ctx)
        if idx >= 0:
            self._ctx_combo.setCurrentIndex(idx)
        settings_form.addRow("Context Size:", self._ctx_combo)

        # CPU threads
        self._threads_spin = QSpinBox()
        self._threads_spin.setRange(1, 64)
        self._threads_spin.setValue(
            self._server_config.get("threads", 4))
        settings_form.addRow("CPU Threads:", self._threads_spin)

        # Port
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1024, 65535)
        self._port_spin.setValue(
            self._server_config.get("port", 8080))
        settings_form.addRow("Port:", self._port_spin)

        # Extra args
        self._extra_args_edit = QLineEdit(
            self._server_config.get("extra_args", ""))
        self._extra_args_edit.setPlaceholderText(
            "Additional CLI arguments (e.g., --flash-attn)")
        settings_form.addRow("Extra Args:", self._extra_args_edit)

        layout.addWidget(settings_group)

        # Download & Server Control
        control_group = QGroupBox("Server Control")
        control_layout = QVBoxLayout(control_group)

        # Download row
        dl_row = QHBoxLayout()
        self._download_btn = QPushButton("Download llama.cpp")
        self._download_btn.setObjectName("llm_download_btn")
        self._download_btn.clicked.connect(self._start_download)
        dl_row.addWidget(self._download_btn)
        self._download_progress = QProgressBar()
        self._download_progress.setVisible(False)
        dl_row.addWidget(self._download_progress)
        control_layout.addLayout(dl_row)

        self._download_status = QLabel("")
        self._download_status.setObjectName("marketplace_status")
        control_layout.addWidget(self._download_status)

        # Start/Stop row
        server_row = QHBoxLayout()
        self._start_btn = QPushButton("Start Server")
        self._start_btn.setObjectName("llm_start_btn")
        self._start_btn.clicked.connect(self._start_server)
        server_row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop Server")
        self._stop_btn.setObjectName("llm_stop_btn")
        self._stop_btn.clicked.connect(self._stop_server)
        server_row.addWidget(self._stop_btn)
        server_row.addStretch()

        # Status indicator
        self._builtin_status = QLabel("Status: Stopped")
        self._builtin_status.setObjectName("llm_status_dot_stopped")
        server_row.addWidget(self._builtin_status)

        control_layout.addLayout(server_row)
        layout.addWidget(control_group)

        layout.addStretch()
        return w

    # ── External Tab ──────────────────────────────────────────

    def _build_external_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        settings_group = QGroupBox("External LLM Server")
        form = QFormLayout(settings_group)

        # Endpoint URL
        self._endpoint_edit = QLineEdit(
            self._llm_config.get("endpoint", ""))
        self._endpoint_edit.setPlaceholderText(
            "http://localhost:5001/api/v1/generate")
        form.addRow("Endpoint URL:", self._endpoint_edit)

        # API format
        self._format_combo = QComboBox()
        formats = ["koboldcpp", "llamacpp", "openai"]
        self._format_combo.addItems(formats)
        current_format = self._llm_config.get("api_format", "koboldcpp")
        idx = self._format_combo.findText(current_format)
        if idx >= 0:
            self._format_combo.setCurrentIndex(idx)
        form.addRow("API Format:", self._format_combo)

        layout.addWidget(settings_group)

        # Test connection
        test_row = QHBoxLayout()
        self._test_btn = QPushButton("Test Connection")
        self._test_btn.clicked.connect(self._test_connection)
        test_row.addWidget(self._test_btn)

        self._external_status = QLabel("Status: Not tested")
        self._external_status.setObjectName("llm_status_dot_stopped")
        test_row.addWidget(self._external_status)
        test_row.addStretch()

        layout.addLayout(test_row)

        # Hint
        hint = QLabel(
            "Tip: For KoboldCpp, the endpoint is usually "
            "http://localhost:5001/api/v1/generate\n"
            "For llama.cpp server, use http://localhost:8080/completion\n"
            "For OpenAI-compatible, use http://localhost:8080/v1/chat/completions"
        )
        hint.setObjectName("talent_description")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        layout.addStretch()
        return w

    # ── Actions ───────────────────────────────────────────────

    def _browse_model(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model",
            os.path.expanduser("~"),
            "GGUF Models (*.gguf);;All Files (*)")
        if filepath:
            self._model_path_edit.setText(filepath)

    def _browse_mmproj(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select mmproj File",
            os.path.expanduser("~"),
            "mmproj Files (*.gguf *.bin);;All Files (*)")
        if filepath:
            self._mmproj_path_edit.setText(filepath)

    def _browse_lora(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select LoRA Adapter File",
            os.path.expanduser("~"),
            "LoRA Files (*.gguf *.bin);;All Files (*)")
        if filepath:
            self._lora_path_edit.setText(filepath)

    def _start_download(self):
        if self._server_manager is None:
            self._download_status.setText("Error: No server manager available.")
            return

        self._download_btn.setEnabled(False)
        self._download_progress.setVisible(True)
        self._download_progress.setValue(0)
        self._download_status.setText("Starting download...")

        from gui.workers import DownloadWorker
        self._download_worker = DownloadWorker(self._server_manager)
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.status.connect(self._download_status.setText)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.error.connect(self._on_download_error)
        self._download_worker.start()

    def _on_download_progress(self, downloaded, total):
        if total > 0:
            pct = int(downloaded / total * 100)
            self._download_progress.setValue(pct)
            mb_dl = downloaded / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            self._download_status.setText(
                f"Downloading: {mb_dl:.1f} / {mb_total:.1f} MB ({pct}%)")
        else:
            mb_dl = downloaded / (1024 * 1024)
            self._download_status.setText(
                f"Downloading: {mb_dl:.1f} MB...")

    def _on_download_finished(self, exe_path):
        self._download_btn.setEnabled(True)
        self._download_progress.setVisible(False)
        self._download_status.setText(f"Download complete! Binary: {exe_path}")
        self._update_server_status()

    def _on_download_error(self, error_msg):
        self._download_btn.setEnabled(True)
        self._download_progress.setVisible(False)
        self._download_status.setText(f"Download failed: {error_msg}")

    def _start_server(self):
        if self._server_manager is None:
            self._builtin_status.setText("Status: No server manager")
            return

        # Update manager config from current UI values before starting
        self._server_manager.model_path = self._model_path_edit.text()
        self._server_manager.port = self._port_spin.value()
        self._server_manager.n_gpu_layers = self._gpu_slider.value()
        self._server_manager.ctx_size = int(self._ctx_combo.currentText())
        self._server_manager.threads = self._threads_spin.value()
        self._server_manager.extra_args = self._extra_args_edit.text()
        self._server_manager.mmproj_path = self._mmproj_path_edit.text()
        self._server_manager.lora_path = self._lora_path_edit.text()

        self._start_btn.setEnabled(False)
        self._builtin_status.setText("Status: Starting...")
        self._builtin_status.setObjectName("llm_status_dot_starting")
        self._builtin_status.style().unpolish(self._builtin_status)
        self._builtin_status.style().polish(self._builtin_status)

        from gui.workers import ServerStartWorker
        self._start_worker = ServerStartWorker(self._server_manager)
        self._start_worker.ready.connect(self._on_server_ready)
        self._start_worker.error.connect(self._on_server_error)
        self._start_worker.status.connect(
            lambda s: self._builtin_status.setText(f"Status: {s}"))
        self._start_worker.start()

    def _on_server_ready(self):
        self._start_btn.setEnabled(True)
        self._update_server_status()

    def _on_server_error(self, error_msg):
        self._start_btn.setEnabled(True)
        self._builtin_status.setText(f"Status: Error - {error_msg}")
        self._builtin_status.setObjectName("llm_status_dot_error")
        self._builtin_status.style().unpolish(self._builtin_status)
        self._builtin_status.style().polish(self._builtin_status)

    def _stop_server(self):
        if self._server_manager:
            self._server_manager.stop()
        self._update_server_status()

    def _test_connection(self):
        """Test external server connection in a lightweight way."""
        import requests as req

        endpoint = self._endpoint_edit.text().strip()
        api_format = self._format_combo.currentText()

        if not endpoint:
            self._external_status.setText("Status: No endpoint configured")
            return

        self._external_status.setText("Status: Testing...")
        self._test_btn.setEnabled(False)

        try:
            if api_format == "llamacpp":
                base = endpoint.rsplit("/completion", 1)[0]
                resp = req.get(f"{base}/health", timeout=5)
                ok = (resp.status_code == 200
                      and resp.json().get("status") == "ok")
            elif api_format == "openai":
                base = endpoint.rsplit("/v1/", 1)[0]
                resp = req.get(f"{base}/v1/models", timeout=5)
                ok = resp.status_code == 200
            else:
                # KoboldCpp — try a minimal generate
                resp = req.post(endpoint, json={
                    "prompt": "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
                    "max_length": 1,
                    "stop_sequence": ["<|im_end|>"]
                }, timeout=10)
                ok = resp.status_code == 200

            if ok:
                self._external_status.setText("Status: Connected!")
                self._external_status.setObjectName("llm_status_dot_running")
            else:
                self._external_status.setText(
                    f"Status: Failed (HTTP {resp.status_code})")
                self._external_status.setObjectName("llm_status_dot_error")
        except Exception as e:
            self._external_status.setText(f"Status: Error - {e}")
            self._external_status.setObjectName("llm_status_dot_error")

        self._external_status.style().unpolish(self._external_status)
        self._external_status.style().polish(self._external_status)
        self._test_btn.setEnabled(True)

    def _update_server_status(self):
        """Update the built-in tab status display based on server manager state."""
        if self._server_manager is None:
            self._builtin_status.setText("Status: Not configured")
            self._download_btn.setEnabled(False)
            self._start_btn.setEnabled(False)
            self._stop_btn.setEnabled(False)
            return

        has_binary = not self._server_manager.needs_download()
        is_running = self._server_manager.is_running()

        self._download_btn.setEnabled(not has_binary and not is_running)
        self._start_btn.setEnabled(has_binary and not is_running)
        self._stop_btn.setEnabled(is_running)

        if is_running:
            self._builtin_status.setText("Status: Running")
            self._builtin_status.setObjectName("llm_status_dot_running")
        elif has_binary:
            self._builtin_status.setText("Status: Stopped (binary ready)")
            self._builtin_status.setObjectName("llm_status_dot_stopped")
        else:
            self._builtin_status.setText(
                "Status: No binary (download required)")
            self._builtin_status.setObjectName("llm_status_dot_stopped")

        self._builtin_status.style().unpolish(self._builtin_status)
        self._builtin_status.style().polish(self._builtin_status)

        if not has_binary:
            self._download_status.setText(
                "llama.cpp not found. Click 'Download' to get the latest release.")

    # ── Save ──────────────────────────────────────────────────

    def _on_save(self):
        """Collect values from both tabs and emit combined config."""
        # Determine mode from active tab
        is_builtin = self.tabs.currentIndex() == 0

        # Server config (always saved)
        server_cfg = {
            "mode": "builtin" if is_builtin else "external",
            "model_path": self._model_path_edit.text(),
            "port": self._port_spin.value(),
            "n_gpu_layers": self._gpu_slider.value(),
            "ctx_size": int(self._ctx_combo.currentText()),
            "threads": self._threads_spin.value(),
            "bin_path": self._server_config.get("bin_path", "bin/"),
            "extra_args": self._extra_args_edit.text(),
            "mmproj_path": self._mmproj_path_edit.text(),
            "lora_path": self._lora_path_edit.text(),
        }

        # LLM config — update endpoint and format based on mode
        llm_cfg = dict(self._llm_config)

        if is_builtin:
            port = self._port_spin.value()
            llm_cfg["endpoint"] = f"http://localhost:{port}/completion"
            llm_cfg["api_format"] = "llamacpp"
        else:
            llm_cfg["endpoint"] = self._endpoint_edit.text()
            llm_cfg["api_format"] = self._format_combo.currentText()

        combined = {"llm": llm_cfg, "llm_server": server_cfg}

        # Persist to disk
        try:
            with open(self._config_path, 'r') as f:
                full_settings = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            full_settings = {}

        full_settings["llm"] = llm_cfg
        full_settings["llm_server"] = server_cfg

        with open(self._config_path, 'w') as f:
            json.dump(full_settings, f, indent=2)

        self.settings_saved.emit(combined)
        self.accept()
