"""Built-in llama.cpp server manager.

Downloads, launches, and monitors a local llama-server process so users
don't need to install KoboldCpp or any other external backend.

Two modes are supported (configured in settings.json → llm_server.mode):
  "builtin"  — Talon manages the server lifecycle automatically
  "external" — user runs their own server (KoboldCpp, llama.cpp, etc.)
"""

import os
import sys
import json
import time
import shutil
import zipfile
import threading
import subprocess
from pathlib import Path

import requests


class LLMServerManager:
    """Manages the lifecycle of a local llama-server process.

    Responsibilities:
      • Auto-download llama.cpp CUDA release from GitHub
      • Start / stop the server subprocess
      • Poll /health until the server is ready
      • Provide status for the GUI (via callbacks)
    """

    # GitHub release API for llama.cpp
    _RELEASES_URL = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"

    # Asset name patterns to match in the release
    # We want the CUDA 12 Windows build (includes cublas DLLs)
    _ASSET_PATTERNS = [
        "cudart-llama",    # CUDA runtime DLLs
        "llama-",          # main binary package
    ]

    def __init__(self, config: dict):
        """
        Args:
            config: The ``llm_server`` section of settings.json.
        """
        self.mode = config.get("mode", "external")
        self.model_path = config.get("model_path", "")
        self.port = config.get("port", 8080)
        self.n_gpu_layers = config.get("n_gpu_layers", -1)
        self.ctx_size = config.get("ctx_size", 4096)
        self.threads = config.get("threads", 4)
        self.bin_path = config.get("bin_path", "bin/")
        self.extra_args = config.get("extra_args", "")

        self._process: subprocess.Popen | None = None
        self._status = "stopped"  # stopped | starting | running | error
        self._status_lock = threading.Lock()
        self._health_thread: threading.Thread | None = None

        # Callbacks (set by GUI layer)
        self.on_status_changed = None   # fn(status_str)
        self.on_ready = None            # fn()
        self.on_error = None            # fn(error_str)

    # ── Status ────────────────────────────────────────────────

    @property
    def status(self) -> str:
        with self._status_lock:
            return self._status

    @status.setter
    def status(self, value: str):
        with self._status_lock:
            self._status = value
        if self.on_status_changed:
            try:
                self.on_status_changed(value)
            except Exception:
                pass

    # ── Download ──────────────────────────────────────────────

    @property
    def server_exe_path(self) -> Path:
        return Path(self.bin_path) / "llama-server.exe"

    def needs_download(self) -> bool:
        """True if llama-server.exe is not found in bin_path."""
        return not self.server_exe_path.is_file()

    def download_server(self, progress_cb=None, status_cb=None):
        """Download the latest llama.cpp CUDA release from GitHub.

        Args:
            progress_cb: fn(bytes_downloaded, total_bytes) called during download
            status_cb:   fn(message_str) for status updates

        Raises:
            RuntimeError: On download or extraction failure.
        """
        bin_dir = Path(self.bin_path)
        bin_dir.mkdir(parents=True, exist_ok=True)

        if status_cb:
            status_cb("Fetching release info from GitHub...")

        # 1. Get latest release metadata
        try:
            resp = requests.get(self._RELEASES_URL, timeout=30,
                                headers={"Accept": "application/vnd.github.v3+json"})
            resp.raise_for_status()
            release = resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch release info: {e}")

        # 2. Find the right Windows CUDA asset
        assets = release.get("assets", [])
        download_url = None
        asset_name = None

        for asset in assets:
            name = asset.get("name", "").lower()
            # Look for the Windows CUDA 12 build ZIP
            if ("win" in name and "cuda" in name and "cu12" in name
                    and name.endswith(".zip")
                    and "cudart" not in name):
                download_url = asset.get("browser_download_url")
                asset_name = asset.get("name")
                break

        if not download_url:
            # Fallback: try any Windows build with CUDA
            for asset in assets:
                name = asset.get("name", "").lower()
                if "win" in name and "cuda" in name and name.endswith(".zip"):
                    download_url = asset.get("browser_download_url")
                    asset_name = asset.get("name")
                    break

        if not download_url:
            raise RuntimeError(
                "Could not find a Windows CUDA build in the latest release. "
                f"Release: {release.get('tag_name', 'unknown')}")

        if status_cb:
            status_cb(f"Downloading {asset_name}...")

        # 3. Download the ZIP
        zip_path = bin_dir / asset_name
        try:
            with requests.get(download_url, stream=True, timeout=600) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_cb:
                            progress_cb(downloaded, total)
        except Exception as e:
            zip_path.unlink(missing_ok=True)
            raise RuntimeError(f"Download failed: {e}")

        if status_cb:
            status_cb("Extracting...")

        # 4. Extract and find llama-server.exe
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(bin_dir)
        except Exception as e:
            zip_path.unlink(missing_ok=True)
            raise RuntimeError(f"Extraction failed: {e}")

        # The ZIP usually contains a single top-level folder.
        # Walk extracted contents to find llama-server.exe (or a renamed
        # variant) and move everything up to bin_dir if needed.
        #
        # Recent llama.cpp releases may restructure the folder layout or
        # rename the binary, so we search for multiple candidate names.
        _SERVER_NAMES = {"llama-server.exe", "llama-server",
                         "server.exe", "llama-cli.exe"}

        server_exe = None
        all_extracted_files = []
        for root, dirs, files in os.walk(bin_dir):
            for fname in files:
                fpath = Path(root) / fname
                # Skip the ZIP itself
                if fpath == zip_path:
                    continue
                all_extracted_files.append(str(fpath))
                if fname.lower() in _SERVER_NAMES and not server_exe:
                    server_exe = fpath

        print(f"   [LLMServer] Extracted {len(all_extracted_files)} files")
        # Log a handful of files for diagnostic purposes
        for fp in all_extracted_files[:15]:
            print(f"   [LLMServer]   {fp}")
        if len(all_extracted_files) > 15:
            print(f"   [LLMServer]   ... and {len(all_extracted_files) - 15} more")

        if server_exe and server_exe.parent != bin_dir:
            # Move all files from the nested folder to bin_dir
            nested = server_exe.parent
            for item in nested.iterdir():
                dest = bin_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
            # Clean up empty nested folder(s)
            try:
                nested.rmdir()
            except OSError:
                pass

        # If the binary was found but has a different name, create an alias
        if server_exe and server_exe.name.lower() != "llama-server.exe":
            renamed = bin_dir / "llama-server.exe"
            if not renamed.exists():
                actual = bin_dir / server_exe.name
                if actual.exists():
                    shutil.copy2(str(actual), str(renamed))
                    print(f"   [LLMServer] Renamed {server_exe.name} -> llama-server.exe")

        # Also try to download CUDA runtime if available
        self._download_cudart(assets, bin_dir, progress_cb, status_cb)

        # Clean up ZIP only after confirming extraction succeeded
        if self.server_exe_path.is_file():
            zip_path.unlink(missing_ok=True)
        else:
            # Leave the ZIP intact so the user can inspect it
            exe_list = [f for f in all_extracted_files
                        if f.lower().endswith(".exe")]
            raise RuntimeError(
                f"llama-server.exe not found after extraction in {bin_dir}.\n"
                f"Extracted .exe files: {exe_list or 'none'}\n"
                f"The downloaded ZIP has been kept at {zip_path} for inspection."
            )

        if status_cb:
            status_cb("Download complete!")

    def _download_cudart(self, assets, bin_dir, progress_cb, status_cb):
        """Download CUDA runtime DLLs if they're in a separate asset."""
        cudart_url = None
        cudart_name = None
        for asset in assets:
            name = asset.get("name", "").lower()
            if "cudart" in name and "win" in name and name.endswith(".zip"):
                cudart_url = asset.get("browser_download_url")
                cudart_name = asset.get("name")
                break

        if not cudart_url:
            return  # Not available as separate asset — may be bundled

        # Check if we already have CUDA DLLs
        has_cuda = any(f.suffix == ".dll" and "cuda" in f.name.lower()
                       for f in bin_dir.iterdir() if f.is_file())
        if has_cuda:
            return  # Already present

        if status_cb:
            status_cb(f"Downloading CUDA runtime ({cudart_name})...")

        zip_path = bin_dir / cudart_name
        try:
            with requests.get(cudart_url, stream=True, timeout=300) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_cb:
                            progress_cb(downloaded, total)

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(bin_dir)

            # Flatten nested folder if present
            for root, dirs, files in os.walk(bin_dir):
                for fname in files:
                    fpath = Path(root) / fname
                    if fpath.parent != bin_dir and fpath.suffix == ".dll":
                        dest = bin_dir / fname
                        if not dest.exists():
                            shutil.move(str(fpath), str(dest))
        except Exception as e:
            print(f"   [LLMServer] CUDA runtime download failed (non-fatal): {e}")
        finally:
            zip_path.unlink(missing_ok=True)

    # ── Server Lifecycle ──────────────────────────────────────

    def start(self):
        """Start the llama-server subprocess and poll until healthy.

        Sets status to "starting" immediately, then "running" when /health
        responds OK, or "error" on failure.
        """
        if self.is_running():
            return

        if not self.server_exe_path.is_file():
            self.status = "error"
            if self.on_error:
                self.on_error("llama-server.exe not found. Download it first.")
            return

        if not self.model_path or not Path(self.model_path).is_file():
            self.status = "error"
            if self.on_error:
                self.on_error("No model file configured or file not found.")
            return

        self.status = "starting"

        # Build command line
        cmd = [
            str(self.server_exe_path),
            "--model", self.model_path,
            "--port", str(self.port),
            "-ngl", str(self.n_gpu_layers),
            "-c", str(self.ctx_size),
            "-t", str(self.threads),
        ]

        # Extra args (user-provided)
        if self.extra_args:
            cmd.extend(self.extra_args.split())

        print(f"   [LLMServer] Starting: {' '.join(cmd)}")

        # Add bin_path to DLL search path for CUDA DLLs
        env = os.environ.copy()
        bin_abs = str(Path(self.bin_path).resolve())
        if sys.platform == "win32":
            env["PATH"] = bin_abs + os.pathsep + env.get("PATH", "")

        try:
            creation_flags = 0
            if sys.platform == "win32":
                creation_flags = subprocess.CREATE_NO_WINDOW

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                creationflags=creation_flags,
            )
        except Exception as e:
            self.status = "error"
            if self.on_error:
                self.on_error(f"Failed to start server: {e}")
            return

        # Poll /health in background thread
        self._health_thread = threading.Thread(
            target=self._poll_health, daemon=True)
        self._health_thread.start()

    def stop(self):
        """Stop the llama-server subprocess."""
        if self._process is None:
            self.status = "stopped"
            return

        print("   [LLMServer] Stopping server...")
        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        except Exception as e:
            print(f"   [LLMServer] Error stopping: {e}")
        finally:
            self._process = None
            self.status = "stopped"

    def is_running(self) -> bool:
        """Check if the server process is alive."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def get_endpoint(self) -> str:
        """Return the base URL for the running server."""
        return f"http://localhost:{self.port}"

    # ── Health Check ──────────────────────────────────────────

    def _poll_health(self):
        """Poll /health endpoint until the server is ready (max 120s)."""
        url = f"http://localhost:{self.port}/health"
        deadline = time.time() + 120  # 2 minute timeout for large models

        while time.time() < deadline:
            # Check if process died
            if self._process is None or self._process.poll() is not None:
                stderr_out = ""
                if self._process and self._process.stderr:
                    try:
                        stderr_out = self._process.stderr.read().decode(
                            errors="replace")[-500:]
                    except Exception:
                        pass
                self.status = "error"
                if self.on_error:
                    self.on_error(
                        f"Server process exited unexpectedly.\n{stderr_out}")
                return

            try:
                resp = requests.get(url, timeout=2)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "ok":
                        print("   [LLMServer] Server is healthy!")
                        self.status = "running"
                        if self.on_ready:
                            self.on_ready()
                        return
            except (requests.ConnectionError, requests.Timeout):
                pass  # Server not ready yet
            except Exception:
                pass

            time.sleep(1)

        # Timed out
        self.status = "error"
        if self.on_error:
            self.on_error("Server failed to become healthy within 120 seconds.")

    # ── Configuration ─────────────────────────────────────────

    def update_config(self, config: dict):
        """Update server configuration from settings dict."""
        self.mode = config.get("mode", self.mode)
        self.model_path = config.get("model_path", self.model_path)
        self.port = config.get("port", self.port)
        self.n_gpu_layers = config.get("n_gpu_layers", self.n_gpu_layers)
        self.ctx_size = config.get("ctx_size", self.ctx_size)
        self.threads = config.get("threads", self.threads)
        self.bin_path = config.get("bin_path", self.bin_path)
        self.extra_args = config.get("extra_args", self.extra_args)

    def to_dict(self) -> dict:
        """Serialize current config for saving to settings.json."""
        return {
            "mode": self.mode,
            "model_path": self.model_path,
            "port": self.port,
            "n_gpu_layers": self.n_gpu_layers,
            "ctx_size": self.ctx_size,
            "threads": self.threads,
            "bin_path": self.bin_path,
            "extra_args": self.extra_args,
        }
