"""FileOrganizerTalent — search, sort, and organize files on disk.

Sort downloads by file type, find large files, and list directory contents.
No external dependencies.

Safety features:
  - Dry-run preview with explicit confirmation before any file is moved
  - System path blocklist (Windows + Linux/macOS system dirs)
  - Manifest log written to the target folder before any moves (undo reference)
  - Collision warnings when a destination file already exists

Examples:
    "organize my downloads folder"
    "find large files in documents"
    "list files in C:/Users/you/Desktop"
    "sort downloads by type"
    "find pdf files in my documents"
"""

import json
import os
import re
import shutil
from datetime import datetime

from talents.base import BaseTalent

# Confirmation phrases that commit a pending organize operation
_CONFIRM_PHRASES = [
    "yes", "yes do it", "do it", "go ahead", "confirm", "proceed",
    "yes please", "ok do it", "okay do it", "yep", "yeah do it",
]

# Cancellation phrases that abort a pending organize operation
_CANCEL_PHRASES = [
    "no", "cancel", "never mind", "nevermind", "stop", "abort", "don't do it",
]

# How long (seconds) a pending confirmation stays valid
_PENDING_TTL = 120


class FileOrganizerTalent(BaseTalent):
    name = "file_organizer"
    description = "Search, move, and organize files on disk"
    keywords = [
        "organize", "sort files", "find file", "find files", "large files",
        "list files", "file organizer", "clean up", "sort downloads",
        "find pdf", "find images", "find documents",
    ]
    priority = 42

    _EXCLUSIONS = [
        "remind", "timer", "email", "note", "weather", "hue",
        "light", "search", "news", "todo", "task", "pomodoro",
    ]

    # Paths that must never be touched regardless of what the user says
    _BLOCKED_PATHS = [
        # Windows system dirs (normalised to lower for comparison)
        "c:\\windows",
        "c:\\program files",
        "c:\\program files (x86)",
        "c:\\programdata",
        "c:\\system32",
        "c:\\syswow64",
        # Unix / macOS system dirs
        "/usr", "/etc", "/bin", "/sbin", "/lib", "/lib64",
        "/boot", "/sys", "/proc", "/dev", "/run",
        "/system", "/library", "/private",
    ]

    # Common file type categories
    _TYPE_MAP = {
        "images":      [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg",
                        ".webp", ".ico", ".tiff"],
        "documents":   [".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt",
                        ".xls", ".xlsx", ".csv", ".ppt", ".pptx"],
        "audio":       [".mp3", ".wav", ".flac", ".ogg", ".aac", ".wma", ".m4a"],
        "video":       [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"],
        "archives":    [".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"],
        "code":        [".py", ".js", ".ts", ".html", ".css", ".java",
                        ".cpp", ".c", ".rs", ".go"],
        "executables": [".exe", ".msi", ".bat", ".sh", ".cmd"],
    }

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        # Pending organize confirmation state
        # { "directory": str, "plan": [(src, dst, category), ...], "expires": float }
        self._pending_organize: dict | None = None

    def get_config_schema(self) -> dict:
        return {
            "fields": [
                {"key": "default_directory", "label": "Default Directory",
                 "type": "string", "default": ""},
                {"key": "large_file_mb",     "label": "Large File Threshold (MB)",
                 "type": "int", "default": 100, "min": 1, "max": 10000},
                {"key": "max_results",       "label": "Max Results to Show",
                 "type": "int", "default": 20,  "min": 5, "max": 100},
            ]
        }

    # ── Routing ───────────────────────────────────────────────────

    def can_handle(self, command: str) -> bool:
        cmd = command.lower().strip()

        # If there's a live pending confirmation, intercept yes / no
        if self._pending_organize is not None:
            import time
            if time.time() < self._pending_organize["expires"]:
                if any(p in cmd for p in _CONFIRM_PHRASES):
                    return True
                if any(p in cmd for p in _CANCEL_PHRASES):
                    return True
            else:
                self._pending_organize = None   # expired — clear it

        if any(ex in cmd for ex in self._EXCLUSIONS):
            return False
        return any(kw in cmd for kw in self.keywords)

    def execute(self, command: str, context: dict) -> dict:
        import time

        cmd = command.lower().strip()

        # ── Handle pending confirmation ───────────────────────────
        if self._pending_organize is not None:
            if time.time() >= self._pending_organize["expires"]:
                self._pending_organize = None
                return self._fail("The pending organize operation timed out. Please ask again.")

            if any(p in cmd for p in _CONFIRM_PHRASES):
                plan = self._pending_organize["plan"]
                directory = self._pending_organize["directory"]
                self._pending_organize = None
                return self._execute_organize(directory, plan)

            if any(p in cmd for p in _CANCEL_PHRASES):
                self._pending_organize = None
                return self._ok("Organize cancelled — no files were moved.")

        # ── Normal command dispatch ───────────────────────────────

        directory = self._extract_path(command)

        # Organize / sort by type
        if any(p in cmd for p in ["organize", "sort by type", "sort files",
                                   "sort downloads", "clean up"]):
            if not directory:
                return self._fail("Which folder should I organize? Please include a path.")
            return self._preview_organize(directory)

        # Find large files
        if "large file" in cmd or "big file" in cmd:
            if not directory:
                directory = self._config.get("default_directory", "")
            if not directory:
                return self._fail("Which folder should I search? Please include a path.")
            return self._find_large_files(directory)

        # Find files by extension
        ext_match = re.search(r'find\s+(\w+)\s+files?', cmd)
        if ext_match:
            file_type = ext_match.group(1)
            if not directory:
                directory = self._config.get("default_directory", "")
            if not directory:
                return self._fail("Which folder should I search? Please include a path.")
            return self._find_by_type(directory, file_type)

        # List files
        if any(p in cmd for p in ["list files", "show files", "what files", "what's in"]):
            if not directory:
                return self._fail("Which folder should I list? Please include a path.")
            return self._list_files(directory)

        return self._fail(
            "I can organize folders by type, find large files, find files by type, "
            "or list directory contents. Please be more specific.")

    # ── Safety helpers ────────────────────────────────────────────

    def _is_blocked(self, directory: str) -> bool:
        """Return True if directory resolves to or is inside a protected system path."""
        try:
            resolved = os.path.realpath(os.path.abspath(directory)).lower()
        except Exception:
            return True   # if we can't resolve it, block it
        return any(resolved == p or resolved.startswith(p + os.sep)
                   for p in self._BLOCKED_PATHS)

    # ── Organize — preview (dry-run) ──────────────────────────────

    def _preview_organize(self, directory: str) -> dict:
        import time

        if not os.path.isdir(directory):
            return self._fail(f"Directory not found: {directory}")

        if self._is_blocked(directory):
            return self._fail(
                f"⛔ I won't organize {directory} — it looks like a protected system folder.")

        plan = []   # list of (src_path, dst_path, category)
        skips = []  # files that would be skipped due to collision

        for fname in os.listdir(directory):
            fpath = os.path.join(directory, fname)
            if not os.path.isfile(fpath):
                continue
            # Never touch the manifest log
            if fname == "_organizer_manifest.json":
                continue
            ext = os.path.splitext(fname)[1].lower()
            category = "other"
            for cat, exts in self._TYPE_MAP.items():
                if ext in exts:
                    category = cat
                    break
            dst_path = os.path.join(directory, category, fname)
            if os.path.exists(dst_path):
                skips.append((fname, category))
            else:
                plan.append((fpath, dst_path, category))

        if not plan and not skips:
            return self._ok("No files to organize in that folder.")

        # Build summary grouped by category
        summary: dict[str, int] = {}
        for _, _, cat in plan:
            summary[cat] = summary.get(cat, 0) + 1

        lines = [f"📋 Preview — {os.path.basename(directory)} ({len(plan)} file(s) to move):\n"]
        for cat, count in sorted(summary.items()):
            lines.append(f"  {cat}/  ← {count} file(s)")
        if skips:
            lines.append(f"\n  ⚠ {len(skips)} file(s) skipped (destination already exists):")
            for fname, cat in skips[:5]:
                lines.append(f"    • {fname} → {cat}/")
            if len(skips) > 5:
                lines.append(f"    ... and {len(skips) - 5} more")
        lines.append(
            "\nA manifest log will be saved to the folder before any files are moved."
            "\nSay **'yes, do it'** to proceed or **'cancel'** to abort."
        )

        self._pending_organize = {
            "directory": directory,
            "plan":      plan,
            "skips":     skips,
            "expires":   time.time() + _PENDING_TTL,
        }
        return self._ok("\n".join(lines))

    # ── Organize — execute (after confirmation) ───────────────────

    def _execute_organize(self, directory: str, plan: list) -> dict:
        # Write manifest before touching anything
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "directory": directory,
            "moves": [{"from": src, "to": dst, "category": cat}
                      for src, dst, cat in plan],
        }
        manifest_path = os.path.join(directory, "_organizer_manifest.json")
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            print(f"   [FileOrganizer] Manifest written: {manifest_path}")
        except Exception as e:
            return self._fail(f"Could not write manifest log ({e}) — aborting to be safe.")

        moved: dict[str, int] = {}
        errors = 0
        late_skips = 0  # collisions that appeared between preview and execution

        for src, dst, cat in plan:
            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.exists(dst):
                    # Collision appeared since preview — warn, don't overwrite
                    late_skips += 1
                    continue
                shutil.move(src, dst)
                moved[cat] = moved.get(cat, 0) + 1
            except Exception:
                errors += 1

        if not moved:
            return self._ok("No files were moved (all destinations already existed).")

        lines = [f"✅ Organized {sum(moved.values())} files in {os.path.basename(directory)}:\n"]
        for cat, count in sorted(moved.items()):
            lines.append(f"  {cat}/  ← {count} file(s)")
        if late_skips:
            lines.append(f"\n  ⚠ {late_skips} file(s) skipped (destination appeared since preview)")
        if errors:
            lines.append(f"\n  ✗ {errors} file(s) could not be moved")
        lines.append(f"\n📄 Manifest saved: {manifest_path}")
        return self._ok("\n".join(lines))

    # ── Find large files ──────────────────────────────────────────

    def _find_large_files(self, directory: str) -> dict:
        if not os.path.isdir(directory):
            return self._fail(f"Directory not found: {directory}")

        threshold_mb = self._config.get("large_file_mb", 100)
        threshold     = threshold_mb * 1024 * 1024
        max_results   = self._config.get("max_results", 20)
        large = []

        try:
            for root, dirs, files in os.walk(directory):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    try:
                        size = os.path.getsize(fpath)
                        if size >= threshold:
                            large.append((fpath, size))
                    except OSError:
                        continue
        except PermissionError:
            return self._fail(f"Permission denied accessing {directory}")

        if not large:
            return self._ok(f"No files larger than {threshold_mb} MB found in {directory}.")

        large.sort(key=lambda x: x[1], reverse=True)
        lines = [f"Files larger than {threshold_mb} MB in {os.path.basename(directory)}:\n"]
        for fpath, size in large[:max_results]:
            size_mb = size / (1024 * 1024)
            rel = os.path.relpath(fpath, directory)
            lines.append(f"  {size_mb:,.1f} MB — {rel}")
        if len(large) > max_results:
            lines.append(f"\n  ...and {len(large) - max_results} more")
        return self._ok("\n".join(lines))

    # ── Find by type ──────────────────────────────────────────────

    def _find_by_type(self, directory: str, file_type: str) -> dict:
        if not os.path.isdir(directory):
            return self._fail(f"Directory not found: {directory}")

        max_results   = self._config.get("max_results", 20)
        extensions    = set()
        file_type_lower = file_type.lower()

        if file_type_lower in self._TYPE_MAP:
            extensions = set(self._TYPE_MAP[file_type_lower])
        else:
            aliases = {
                "pdf":        [".pdf"],
                "image":      self._TYPE_MAP["images"],
                "photo":      self._TYPE_MAP["images"],
                "picture":    self._TYPE_MAP["images"],
                "video":      self._TYPE_MAP["video"],
                "music":      self._TYPE_MAP["audio"],
                "audio":      self._TYPE_MAP["audio"],
                "zip":        [".zip", ".rar", ".7z"],
                "python":     [".py"],
                "javascript": [".js"],
            }
            extensions = set(aliases.get(file_type_lower, [f".{file_type_lower}"]))

        found = []
        try:
            for root, dirs, files in os.walk(directory):
                for fname in files:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in extensions:
                        fpath = os.path.join(root, fname)
                        try:
                            size = os.path.getsize(fpath)
                            found.append((fpath, size))
                        except OSError:
                            found.append((fpath, 0))
        except PermissionError:
            return self._fail(f"Permission denied accessing {directory}")

        if not found:
            return self._ok(f"No {file_type} files found in {directory}.")

        found.sort(key=lambda x: x[1], reverse=True)
        lines = [f"Found {len(found)} {file_type} file(s) in {os.path.basename(directory)}:\n"]
        for fpath, size in found[:max_results]:
            size_str = (f"{size / 1024:.0f} KB" if size < 1024 * 1024
                        else f"{size / (1024 * 1024):.1f} MB")
            rel = os.path.relpath(fpath, directory)
            lines.append(f"  {rel} ({size_str})")
        if len(found) > max_results:
            lines.append(f"\n  ...and {len(found) - max_results} more")
        return self._ok("\n".join(lines))

    # ── List files ────────────────────────────────────────────────

    def _list_files(self, directory: str) -> dict:
        if not os.path.isdir(directory):
            return self._fail(f"Directory not found: {directory}")

        max_results = self._config.get("max_results", 20)
        entries = []
        try:
            for name in os.listdir(directory):
                fpath  = os.path.join(directory, name)
                is_dir = os.path.isdir(fpath)
                try:
                    size  = os.path.getsize(fpath) if not is_dir else 0
                    mtime = os.path.getmtime(fpath)
                except OSError:
                    size, mtime = 0, 0
                entries.append((name, is_dir, size, mtime))
        except PermissionError:
            return self._fail(f"Permission denied accessing {directory}")

        if not entries:
            return self._ok(f"{directory} is empty.")

        entries.sort(key=lambda e: (not e[1], e[0].lower()))
        lines = [f"Contents of {os.path.basename(directory)} ({len(entries)} items):\n"]
        for name, is_dir, size, mtime in entries[:max_results]:
            if is_dir:
                lines.append(f"  📁 {name}/")
            else:
                size_str = (f"{size / 1024:.0f} KB" if size < 1024 * 1024
                            else f"{size / (1024 * 1024):.1f} MB")
                lines.append(f"  📄 {name} ({size_str})")
        if len(entries) > max_results:
            lines.append(f"\n  ...and {len(entries) - max_results} more")
        return self._ok("\n".join(lines))

    # ── Path extraction ───────────────────────────────────────────

    def _extract_path(self, command: str) -> str:
        """Try to extract a filesystem path from the command."""
        path_match = re.search(
            r'(?:in|from|at|to)\s+([A-Za-z]:[/\\][^\s,]+|/[^\s,]+|~[/\\][^\s,]+)',
            command)
        if path_match:
            return os.path.expanduser(path_match.group(1))

        known = {
            "downloads": os.path.expanduser("~/Downloads"),
            "desktop":   os.path.expanduser("~/Desktop"),
            "documents": os.path.expanduser("~/Documents"),
            "pictures":  os.path.expanduser("~/Pictures"),
            "music":     os.path.expanduser("~/Music"),
            "videos":    os.path.expanduser("~/Videos"),
        }
        cmd_lower = command.lower()
        for name, path in known.items():
            if name in cmd_lower:
                return path
        return self._config.get("default_directory", "")

    # ── Return helpers ────────────────────────────────────────────

    def _ok(self, msg: str) -> dict:
        return {"success": True,  "response": msg,
                "actions_taken": [{"action": "file_organizer"}], "spoken": False}

    def _fail(self, msg: str) -> dict:
        return {"success": False, "response": msg,
                "actions_taken": [], "spoken": False}
