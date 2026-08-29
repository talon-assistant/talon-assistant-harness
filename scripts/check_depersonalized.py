"""Fail CI when publishable repository content contains common local PII.

This deliberately excludes synthetic classifier/training fixtures, whose job
is to contain realistic-looking emails and phone numbers. It does not rewrite
Git history; run a separate history audit before publishing an existing repo.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LIVE_CONFIGS = {
    "config/settings.json",
    "config/talents.json",
    "config/hue_config.json",
    "config/mcp_servers.json",
    "config/hermes_clients.json",
    "config/news_digest.json",
    "config/scheduled_tasks.json",
}
SYNTHETIC_PREFIXES = (
    "data/security_classifier/",
    "scripts/gen_external_training_data.py",
    "scripts/generate_security_training_data.py",
    "tests/",
)
TEXT_SUFFIXES = {
    ".py", ".md", ".json", ".jsonl", ".txt", ".yml", ".yaml",
    ".toml", ".ini", ".cfg", ".html", ".css", ".qss",
}
HOME_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]+Users[\\/]+(?!you(?:[\\/]|$))[^\\/\s]+|"
    r"/Users/[^/\s]+|/home/[^/\s]+)",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])([A-Za-z0-9._%+-]+)@"
    r"([A-Za-z0-9.-]+\.[A-Za-z]{2,})(?![A-Za-z0-9.-])"
)
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[-. (]*)?(?:\(\d{3}\)|\d{3}[-. ])"
    r"\s*\d{3}[-. ]\d{4}(?!\d)"
)
SAFE_EMAIL_DOMAINS = {
    "example.com", "example.net", "example.org", "b.com",
}


def repository_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return sorted({line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()})


def scan_text(relative_path: str, text: str) -> list[str]:
    issues = []
    if HOME_PATH_RE.search(text):
        issues.append("absolute user-home path")
    if not relative_path.startswith(SYNTHETIC_PREFIXES):
        unsafe_emails = [
            match.group(0) for match in EMAIL_RE.finditer(text)
            if match.group(2).lower() not in SAFE_EMAIL_DOMAINS
        ]
        if unsafe_emails:
            issues.append("non-example email address")
        if PHONE_RE.search(text):
            issues.append("phone-number pattern")
    return issues


def scan_pdf(path: Path) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return []
    try:
        reader = PdfReader(str(path))
        metadata = " ".join(str(value) for value in (reader.metadata or {}).values())
        text = metadata + "\n" + "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    except Exception as exc:
        return [f"PDF inspection failed: {type(exc).__name__}"]
    return scan_text(path.as_posix(), text)


def main() -> int:
    files = repository_files()
    findings: list[tuple[str, str]] = []
    for relative in files:
        path = ROOT / relative
        if not path.is_file():
            continue
        if relative == "scripts/check_depersonalized.py":
            continue
        if relative in LIVE_CONFIGS:
            findings.append((relative, "live user configuration is tracked"))
            continue
        if path.suffix.lower() == ".pdf":
            findings.extend((relative, issue) for issue in scan_pdf(path))
        elif path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitignore"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeError:
                findings.append((relative, "text file is not valid UTF-8"))
                continue
            findings.extend((relative, issue) for issue in scan_text(relative, text))

    settings_path = ROOT / "config" / "settings.example.json"
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        profile = settings.get("user_profile", {})
        populated = [
            key for key in ("display_name", "email", "phone", "location")
            if str(profile.get(key, "") or "").strip()
        ]
        if populated:
            findings.append((
                "config/settings.example.json",
                "user_profile defaults must be blank: " + ", ".join(populated),
            ))
    except (OSError, json.JSONDecodeError) as exc:
        findings.append(("config/settings.example.json", f"cannot validate: {exc}"))

    if findings:
        print("Depersonalization check failed:")
        for path, issue in findings:
            print(f"- {path}: {issue}")
        return 1
    print(f"Depersonalization check passed ({len(files)} publishable files scanned).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
