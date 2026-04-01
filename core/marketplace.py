"""MarketplaceClient — fetch, cache, and manage the talent catalog.

The catalog is a JSON file hosted on GitHub (or any URL). It lists
available talents with metadata: name, description, author, version,
download_url, category, etc.

Talents are downloaded as single .py files into talents/user/.
"""

import os
import json
import time
import requests
import ast
import inspect
from talents.base import BaseTalent

import logging
log = logging.getLogger(__name__)

# Default catalog URL — can be overridden in settings
DEFAULT_CATALOG_URL = (
    "https://raw.githubusercontent.com/talon-assistant/talent-catalog/main/catalog.json"
)

# How long to cache the catalog locally (seconds)
CACHE_TTL = 600  # 10 minutes


def _data_dir():
    """Ensure data/ directory exists and return its path."""
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(d, exist_ok=True)
    return d


def _user_talents_dir():
    """Ensure talents/user/ directory exists and return its path."""
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "talents", "user")
    os.makedirs(d, exist_ok=True)
    return d


class MarketplaceClient:
    """Handles catalog fetching, caching, downloading, and validation."""

    def __init__(self, catalog_url=None):
        self.catalog_url = catalog_url or DEFAULT_CATALOG_URL
        self._cache_path = os.path.join(_data_dir(), "marketplace_cache.json")
        self._catalog = None
        self._cache_time = 0

    # ── Catalog ────────────────────────────────────────────────────

    def get_catalog(self, force_refresh=False):
        """Return the talent catalog (list of dicts).

        Uses a local cache file to avoid hitting the network on every call.
        Returns [] on failure (never raises).
        """
        # Return in-memory cache if fresh
        if (not force_refresh
                and self._catalog is not None
                and (time.time() - self._cache_time) < CACHE_TTL):
            return self._catalog

        # Try disk cache
        if not force_refresh:
            disk = self._load_disk_cache()
            if disk is not None:
                self._catalog = disk
                self._cache_time = time.time()
                return self._catalog

        # Fetch from network
        catalog = self._fetch_remote_catalog()
        if catalog is not None:
            self._catalog = catalog
            self._cache_time = time.time()
            self._save_disk_cache(catalog)
            return self._catalog

        # Fall back to stale disk cache
        disk = self._load_disk_cache(ignore_ttl=True)
        if disk is not None:
            self._catalog = disk
            return self._catalog

        # Fall back to local catalog.json (dev/offline mode)
        local = self._load_local_catalog()
        if local is not None:
            self._catalog = local
            self._cache_time = time.time()
            return self._catalog

        return []

    def _fetch_remote_catalog(self):
        """Fetch catalog JSON from the configured URL."""
        try:
            log.info(f"[Marketplace] Fetching catalog from {self.catalog_url}")
            bust = f"?t={int(time.time())}"
            resp = requests.get(self.catalog_url + bust, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                talents = data if isinstance(data, list) else data.get("talents", [])
                log.info(f"[Marketplace] Got {len(talents)} talents from catalog")
                return talents
            else:
                log.info(f"[Marketplace] Catalog fetch returned {resp.status_code}")
                return None
        except Exception as e:
            log.error(f"[Marketplace] Catalog fetch error: {e}")
            return None

    def _load_local_catalog(self):
        """Load catalog from a local marketplace/catalog.json (dev fallback)."""
        local_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "marketplace", "catalog.json")
        try:
            if os.path.exists(local_path):
                with open(local_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                talents = data if isinstance(data, list) else data.get("talents", [])
                log.info(f"[Marketplace] Loaded {len(talents)} talents from local catalog")
                return talents
        except Exception as e:
            log.error(f"[Marketplace] Local catalog error: {e}")
        return None

    def _load_disk_cache(self, ignore_ttl=False):
        """Load catalog from disk cache if fresh enough."""
        try:
            if not os.path.exists(self._cache_path):
                return None
            with open(self._cache_path, 'r') as f:
                cached = json.load(f)
            cached_at = cached.get("cached_at", 0)
            if not ignore_ttl and (time.time() - cached_at) > CACHE_TTL:
                return None
            return cached.get("talents", [])
        except Exception:
            return None

    def _save_disk_cache(self, talents):
        """Save catalog to disk cache."""
        try:
            with open(self._cache_path, 'w') as f:
                json.dump({"cached_at": time.time(), "talents": talents}, f)
        except Exception as e:
            log.error(f"[Marketplace] Cache save error: {e}")

    # ── Install ────────────────────────────────────────────────────

    def install_talent(self, talent_entry):
        """Download and validate a talent from the catalog.

        Args:
            talent_entry: dict from catalog with at least 'download_url' and 'filename'

        Returns:
            dict: {"success": bool, "filepath": str, "error": str}
        """
        download_url = talent_entry.get("download_url", "")
        filename = talent_entry.get("filename", "")

        if not download_url or not filename:
            return {"success": False, "filepath": "",
                    "error": "Missing download_url or filename in catalog entry"}

        if not filename.endswith(".py"):
            return {"success": False, "filepath": "",
                    "error": "Talent file must be a .py file"}

        # Download the file
        try:
            log.info(f"[Marketplace] Downloading {filename} from {download_url}")
            bust = f"{'&' if '?' in download_url else '?'}t={int(time.time())}"
            resp = requests.get(download_url + bust, timeout=30)
            if resp.status_code != 200:
                return {"success": False, "filepath": "",
                        "error": f"Download failed (HTTP {resp.status_code})"}
            source_code = resp.text
        except Exception as e:
            return {"success": False, "filepath": "",
                    "error": f"Download error: {e}"}

        # Validate via AST (same approach as ImportTalentDialog)
        validation = self.validate_source(source_code, filename)
        if not validation["valid"]:
            return {"success": False, "filepath": "",
                    "error": validation["error"]}

        # Write to talents/user/
        dest_path = os.path.join(_user_talents_dir(), filename)
        try:
            with open(dest_path, 'w', encoding='utf-8') as f:
                f.write(source_code)
            log.info(f"[Marketplace] Saved to {dest_path}")
            return {"success": True, "filepath": dest_path, "error": ""}
        except Exception as e:
            return {"success": False, "filepath": "",
                    "error": f"File write error: {e}"}

    # ── Uninstall ──────────────────────────────────────────────────

    def uninstall_talent(self, talent_name):
        """Remove a user-installed talent file from talents/user/.

        Returns:
            dict: {"success": bool, "error": str}
        """
        user_dir = _user_talents_dir()

        # Find the file — scan for a .py file containing a class with this name
        for fname in os.listdir(user_dir):
            if not fname.endswith('.py') or fname.startswith('__'):
                continue
            fpath = os.path.join(user_dir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    source = f.read()
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check for name = "talent_name" assignment
                        for item in node.body:
                            if (isinstance(item, ast.Assign)
                                    and len(item.targets) == 1
                                    and isinstance(item.targets[0], ast.Name)
                                    and item.targets[0].id == "name"
                                    and isinstance(item.value, ast.Constant)
                                    and item.value.value == talent_name):
                                os.remove(fpath)
                                log.info(f"[Marketplace] Removed {fpath}")
                                return {"success": True, "error": ""}
            except Exception:
                continue

        return {"success": False, "error": f"Could not find talent file for '{talent_name}'"}

    # ── Validation ─────────────────────────────────────────────────

    @staticmethod
    def validate_source(source_code, filename="talent.py"):
        """AST-based validation of a talent source file.

        Checks:
        - Valid Python syntax
        - Contains at least one BaseTalent subclass
        - Has required class attributes (name, description)

        Returns:
            dict: {"valid": bool, "error": str, "talent_info": dict}
        """
        try:
            tree = ast.parse(source_code, filename=filename)
        except SyntaxError as e:
            return {"valid": False, "error": f"Syntax error: {e}",
                    "talent_info": {}}

        # Find BaseTalent subclasses
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            # Check if any base class refers to BaseTalent
            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name == "BaseTalent":
                    # Extract class-level attributes
                    info = {"class_name": node.name}
                    for item in node.body:
                        if isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name) and isinstance(item.value, ast.Constant):
                                    info[target.id] = item.value.value
                    found.append(info)

        if not found:
            return {"valid": False,
                    "error": "No BaseTalent subclass found in file",
                    "talent_info": {}}

        talent_info = found[0]
        if "name" not in talent_info:
            return {"valid": False,
                    "error": "Talent class missing 'name' attribute",
                    "talent_info": talent_info}

        return {"valid": True, "error": "", "talent_info": talent_info}

    # ── Installed status ───────────────────────────────────────────

    @staticmethod
    def get_installed_talent_names():
        """Return a set of talent names currently installed in talents/user/."""
        user_dir = _user_talents_dir()
        names = set()

        for fname in os.listdir(user_dir):
            if not fname.endswith('.py') or fname.startswith('__'):
                continue
            fpath = os.path.join(user_dir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    source = f.read()
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        for item in node.body:
                            if (isinstance(item, ast.Assign)
                                    and len(item.targets) == 1
                                    and isinstance(item.targets[0], ast.Name)
                                    and item.targets[0].id == "name"
                                    and isinstance(item.value, ast.Constant)):
                                names.add(item.value.value)
            except Exception:
                continue

        return names
