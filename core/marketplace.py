"""MarketplaceClient — fetch, cache, and manage the talent catalog.

The catalog is a JSON file hosted on GitHub (or any URL). It lists
available talents with metadata: name, description, author, version,
download_url, category, etc.

Talents are downloaded as single .py files into talents/user/.
"""

import os
import json
import time
import hashlib
import requests
import ast
import inspect
from urllib.parse import urlparse
from talents.base import BaseTalent
from core.capability_manifest import validate_third_party_manifest

import logging
log = logging.getLogger(__name__)

# Default catalog URL — can be overridden in settings
DEFAULT_CATALOG_URL = (
    "https://raw.githubusercontent.com/talon-assistant/talent-catalog/main/catalog.json"
)

# How long to cache the catalog locally (seconds)
CACHE_TTL = 600  # 10 minutes

# Hosts a talent .py may be downloaded from. The configured catalog's own
# host is always added to this set at runtime. Anything else is rejected so
# a tampered catalog entry can't point the download at an arbitrary server.
_ALLOWED_DOWNLOAD_HOSTS = frozenset({
    "raw.githubusercontent.com",
    "github.com",
    "objects.githubusercontent.com",
    "codeload.github.com",
    "gist.githubusercontent.com",
})

# Hard ceiling on a downloaded talent file (bytes). A talent is a single .py;
# anything larger is almost certainly hostile or a misconfigured URL.
_MAX_TALENT_BYTES = 1_000_000


def _host_allowed(url, catalog_url):
    """True if `url` is https and its host is on the download allowlist.

    The catalog's own host is always permitted so a self-hosted catalog can
    serve its own talents without extra configuration.
    """
    try:
        p = urlparse(url)
    except Exception:
        return False
    if p.scheme != "https":
        return False
    host = (p.hostname or "").lower()
    if not host:
        return False
    allowed = set(_ALLOWED_DOWNLOAD_HOSTS)
    try:
        cat_host = (urlparse(catalog_url).hostname or "").lower()
        if cat_host:
            allowed.add(cat_host)
    except Exception:
        pass
    return host in allowed


# AST node patterns that almost never appear in a legitimate talent and are
# the classic obfuscation / sandbox-escape tells. Surfaced to the user in the
# install review step rather than hard-blocked: the human approval is the gate.
_DANGEROUS_DUNDERS = frozenset({
    "__subclasses__", "__globals__", "__bases__", "__mro__",
    "__builtins__", "__base__", "__code__", "__import__",
})
_SUBPROCESS_FUNCS = frozenset({
    "system", "popen", "Popen", "run", "call", "check_call", "check_output",
})


def _scan_source_security(tree):
    """Return a list of human-readable warnings about risky constructs.

    This is advisory, not a verdict — talents legitimately need real power
    (file access, network, subprocess). The list is shown to the user before
    the code is ever written or imported so they can make an informed call.
    """
    warnings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            # Bare calls: eval(...), exec(...), compile(...), __import__(...)
            if isinstance(fn, ast.Name) and fn.id in (
                "eval", "exec", "compile", "__import__",
            ):
                warnings.append(f"line {node.lineno}: calls {fn.id}()")
            # Attribute calls: os.system / os.popen / subprocess.*
            elif isinstance(fn, ast.Attribute):
                attr = fn.attr
                mod = fn.value.id if isinstance(fn.value, ast.Name) else ""
                if mod == "os" and attr in ("system", "popen"):
                    warnings.append(f"line {node.lineno}: calls os.{attr}()")
                elif mod == "subprocess" and attr in _SUBPROCESS_FUNCS:
                    shelled = any(
                        kw.arg == "shell"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                        for kw in node.keywords
                    )
                    suffix = " with shell=True" if shelled else ""
                    warnings.append(
                        f"line {node.lineno}: subprocess.{attr}(){suffix}")
                elif attr in ("loads", "load") and mod in ("pickle", "marshal"):
                    warnings.append(
                        f"line {node.lineno}: {mod}.{attr}() (arbitrary "
                        "object/code deserialization)")
        # Attribute access to dunder internals used for sandbox escapes
        elif isinstance(node, ast.Attribute) and node.attr in _DANGEROUS_DUNDERS:
            warnings.append(f"line {node.lineno}: accesses {node.attr}")
    return warnings


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

    def __init__(self, catalog_url=None, capabilities=None,
                 command_source="local"):
        self.catalog_url = catalog_url or DEFAULT_CATALOG_URL
        self._cache_path = os.path.join(_data_dir(), "marketplace_cache.json")
        self._catalog = None
        self._cache_time = 0
        self.capabilities = capabilities
        self.command_source = command_source

    def request_plugin_change(self, operation, talent_name, filename=""):
        """Create a broker authorization for an install/remove operation."""
        if not self.capabilities:
            return None
        verb = "Install" if operation == "install" else "Remove"
        return self.capabilities.request(
            "plugin_install",
            source=self.command_source,
            summary=f"{verb} talent {talent_name!r}",
            metadata={"operation": operation, "talent": talent_name,
                      "filename": filename},
        )

    def _consume_plugin_authorization(self, authorization, *, operation,
                                      target):
        """Validate and consume the one-time grant immediately before write."""
        if not self.capabilities:
            return None, ""
        if authorization is None:
            return None, "Plugin change is missing capability authorization"
        if authorization.confirmation_required:
            request = self.capabilities.approve(
                authorization.request.request_id, source=self.command_source)
            if request is None:
                return None, "Plugin approval expired or did not match this session"
        elif authorization.allowed:
            request = authorization.request
        else:
            return None, self.capabilities.denial_message(authorization)
        if (request.capability != "plugin_install"
                or request.source != self.command_source):
            return None, "Plugin approval does not match this operation"
        metadata = request.metadata
        if metadata.get("operation") != operation:
            return None, "Plugin approval does not match this operation"
        approved_target = (metadata.get("filename") if operation == "install"
                           else metadata.get("talent"))
        if approved_target != target:
            return None, "Plugin approval does not match this target"
        return request, ""

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
            with open(self._cache_path, 'r', encoding="utf-8") as f:
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
            with open(self._cache_path, 'w', encoding="utf-8") as f:
                json.dump({"cached_at": time.time(), "talents": talents}, f)
        except Exception as e:
            log.error(f"[Marketplace] Cache save error: {e}")

    # ── Install ────────────────────────────────────────────────────

    def fetch_and_validate(self, talent_entry):
        """Download a talent and validate it WITHOUT writing it to disk.

        Enforces transport security (https + host allowlist), integrity
        (sha256 when the catalog entry provides one), structural validity,
        and a static danger scan. The source is returned so the caller can
        show it to the user for explicit approval before commit_install().

        Args:
            talent_entry: catalog dict with 'download_url' and 'filename'
                          (optional 'sha256' for integrity verification).

        Returns:
            dict: {success, source_code, filename, talent_info, warnings,
                   verified, error}
        """
        download_url = talent_entry.get("download_url", "")
        filename = talent_entry.get("filename", "")
        expected_sha = (talent_entry.get("sha256") or "").strip().lower()

        if not download_url or not filename:
            return {"success": False,
                    "error": "Missing download_url or filename in catalog entry"}

        # Reject path traversal / nested paths in the filename.
        if (not filename.endswith(".py")
                or filename != os.path.basename(filename)
                or filename.startswith(".")):
            return {"success": False,
                    "error": "Talent filename must be a simple .py file name"}

        # Transport security: https + pinned host allowlist.
        if not _host_allowed(download_url, self.catalog_url):
            return {"success": False,
                    "error": ("Refusing to download: URL must be https and on "
                              f"an allowed host. Got: {download_url}")}

        # Download the file (cap size, verify the final URL host too in case
        # an allowed host redirected somewhere untrusted).
        try:
            log.info(f"[Marketplace] Downloading {filename} from {download_url}")
            bust = f"{'&' if '?' in download_url else '?'}t={int(time.time())}"
            resp = requests.get(download_url + bust, timeout=30,
                                stream=True)
            if resp.status_code != 200:
                return {"success": False,
                        "error": f"Download failed (HTTP {resp.status_code})"}
            if not _host_allowed(resp.url, self.catalog_url):
                return {"success": False,
                        "error": f"Download redirected to a disallowed host: {resp.url}"}
            raw = resp.raw.read(_MAX_TALENT_BYTES + 1, decode_content=True)
            if len(raw) > _MAX_TALENT_BYTES:
                return {"success": False,
                        "error": f"Talent file exceeds {_MAX_TALENT_BYTES} bytes"}
            source_code = raw.decode("utf-8", errors="replace")
        except Exception as e:
            return {"success": False, "error": f"Download error: {e}"}

        # Integrity: if the catalog declares a hash, it MUST match.
        actual_sha = hashlib.sha256(raw).hexdigest()
        if expected_sha:
            if actual_sha != expected_sha:
                return {"success": False,
                        "error": ("Integrity check failed: sha256 mismatch "
                                  f"(expected {expected_sha[:12]}…, "
                                  f"got {actual_sha[:12]}…)")}
            verified = True
        else:
            verified = False  # no declared hash — caller should warn the user

        # Structural validation + static danger scan.
        validation = self.validate_source(source_code, filename)
        if not validation["valid"]:
            return {"success": False, "error": validation["error"]}

        return {
            "success": True,
            "source_code": source_code,
            "filename": filename,
            "talent_info": validation.get("talent_info", {}),
            "warnings": validation.get("warnings", []),
            "verified": verified,
            "sha256": actual_sha,
            "error": "",
        }

    def commit_install(self, filename, source_code, authorization=None):
        """Write already-validated talent source to talents/user/.

        Call this only after fetch_and_validate() succeeded and the user has
        reviewed the source. Re-validates defensively before writing.

        Returns:
            dict: {"success": bool, "filepath": str, "error": str}
        """
        if (not filename.endswith(".py")
                or filename != os.path.basename(filename)
                or filename.startswith(".")):
            return {"success": False, "filepath": "",
                    "error": "Invalid talent filename"}

        validation = self.validate_source(source_code, filename)
        if not validation["valid"]:
            return {"success": False, "filepath": "",
                    "error": validation["error"]}

        approved_request, auth_error = self._consume_plugin_authorization(
            authorization, operation="install", target=filename)
        if auth_error:
            return {"success": False, "filepath": "", "error": auth_error}

        dest_path = os.path.join(_user_talents_dir(), filename)
        try:
            with open(dest_path, 'w', encoding='utf-8') as f:
                f.write(source_code)
            log.info(f"[Marketplace] Saved to {dest_path}")
            if approved_request:
                self.capabilities.record_outcome(
                    approved_request, success=True)
            return {"success": True, "filepath": dest_path, "error": ""}
        except Exception as e:
            if approved_request:
                self.capabilities.record_outcome(
                    approved_request, success=False, error=str(e))
            return {"success": False, "filepath": "",
                    "error": f"File write error: {e}"}

    # ── Uninstall ──────────────────────────────────────────────────

    def uninstall_talent(self, talent_name, authorization=None):
        """Remove a user-installed talent file from talents/user/.

        Returns:
            dict: {"success": bool, "error": str}
        """
        approved_request, auth_error = self._consume_plugin_authorization(
            authorization, operation="remove", target=talent_name)
        if auth_error:
            return {"success": False, "error": auth_error}

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
                                if approved_request:
                                    self.capabilities.record_outcome(
                                        approved_request, success=True)
                                return {"success": True, "error": ""}
            except Exception:
                continue

        error = f"Could not find talent file for '{talent_name}'"
        if approved_request:
            self.capabilities.record_outcome(
                approved_request, success=False, error=error)
        return {"success": False, "error": error}

    # ── Validation ─────────────────────────────────────────────────

    @staticmethod
    def validate_source(source_code, filename="talent.py"):
        """AST-based validation of a talent source file.

        Checks:
        - Valid Python syntax
        - Contains at least one BaseTalent subclass
        - Has required class attributes (name, description)
        - Scans for risky constructs (advisory warnings, not a hard block)

        Returns:
            dict: {"valid": bool, "error": str, "talent_info": dict,
                   "warnings": list[str]}
        """
        try:
            tree = ast.parse(source_code, filename=filename)
        except SyntaxError as e:
            return {"valid": False, "error": f"Syntax error: {e}",
                    "talent_info": {}, "warnings": []}

        warnings = _scan_source_security(tree)

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
                                if not isinstance(target, ast.Name):
                                    continue
                                if isinstance(item.value, ast.Constant):
                                    info[target.id] = item.value.value
                                elif target.id == "capability_manifest":
                                    try:
                                        info[target.id] = ast.literal_eval(item.value)
                                    except (ValueError, TypeError):
                                        info[target.id] = None
                    found.append(info)

        if not found:
            return {"valid": False,
                    "error": "No BaseTalent subclass found in file",
                    "talent_info": {}, "warnings": warnings}

        talent_info = found[0]
        if "name" not in talent_info:
            return {"valid": False,
                    "error": "Talent class missing 'name' attribute",
                    "talent_info": talent_info, "warnings": warnings}

        manifest = talent_info.get("capability_manifest")
        if not isinstance(manifest, dict):
            return {
                "valid": False,
                "error": "Talent class missing a literal capability_manifest",
                "talent_info": talent_info,
                "warnings": warnings,
            }
        declaration = validate_third_party_manifest(
            manifest, str(talent_info.get("name", filename))
        )
        if declaration.status == "undeclared":
            return {
                "valid": False,
                "error": declaration.detail,
                "talent_info": talent_info,
                "warnings": warnings,
            }

        return {"valid": True, "error": "", "talent_info": talent_info,
                "warnings": warnings}

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
