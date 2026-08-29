#!/usr/bin/env python3
"""Talon Assistant — Setup & Migration Script

Run after cloning or pulling updates:
    python setup.py

What it does:
  1. Creates required directories (data/, documents/, talents/user/, config/)
  2. Copies *.example.json -> *.json for any missing config files (fresh install)
  3. Deep-merges new keys from example configs into existing configs (upgrade)
     - NEVER overwrites user values — only adds keys that don't exist yet
  4. Reports what changed so users know what happened

Safe to run repeatedly — idempotent by design.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime


# ── Paths ────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.resolve()
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "documents"
USER_TALENTS_DIR = ROOT / "talents" / "user"

REQUIRED_DIRS = [CONFIG_DIR, DATA_DIR, DOCS_DIR, USER_TALENTS_DIR]

CONFIG_FILES = [
    ("settings.example.json", "settings.json"),
    ("hue_config.example.json", "hue_config.json"),
    ("talents.example.json", "talents.json"),
    ("news_digest.example.json", "news_digest.json"),
    ("scheduled_tasks.example.json", "scheduled_tasks.json"),
]


# ── Deep merge ───────────────────────────────────────────────────

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge base into override.

    Returns a new dict where:
      - Keys in override are preserved as-is (user values win)
      - Keys in base that are missing from override are added
      - Nested dicts are merged recursively
    """
    result = dict(override)
    for key, base_val in base.items():
        if key not in result:
            result[key] = base_val
        elif isinstance(base_val, dict) and isinstance(result[key], dict):
            result[key] = deep_merge(base_val, result[key])
        # else: user value wins, don't touch it
    return result


def find_new_keys(base: dict, existing: dict, prefix="") -> list:
    """Return list of dot-paths for keys in base that are missing from existing."""
    new_keys = []
    for key, val in base.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in existing:
            new_keys.append(path)
        elif isinstance(val, dict) and isinstance(existing.get(key), dict):
            new_keys.extend(find_new_keys(val, existing[key], path))
    return new_keys


# ── Main setup ───────────────────────────────────────────────────

def setup():
    print("=" * 50)
    print("  Talon Assistant — Setup & Migration")
    print("=" * 50)
    print()

    changes = []

    # 1. Create directories
    print("[1/3] Checking directories...")
    for d in REQUIRED_DIRS:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            changes.append(f"  Created: {d.relative_to(ROOT)}/")
            print(f"  + Created {d.relative_to(ROOT)}/")
        else:
            print(f"  . {d.relative_to(ROOT)}/ exists")

    # Ensure talents/user/__init__.py exists
    init_file = USER_TALENTS_DIR / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")
        changes.append("  Created: talents/user/__init__.py")
        print("  + Created talents/user/__init__.py")

    print()

    # 2. Config files
    print("[2/3] Checking config files...")
    for example_name, config_name in CONFIG_FILES:
        example_path = CONFIG_DIR / example_name
        config_path = CONFIG_DIR / config_name

        if not example_path.exists():
            print(f"  ! {example_name} not found (skipping)")
            continue

        if not config_path.exists():
            # Fresh install — copy example as-is
            shutil.copy2(example_path, config_path)
            changes.append(f"  Created: config/{config_name} (from example)")
            print(f"  + Created config/{config_name} from {example_name}")
        else:
            # Upgrade — deep merge new keys without overwriting
            try:
                with open(example_path, 'r', encoding='utf-8') as f:
                    example_data = json.load(f)
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)

                new_keys = find_new_keys(example_data, user_data)

                if new_keys:
                    # Backup before modifying
                    backup_name = f"{config_name}.backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    backup_path = CONFIG_DIR / backup_name
                    shutil.copy2(config_path, backup_path)

                    merged = deep_merge(example_data, user_data)
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(merged, f, indent=2, ensure_ascii=False)
                        f.write("\n")

                    for k in new_keys:
                        changes.append(f"  Added key: {config_name} -> {k}")
                    print(f"  ~ Updated config/{config_name} (+{len(new_keys)} new keys)")
                    print(f"    Backup: {backup_name}")
                    for k in new_keys:
                        print(f"    + {k}")
                else:
                    print(f"  . config/{config_name} is up to date")

            except (json.JSONDecodeError, IOError) as e:
                print(f"  ! Error processing {config_name}: {e}")
                print(f"    Your config was NOT modified.")

    print()

    # 3. Summary
    print("[3/3] Summary")
    if changes:
        print(f"  {len(changes)} change(s) made:")
        for c in changes:
            print(c)
    else:
        print("  Everything is up to date. No changes needed.")

    print()
    print("Setup complete! Run 'python main.py' to start Talon.")
    print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(setup())
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
