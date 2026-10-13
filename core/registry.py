"""Hash registries: results/<dataset>/<split>/hash_<kind>.json maps a short hash to what it stands for.

Written at the start of every run: an existing hash is left untouched (and must describe the same
content), a new one is appended. The write is atomic and serialised with a file lock.
"""
import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def short_hash(value):
    """8 hex characters of sha256 over the canonical JSON of `value`."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def register(path, digest, entry, key):
    """Add `entry` under `digest` unless present; `entry[key]` must match an existing entry's."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name("." + path.name + ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = json.loads(path.read_text()) if path.exists() else {}
        if digest in registry:
            if registry[digest][key] != entry[key]:
                raise SystemExit(f"hash collision in {path}: {digest} already names different {key}")
            return digest
        registry[digest] = {**entry, "first_used": datetime.now(timezone.utc).isoformat()}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    return digest
