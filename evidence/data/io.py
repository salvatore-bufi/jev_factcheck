"""Reading and writing built rows: jsonl files, their statistics, and the folder manifest."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dataset" / "evidence"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_jsonl(path, rows):
    _atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def write_json(path, obj):
    _atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def update_manifest(out_dir, source, entry):
    """Record one source's output in out_dir/manifest.json, keeping the other sources' entries."""
    path = Path(out_dir) / "manifest.json"
    manifest = json.loads(path.read_text()) if path.is_file() else {"sources": {}}
    manifest["sources"][source] = entry
    write_json(path, manifest)
