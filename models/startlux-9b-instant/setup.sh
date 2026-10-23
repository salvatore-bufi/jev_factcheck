#!/usr/bin/env bash
# One-time setup for startlux-9b-instant: its own Python environment (models/startlux-9b-instant/.venv) with the
# authors' inference package, startlux-decide, and the vLLM version its Instant plugin is validated on (0.31.0).
# A separate environment because the benchmark's own environment runs another vLLM (the plugin refuses it) and
# transformers 5.14+ (the package's transformers engine needs 5.8). Idempotent: an existing environment with the
# pinned package is kept. Needs Python 3.10+ with venv and internet access; downloads about 5 GB (vLLM, torch).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
PY="${PYTHON:-python3}"
COMMIT="a99977d9332decc9fb27a2488b4a83b533b6cd5f"     # StartLuxLabs/StartLux-Decision, 2026-10-23
SPEC="startlux-decide[vllm] @ git+https://github.com/StartLuxLabs/StartLux-Decision.git@$COMMIT"

if [ -x "$VENV/bin/startlux-decide" ] && grep -qs "$COMMIT" "$VENV/.startlux-commit"; then
  echo "Already set up: $VENV (startlux-decide @ ${COMMIT:0:7})"; exit 0
fi

[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install "$SPEC"
echo "$COMMIT" > "$VENV/.startlux-commit"

# Instant models need the package's vLLM plugin in the Python that runs vLLM (registered by the install).
"$VENV/bin/python" - <<'EOF'
from importlib.metadata import entry_points, version
names = [e.name for e in entry_points(group="vllm.general_plugins")]
assert "startlux_prefix_bidir" in names, f"the startlux vLLM plugin is not registered: {names}"
print("vllm", version("vllm"), "| startlux-decide", version("startlux-decide"), "| plugin startlux_prefix_bidir")
EOF
echo "Ready: $VENV/bin/startlux-decide"
