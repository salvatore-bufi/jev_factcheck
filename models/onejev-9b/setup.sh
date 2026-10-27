#!/usr/bin/env bash
# One-time setup for onejev-9b: its own Python environment (models/onejev-9b/.venv) with Flide, the serving engine the
# OneJev 1.6 card names (`flide serve --model OmniJev/OneJev1.6-9B`; the checkpoint's decision_config.json says
# {"format_id": "onejev-slot-v2", "package": "flide"}). A separate environment because Flide needs torch 2.12+ and
# flash-linear-attention 0.5.x, which the benchmark's environment does not pin. Idempotent: an existing environment with
# the pinned commit is kept. Needs Python 3.10+ with venv and internet access.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
PY="${PYTHON:-python3}"
COMMIT="08b87d033ef9bf76fc01b0a930517b67127b8e85"     # OmniJev/Flide 0.1.4, 2026-10-26
SPEC="flide @ git+https://github.com/OmniJev/Flide.git@$COMMIT"

if [ -x "$VENV/bin/flide" ] && grep -qs "$COMMIT" "$VENV/.flide-commit"; then
  echo "Already set up: $VENV (flide @ ${COMMIT:0:7})"; exit 0
fi

[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install "$SPEC" torchvision requests
echo "$COMMIT" > "$VENV/.flide-commit"

"$VENV/bin/flide" formats | grep -qi onejev || { echo "the OneJev format is not registered in Flide"; exit 1; }
"$VENV/bin/python" -c "from importlib.metadata import version; print('flide', version('flide'), '| torch', version('torch'))"
echo "Ready: $VENV/bin/flide"
