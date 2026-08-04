#!/usr/bin/env bash
# Install any missing runtime/test dependencies into a repository-local .deps directory
# for the given interpreter (default: python3), without touching its environment.
# handeye/__init__.py appends .deps to sys.path when the directory exists.
set -euo pipefail
PY="${1:-python3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
missing=()
"$PY" -c "import cv2; assert cv2.__version__ < '5'" 2>/dev/null || missing+=("opencv-python-headless>=4.8,<5")
for mod in numpy scipy matplotlib pytest hypothesis; do
  "$PY" -c "import $mod" 2>/dev/null || missing+=("$mod")
done
if [ ${#missing[@]} -eq 0 ]; then echo "all dependencies present"; exit 0; fi
"$PY" -m pip install --no-deps --target "$ROOT/.deps" "${missing[@]}"
