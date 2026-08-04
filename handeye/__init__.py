"""Hand-eye calibration linting, robust solving and benchmarking."""
import sys as _sys
from pathlib import Path as _Path

# Optional repository-local dependency directory (see scripts/install_local_deps.sh);
# appended so that packages installed in the active environment take precedence.
_deps = _Path(__file__).resolve().parent.parent / ".deps"
if _deps.is_dir() and str(_deps) not in _sys.path:
    _sys.path.append(str(_deps))

__version__ = "0.1.0"
