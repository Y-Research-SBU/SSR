"""Early startup hooks (must run before torch / project imports)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def bootstrap_cuda_from_argv() -> None:
    """Set CUDA_VISIBLE_DEVICES before torch is imported anywhere."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return
    argv = sys.argv[1:]
    for idx, arg in enumerate(argv):
        if arg == "--gpu" and idx + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = argv[idx + 1]
            return
        if arg.startswith("--gpu="):
            os.environ["CUDA_VISIBLE_DEVICES"] = arg.partition("=")[2]
            return


def bootstrap_venv_paths() -> None:
    """Fix sys.path when launched via exec -a python (nvidia-smi shows "python")."""
    import site

    root = Path(__file__).resolve().parents[1]
    venv = root / ".venv"
    if not (venv / "pyvenv.cfg").is_file():
        return

    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    site_dir = venv / "lib" / f"python{ver}" / "site-packages"
    if not site_dir.is_dir():
        return

    site_str = str(site_dir)
    needs_venv_fix = Path(sys.prefix).resolve() != venv.resolve()

    if needs_venv_fix:
        site.addsitedir(site_str)
        sys.prefix = str(venv)
        sys.exec_prefix = str(venv)
        sys.executable = str(venv / "bin" / "python")
        os.environ.setdefault("VIRTUAL_ENV", str(venv))
    elif site_str not in sys.path:
        site.addsitedir(site_str)


bootstrap_cuda_from_argv()
bootstrap_venv_paths()
