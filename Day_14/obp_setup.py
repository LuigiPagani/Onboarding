from __future__ import annotations

import glob
import importlib.util
import site
import shutil
import subprocess
import sys
from typing import Tuple


def _pip_install(pkg: str) -> Tuple[int, str]:
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", pkg]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode, (res.stderr or res.stdout or "").strip()


def _ensure_pkg(pkg: str, alt_pkgs: list[str] | None = None) -> None:
    if importlib.util.find_spec(pkg) is not None:
        return

    # Try the primary package name first.
    code, msg = _pip_install(pkg)
    if code == 0:
        return

    # Try alternate package names if provided.
    if alt_pkgs:
        for alt in alt_pkgs:
            code, msg = _pip_install(alt)
            if code == 0:
                return

    raise RuntimeError(
        "Package install failed. Try manually in a terminal:\n"
        f"  {sys.executable} -m pip install {pkg}\n"
        "If this is a network issue, you may need internet access or a local wheel.\n"
        f"pip output:\n{msg}"
    )


def _fix_numpy_scipy() -> None:
    try:
        import numpy as np

        _ = np.char  # fails if numpy is broken
        import scipy  # noqa: F401
        return
    except Exception:
        # Remove broken installs (pip reinstall can leave corrupted files behind)
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            for path in glob.glob(str(sp) + "/numpy*"):
                shutil.rmtree(path, ignore_errors=True)
            for path in glob.glob(str(sp) + "/scipy*"):
                shutil.rmtree(path, ignore_errors=True)

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--upgrade",
                "numpy==1.26.4",
                "scipy==1.10.1",
            ]
        )
        # Clear cached modules so the fresh installs load.
        for m in ["numpy", "scipy"]:
            if m in sys.modules:
                del sys.modules[m]


def load_obd(campaign: str = "all") -> Tuple[object, dict]:
    # Core libs
    _ensure_pkg("numpy", alt_pkgs=["numpy<2.0"])
    _ensure_pkg("pandas")
    # OBP pulls seaborn, which expects matplotlib to have register_cmap.
    _ensure_pkg("matplotlib", alt_pkgs=["matplotlib<3.8"])
    _ensure_pkg("scikit-learn")

    _fix_numpy_scipy()

    # OBP (Open Bandit Pipeline)
    _ensure_pkg("obp", alt_pkgs=["open-bandit-pipeline"])

    from obp.dataset import OpenBanditDataset

    last_err = None
    obd = None
    for behavior_policy in ["bts", "random"]:
        try:
            obd = OpenBanditDataset(behavior_policy=behavior_policy, campaign=campaign)
            print(f"Loaded OBD with behavior_policy={behavior_policy!r}, campaign={campaign!r}")
            break
        except Exception as e:
            last_err = e

    if obd is None:
        raise RuntimeError(f"Failed to load OpenBanditDataset: {last_err}")

    bandit_feedback = obd.obtain_batch_bandit_feedback()
    return obd, bandit_feedback

