from __future__ import annotations

import os
import random
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np


def detect_environment(environment: str = "auto") -> dict[str, bool | str]:
    env = str(environment or "auto").lower()
    is_kaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/input").exists()
    is_colab = bool(os.environ.get("COLAB_RELEASE_TAG")) or (Path("/content").exists() and "google.colab" in str(os.environ.get("PYTHONPATH", "")))
    if env == "kaggle":
        is_kaggle, is_colab = True, False
    elif env == "colab":
        is_kaggle, is_colab = False, True
    elif env == "local":
        is_kaggle, is_colab = False, False
    label = "kaggle" if is_kaggle else ("colab" if is_colab else "local")
    return {"is_kaggle": is_kaggle, "is_colab": is_colab, "environment_label": label}


def suppress_known_warnings() -> None:
    """Hide known non-actionable runtime warnings while preserving real errors."""
    warnings.filterwarnings("ignore", message=r".*'penalty' was deprecated.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=r".*The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=r".*l1_ratio parameter is only used.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=r".*Falling back to prediction using DMatrix due to mismatched devices.*", category=UserWarning)
    os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")


def set_reproducibility(seed: int = 42) -> None:
    suppress_known_warnings()
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def log(message: str) -> None:
    print(message, flush=True)


def _candidate_paths(csv_path: str | None, csv_candidates: Iterable[str] = ()) -> list[Path]:
    out: list[Path] = []
    names: list[str] = []
    if csv_path:
        names.append(str(csv_path))
    names.extend([str(x) for x in csv_candidates or ()])
    roots = [Path.cwd(), Path("/mnt/data"), Path("/kaggle/input"), Path("/content")]
    seen: set[str] = set()
    for name in names:
        p = Path(name).expanduser()
        candidates = [p] if p.is_absolute() else [Path.cwd() / p, Path("/mnt/data") / p]
        for root in roots:
            candidates.append(root / p)
        for cand in candidates:
            key = str(cand)
            if key not in seen:
                seen.add(key)
                out.append(cand)
    return out


def find_csv_path(csv_path: str | None, csv_candidates: Iterable[str] = ()) -> Path:
    for path in _candidate_paths(csv_path, csv_candidates):
        if path.exists() and path.is_file():
            return path.resolve()

    search_roots = [Path.cwd(), Path("/mnt/data"), Path("/kaggle/input"), Path("/content")]
    names = []
    if csv_path:
        names.append(Path(str(csv_path)).name)
    names.extend([Path(str(x)).name for x in csv_candidates or ()])
    names = [x for x in names if x]
    for root in search_roots:
        if not root.exists():
            continue
        for name in names:
            matches = list(root.rglob(name))[:3]
            if matches:
                return matches[0].resolve()

    message = "Could not find CSV file. Checked explicit path and common local/Kaggle/Colab roots."
    if csv_path:
        message += f" csv_path={csv_path!r}."
    if csv_candidates:
        message += f" csv_candidates={list(csv_candidates)!r}."
    raise FileNotFoundError(message)


def package_available(module_name: str) -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def gpu_status(use_gpu: str | bool | None = "auto") -> dict[str, object]:
    """Detect GPU availability in local, Colab and Kaggle-like environments.

    The tabular pipeline is primarily scikit-learn based. GPU acceleration is
    therefore opportunistic and currently routed only to GPU-capable estimators,
    mainly XGBoost when available.
    """
    requested = str(use_gpu).lower() if not isinstance(use_gpu, bool) else ("true" if use_gpu else "false")
    if requested in {"0", "false", "no", "off", "cpu"}:
        return {
            "gpu_requested": False,
            "gpu_available": False,
            "gpu_backend": "disabled_by_configuration",
            "gpu_device": "none",
            "gpu_usage_note": "GPU usage disabled by configuration; all estimators run on CPU.",
        }
    available = False
    backend = "none"
    device = "none_detected"
    try:
        import torch  # type: ignore
        if bool(torch.cuda.is_available()):
            available = True
            backend = "torch.cuda"
            device = str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    if not available and shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], stderr=subprocess.DEVNULL, text=True, timeout=3).strip().splitlines()
            if out:
                available = True
                backend = "nvidia-smi"
                device = out[0].strip()
        except Exception:
            pass
    return {
        "gpu_requested": requested in {"auto", "1", "true", "yes", "on", "gpu", "cuda"},
        "gpu_available": bool(available),
        "gpu_backend": backend,
        "gpu_device": device,
        "gpu_usage_note": (
            "GPU detected; GPU availability is reported. XGBoost CUDA is optional, but the default safe policy keeps XGBoost on CPU inside sklearn pipelines to avoid CPU/GPU prediction-device mismatch."
            if available else
            "No compatible GPU was detected. All estimators and mitigation refits will run on CPU."
        ),
    }


def detect_gpu_availability(use_gpu: str | bool = "auto") -> dict[str, object]:
    return gpu_status(use_gpu)


def detect_gpu_status(use_gpu: str | bool | None = "auto") -> dict[str, object]:
    return gpu_status(use_gpu)


def detect_accelerators(use_gpu: str | bool | None = "auto") -> dict[str, object]:
    return gpu_status(use_gpu)


def accelerator_status(use_gpu: str | bool | None = "auto") -> dict[str, object]:
    return gpu_status(use_gpu)


def fairness_support_status(*args, **kwargs) -> dict[str, bool | str]:
    """Report external fairness-library availability and the active fallback mode."""
    first = args[0] if args else None
    use_gpu = getattr(first, "use_gpu", first if first is not None else kwargs.get("use_gpu", "auto"))
    xgb_policy = str(getattr(first, "xgboost_device_policy", kwargs.get("xgboost_device_policy", "safe_cpu")) or "safe_cpu").lower()
    fairlearn = package_available("fairlearn")
    aif360 = package_available("aif360")
    imblearn = package_available("imblearn")
    if fairlearn and aif360:
        mode = "external fairness libraries available; built-in support operations also enabled"
    elif fairlearn:
        mode = "fairlearn available; AIF360 unavailable; built-in support operations enabled"
    elif aif360:
        mode = "AIF360 available; fairlearn unavailable; built-in support operations enabled"
    else:
        mode = "support operations fallback: Fairlearn/AIF360 unavailable"
    gpu = gpu_status(use_gpu)
    return {
        "fairlearn_available": fairlearn,
        "aif360_available": aif360,
        "imbalanced_learn_available": imblearn,
        "fairness_support_mode": mode,
        "support_operations_available": True,
        "gpu_requested": bool(gpu.get("gpu_requested", False)),
        "gpu_available": bool(gpu.get("gpu_available", False)),
        "gpu_name": str(gpu.get("gpu_device", "none")),
        "gpu_backend": str(gpu.get("gpu_backend", "none")),
        "gpu_detection_source": str(gpu.get("gpu_backend", "none")),
        "torch_cuda_device": str(gpu.get("gpu_device", "none")),
        "xgboost_gpu_usable": bool(gpu.get("gpu_available", False) and package_available("xgboost")),
        "xgboost_default_device_policy": "safe_cpu",
        "xgboost_device_policy": xgb_policy,
        "xgboost_effective_device": "cuda_for_fit_cpu_for_prediction" if (xgb_policy in {"cuda", "gpu", "force_cuda", "cuda_if_available", "auto_cuda"} and bool(gpu.get("gpu_available", False))) else "cpu",
        "gpu_acceleration_note": str(gpu.get("gpu_usage_note", "GPU availability is reported; XGBoost CUDA is optional but safe_cpu is the default for sklearn pipelines.")),
    }


def should_use_gpu(config=None) -> bool:
    """Return True when configuration requests GPU and a compatible device is visible."""
    requested = getattr(config, "use_gpu", "auto") if config is not None else "auto"
    return bool(gpu_status(requested).get("gpu_available", False))
