"""Device utilities for CPU/CUDA mode selection."""

import os
import torch

_CPU_MODE = os.environ.get("TDATR_CPU_MODE", "0") == "1"


def use_cpu_mode() -> bool:
    return _CPU_MODE


def get_device() -> torch.device:
    if _CPU_MODE:
        return torch.device("cpu")
    return torch.device("cuda")


def current_device() -> torch.device:
    """Replacement for torch.cuda.current_device() that works on CPU."""
    if _CPU_MODE:
        return torch.device("cpu")
    return torch.device(torch.cuda.current_device())


def empty_on_device(*args, **kwargs):
    """Like torch.empty but places on the correct device."""
    if "device" in kwargs:
        return torch.empty(*args, **kwargs)
    return torch.empty(*args, device=current_device(), **kwargs)


def zeros_on_device(*args, **kwargs):
    if "device" in kwargs:
        return torch.zeros(*args, **kwargs)
    return torch.zeros(*args, device=current_device(), **kwargs)


def tensor_on_device(*args, **kwargs):
    if "device" in kwargs:
        return torch.tensor(*args, **kwargs)
    return torch.tensor(*args, device=current_device(), **kwargs)
