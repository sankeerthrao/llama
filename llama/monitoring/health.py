"""Health checking for LLaMA inference serving.

Reports readiness, uptime, device information, and degraded/unhealthy
states based on error counts and model-load status.
"""

import platform
import time
from typing import Any, Dict, Optional

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


class HealthChecker:
    """Lightweight health probe for an inference service.

    Status logic:
    - **unhealthy** — model has not been loaded yet.
    - **degraded** — model is loaded but ``error_count > 10``.
    - **healthy** — model loaded and error count within tolerance.

    Example::

        health = HealthChecker()
        health.mark_model_loaded()
        health.record_inference()
        print(health.check())
    """

    def __init__(self) -> None:
        """Initialise health state."""
        self.start_time: float = time.time()
        self.model_loaded: bool = False
        self.last_inference_time: Optional[float] = None
        self.error_count: int = 0
        self.device_info: Dict[str, Any] = self._detect_device_info()

    # ------------------------------------------------------------------
    # State mutation helpers
    # ------------------------------------------------------------------

    def mark_model_loaded(self) -> None:
        """Signal that the model has been loaded and is ready for inference."""
        self.model_loaded = True

    def record_inference(self) -> None:
        """Record a successful inference (updates ``last_inference_time``)."""
        self.last_inference_time = time.time()

    def record_error(self) -> None:
        """Increment the running error counter."""
        self.error_count += 1

    # ------------------------------------------------------------------
    # Health probe
    # ------------------------------------------------------------------

    def check(self) -> Dict[str, Any]:
        """Return the current health status.

        Returns:
            A dict with keys: ``status``, ``uptime``, ``model_loaded``,
            ``last_inference_time``, ``error_count``, ``device_info``.
        """
        uptime = time.time() - self.start_time

        if not self.model_loaded:
            status = "unhealthy"
        elif self.error_count > 10:
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "uptime": round(uptime, 2),
            "model_loaded": self.model_loaded,
            "last_inference_time": self.last_inference_time,
            "error_count": self.error_count,
            "device_info": self.device_info,
        }

    # ------------------------------------------------------------------
    # Device detection (works without CUDA)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device_info() -> Dict[str, Any]:
        """Detect available compute device and memory.

        Works on CPU-only machines; enriches with CUDA info when
        ``torch.cuda`` is available.
        """
        info: Dict[str, Any] = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
        }

        # Determine device type
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            info["device"] = "cuda"
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem
            info["gpu_memory_total_bytes"] = mem
            try:
                free, total = torch.cuda.mem_get_info(0)
                info["gpu_memory_free_bytes"] = free
            except Exception:
                pass
        else:
            info["device"] = "cpu"

        # CPU memory via psutil (optional)
        if _PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            info["cpu_memory_total_bytes"] = vm.total
            info["cpu_memory_available_bytes"] = vm.available

        return info
