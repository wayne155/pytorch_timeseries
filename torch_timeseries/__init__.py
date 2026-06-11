# Some conda environments ship a newer libstdc++ than the system one; compiled
# extensions of downstream deps (e.g. matplotlib via torchmetrics) need it but
# resolve the old system library unless the env's copy is loaded first. Only
# inject it while no libstdc++ is mapped yet — loading a second copy into a
# process that already has one crashes.
try:
    import ctypes as _ctypes
    import os as _os
    import sys as _sys

    if _sys.platform == "linux":
        _libstdcxx = _os.path.join(_sys.prefix, "lib", "libstdc++.so.6")
        if _os.path.exists(_libstdcxx):
            with open("/proc/self/maps") as _maps:
                _already_loaded = "libstdc++" in _maps.read()
            if not _already_loaded:
                _ctypes.CDLL(_libstdcxx, mode=_ctypes.RTLD_GLOBAL)
    del _ctypes, _os, _sys
except OSError:
    pass

import torch_timeseries.dataset
import torch_timeseries.dataloader
import torch_timeseries.model
from .cli.exp import exp
from .results import RunResult, LocalBackend, WandbBackend
from .experiment import Experiment, register_model
