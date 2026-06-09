import os
import random
import torch
import numpy as np


def reproducible(seed, dtype=torch.float32):
    """Set seeds and configure cuDNN for reproducible training.

    Note: ``torch.set_default_tensor_type`` is deprecated in PyTorch >= 2.3,
    so we use ``torch.set_default_dtype`` instead. Pass a torch dtype
    (e.g. ``torch.float32``) rather than a legacy tensor type class.
    """
    if isinstance(dtype, torch.dtype):
        torch.set_default_dtype(dtype)
    else:
        # Backward compatibility: accept legacy tensor type classes
        # like ``torch.FloatTensor`` by mapping to their dtype.
        legacy_map = {
            torch.FloatTensor: torch.float32,
            torch.DoubleTensor: torch.float64,
            torch.HalfTensor: torch.float16,
        }
        torch.set_default_dtype(legacy_map.get(dtype, torch.float32))

    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_rng_state():
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    if not state:
        return

    torch_state = state.get("torch")
    if torch_state is not None:
        if isinstance(torch_state, torch.Tensor):
            torch_state = torch_state.cpu()
        torch.set_rng_state(torch_state)

    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    random_state = state.get("random")
    if random_state is not None:
        random.setstate(random_state)

    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
