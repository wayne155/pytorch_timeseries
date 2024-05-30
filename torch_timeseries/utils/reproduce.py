import os
import random
import torch
import numpy as np

def reproducible(seed, dtype=torch.FloatTensor):
    # for reproducibility
    # torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type(dtype)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.determinstic = True
