from typing import Optional

import numpy as np
from .dataset import TimeSeriesDataset

class AnomalyDataset(TimeSeriesDataset):
    train_data : Optional[np.array]
    test_data : Optional[np.array]
    test_labels : Optional[np.array]
        