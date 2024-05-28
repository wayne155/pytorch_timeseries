import numpy as np
from .scaler import Scaler
from torch_timeseries.dataset import TimeSeriesDataset, AnomalyDataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset


class AnomalySliced(TimeSeriesDataset):
    def __init__(
        self,
        dataset: AnomalyDataset,
        scaler: Scaler,
        train_ratio: float = 0.8,
        step=1,
        window: int = 168,
        scaler_fit=True,
        flag="train",
    ):
        self.dataset = dataset
        self.window = window
        self.step = step
        self.scaler = scaler
        self.flag = flag
        self.train_ratio = train_ratio

        _len = self.dataset.train_data.shape[0]

        self.train_len = int(_len * self.train_ratio)
        if flag == "train":
            self.train_data = self.dataset.train_data[: self.train_len]
            if scaler_fit:
                self.scaler.fit(self.train_data)
            self.scaled_train_data = self.scaler.transform(self.train_data)
        elif flag == "val":
            self.val_data = self.dataset.train_data[self.train_len :]
            self.scaled_val_data = self.scaler.transform(self.val_data)
        elif flag == "test":
            self.test_data = self.dataset.test_data
            self.test_labels = self.dataset.test_labels
            self.scaled_test_data = self.scaler.transform(self.test_data)

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(
                self.scaled_train_data[index : index + self.window]
            ), np.float32(self.train_data[index : index + self.window])
        elif self.flag == "val":
            return (
                np.float32(self.scaled_val_data[index : index + self.window]),
                np.float32(self.val_data[index : index + self.window]),
            )
        elif self.flag == "test":
            return (
                np.float32(self.scaled_test_data[index : index + self.window]),
                np.float32(self.test_data[index : index + self.window]),
                np.float32(self.test_labels[index : index + self.window]),
            )
        else:
            NotImplementedError("not implemented")

    def __len__(self):
        if self.flag == "train":
            return (self.train_data.shape[0] - self.window) // self.step + 1
        elif self.flag == "val":
            return (self.val_data.shape[0] - self.window) // self.step + 1
        elif self.flag == "test":
            return (self.test_data.shape[0] - self.window) // self.step + 1
        else:
            raise NotImplementedError("Not implemented!!!")


class AnomalyLoader:
    def __init__(
        self,
        dataset: TimeSeriesDataset,
        scaler: Scaler,
        window: int = 168,
        shuffle_train=True,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        num_worker: int = 3,
    ) -> None:

        self.train_ratio = train_ratio
        self.val_ratio = 1 - self.train_ratio
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.dataset = dataset

        self.scaler = scaler
        self.window = window
        self.shuffle_train = shuffle_train

        self._load()

    def _load(self):
        self._load_dataset()
        self._load_dataloader()

    def _load_dataset(self):
        self.train_dataset = AnomalySliced(
            self.dataset,
            self.scaler,
            self.train_ratio,
            self.window,
            scaler_fit=True,
            flag="train",
        )
        self.val_dataset = AnomalySliced(
            self.dataset,
            self.scaler,
            self.train_ratio,
            self.window,
            scaler_fit=False,
            flag="val",
        )
        self.test_dataset = AnomalySliced(
            self.dataset,
            self.scaler,
            self.train_ratio,
            self.window,
            scaler_fit=False,
            flag="test",
        )

    def _load_dataloader(self):
        self.train_size = len(self.train_dataset)
        self.val_size = len(self.val_dataset)
        self.test_size = len(self.test_dataset)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_worker,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
        )
