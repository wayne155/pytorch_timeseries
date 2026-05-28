# torch_timeseries/dataset/irregular/physionet2019.py
from __future__ import annotations
import urllib.request
import zipfile
from pathlib import Path
import numpy as np

from .base import IrregularTimeSeriesDataset


class PhysioNet2019(IrregularTimeSeriesDataset):
    """PhysioNet Challenge 2019 — sepsis onset prediction (binary classification).

    ~40,000 ICU patient records, 40 clinical variables, hourly observations.
    Label: SepsisLabel (0/1) — 1 if sepsis onset occurred during the stay.

    Auto-downloads from PhysioNet Challenge 2019 public page if not present.
    """

    # 40 clinical variables (SepsisLabel is the 41st column — excluded from features)
    VARIABLES = [
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
        "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
        "Fibrinogen", "Platelets",
        "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
    ]
    _VAR_IDX = {v: i for i, v in enumerate(VARIABLES)}

    _BASE_URL = "https://physionet.org/files/challenge-2019/1.0.0/"
    _SETS = ["training_setA", "training_setB"]

    def download(self) -> None:
        dest = Path(self.root) / "physionet2019"
        if all((dest / s).exists() for s in self._SETS):
            return
        dest.mkdir(parents=True, exist_ok=True)
        for s in self._SETS:
            zip_url = f"{self._BASE_URL}{s}.zip"
            zip_path = dest / f"{s}.zip"
            print(f"Downloading {zip_url} ...")
            urllib.request.urlretrieve(zip_url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest)
            zip_path.unlink()

    def _load(self) -> None:
        dest = Path(self.root) / "physionet2019"
        all_samples, all_times, all_masks, all_labels = [], [], [], []
        F = len(self.VARIABLES)

        for s in self._SETS:
            set_dir = dest / s
            if not set_dir.exists():
                continue
            for psv_file in sorted(set_dir.glob("*.psv")):
                try:
                    x, t_arr, mask, label = self._parse_patient(psv_file, F)
                except Exception as exc:
                    print(f"Warning: skipping {psv_file.name}: {exc}")
                    continue
                all_samples.append(x)
                all_times.append(t_arr)
                all_masks.append(mask)
                all_labels.append(label)

        self.samples = all_samples
        self.times = all_times
        self.masks = all_masks
        self.labels = np.array(all_labels, dtype=np.int64)
        self.num_features = F
        self.num_classes = 2

    def _parse_patient(self, path: Path, F: int):
        """Parse one PSV patient file → (x, t_arr, mask, label).

        label is 1 if SepsisLabel == 1 in any row (patient ever developed sepsis).
        """
        with open(path) as f:
            header = f.readline().strip().split("|")
            rows = [line.strip().split("|") for line in f if line.strip()]

        col_idx = {name: i for i, name in enumerate(header)}
        iculos_col = col_idx.get("ICULOS", -1)
        sepsis_col = col_idx.get("SepsisLabel", -1)

        T = len(rows)
        x = np.zeros((T, F), dtype=np.float32)
        mask = np.zeros((T, F), dtype=np.float32)
        t_arr = np.zeros(T, dtype=np.float32)
        label = 0

        for t_idx, row in enumerate(rows):
            if iculos_col >= 0:
                try:
                    t_arr[t_idx] = float(row[iculos_col])
                except (ValueError, IndexError):
                    t_arr[t_idx] = float(t_idx)
            else:
                t_arr[t_idx] = float(t_idx)

            if sepsis_col >= 0:
                try:
                    lbl = int(float(row[sepsis_col]))
                    if lbl == 1:
                        label = 1
                except (ValueError, IndexError):
                    pass

            for var_name, var_i in self._VAR_IDX.items():
                col = col_idx.get(var_name, -1)
                if col < 0 or col >= len(row):
                    continue
                val_str = row[col]
                if val_str not in ("NaN", "nan", "", "NA"):
                    try:
                        x[t_idx, var_i] = float(val_str)
                        mask[t_idx, var_i] = 1.0
                    except ValueError:
                        pass

        return x, t_arr, mask, label
