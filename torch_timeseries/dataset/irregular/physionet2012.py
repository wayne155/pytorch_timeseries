# torch_timeseries/dataset/irregular/physionet2012.py
from __future__ import annotations
import urllib.request
import tarfile
from pathlib import Path
import numpy as np

from .base import IrregularTimeSeriesDataset


class PhysioNet2012(IrregularTimeSeriesDataset):
    """PhysioNet Challenge 2012 — in-hospital mortality (binary classification).

    12,000 ICU patient records from set-a/b/c. 41 time-varying variables;
    up to 48 hours at irregular intervals. Label: In-hospital_death (0/1).

    Auto-downloads from https://physionet.org/files/challenge-2012/1.0.0/
    if data is not already present at {root}/physionet2012/.
    """

    VARIABLES = [
        "Age", "Gender", "Height", "ICUType",
        "ALP", "ALT", "AST", "Albumin", "BUN",
        "Bicarbonate", "Bilirubin", "Cholesterol", "Creatinine",
        "DiasABP", "FiO2", "GCS", "Glucose", "HCT", "HR",
        "K", "Lactate", "MAP", "MechVent", "Mg", "Na",
        "NIDiasABP", "NIMAP", "NISysABP", "PaCO2", "PaO2",
        "Platelets", "RespRate", "SaO2", "SysABP", "Temp",
        "TroponinI", "TroponinT", "Urine", "WBC", "Weight", "pH",
    ]
    _VAR_IDX = {v: i for i, v in enumerate(VARIABLES)}

    _BASE_URL = "https://physionet.org/files/challenge-2012/1.0.0/"
    _SETS = ["a", "b", "c"]

    def download(self) -> None:
        dest = Path(self.root) / "physionet2012"
        if (dest / "set-a").exists():
            return
        dest.mkdir(parents=True, exist_ok=True)
        for s in self._SETS:
            tar_url = f"{self._BASE_URL}set-{s}.tar.gz"
            tar_path = dest / f"set-{s}.tar.gz"
            print(f"Downloading {tar_url} ...")
            urllib.request.urlretrieve(tar_url, tar_path)
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(dest)
            tar_path.unlink()
            outcome_url = f"{self._BASE_URL}Outcomes-{s}.txt"
            urllib.request.urlretrieve(outcome_url, dest / f"Outcomes-{s}.txt")

    def _load(self) -> None:
        dest = Path(self.root) / "physionet2012"
        all_samples, all_times, all_masks, all_labels = [], [], [], []
        F = len(self.VARIABLES)

        for s in self._SETS:
            set_dir = dest / f"set-{s}"
            outcome_path = dest / f"Outcomes-{s}.txt"
            if not set_dir.exists() or not outcome_path.exists():
                continue

            outcomes = {}
            with open(outcome_path) as f:
                next(f)  # skip header
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 6:
                        try:
                            outcomes[int(parts[0])] = int(parts[5])
                        except ValueError:
                            pass

            for txt_file in sorted(set_dir.glob("*.txt")):
                try:
                    rec_id, x, t_arr, mask = self._parse_patient(txt_file, F)
                except Exception:
                    continue
                if rec_id not in outcomes:
                    continue
                all_samples.append(x)
                all_times.append(t_arr)
                all_masks.append(mask)
                all_labels.append(outcomes[rec_id])

        self.samples = all_samples
        self.times = all_times
        self.masks = all_masks
        self.labels = np.array(all_labels, dtype=np.int64)
        self.num_features = F
        self.num_classes = 2

    def _parse_patient(self, path: Path, F: int):
        """Parse one PhysioNet 2012 patient file → (rec_id, x, t_arr, mask)."""
        header = {}
        obs = {}  # minute → {var_idx: value}

        in_header = True
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    in_header = False
                    continue
                if in_header:
                    k, v = line.split(",", 1)
                    header[k] = v
                else:
                    if line.startswith("Time,"):
                        continue
                    parts = line.split(",")
                    if len(parts) != 3:
                        continue
                    hhmm, param, val_str = parts
                    if param not in self._VAR_IDX:
                        continue
                    try:
                        hh, mm = hhmm.split(":")
                        minutes = int(hh) * 60 + int(mm)
                        val = float(val_str)
                    except ValueError:
                        continue
                    if minutes not in obs:
                        obs[minutes] = {}
                    obs[minutes][self._VAR_IDX[param]] = val

        rec_id = int(header.get("RecordID", -1))

        # Static header variables at t=0
        for svar in ["Age", "Gender", "Height", "ICUType"]:
            if svar in header:
                try:
                    val = float(header[svar])
                    if val >= 0:  # -1 means missing in PhysioNet 2012
                        if 0 not in obs:
                            obs[0] = {}
                        obs[0][self._VAR_IDX[svar]] = val
                except ValueError:
                    pass

        if not obs:
            obs[0] = {}

        sorted_times = sorted(obs.keys())
        T = len(sorted_times)
        x = np.zeros((T, F), dtype=np.float32)
        mask = np.zeros((T, F), dtype=np.float32)
        t_arr = np.array(sorted_times, dtype=np.float32)

        for i, minute in enumerate(sorted_times):
            for var_idx, val in obs[minute].items():
                x[i, var_idx] = val
                mask[i, var_idx] = 1.0

        return rec_id, x, t_arr, mask
