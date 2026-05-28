# tests/dataset/test_irregular_datasets.py
import numpy as np
import pytest


def _write_fake_physionet2012(tmp_path, record_ids, outcomes):
    """Create minimal PhysioNet 2012 set-a directory."""
    set_a = tmp_path / "physionet2012" / "set-a"
    set_a.mkdir(parents=True)
    for rec_id in record_ids:
        lines = [
            f"RecordID,{rec_id}",
            "Age,50", "Gender,1", "Height,170", "ICUType,1", "Weight,70",
            "",
            "Time,Parameter,Value",
            "00:07,HR,109",
            "00:07,GCS,15",
            "01:35,HR,122",
            "02:00,Temp,37.2",
        ]
        (set_a / f"{rec_id}.txt").write_text("\n".join(lines))
    outcome_lines = [
        "RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death"
    ]
    for rec_id, label in zip(record_ids, outcomes):
        outcome_lines.append(f"{rec_id},6,1,5,3950.0,{label}")
    (tmp_path / "physionet2012" / "Outcomes-a.txt").write_text("\n".join(outcome_lines))


def test_physionet2012_loads(tmp_path):
    _write_fake_physionet2012(tmp_path, [140501, 140936, 141091], [0, 1, 0])
    from torch_timeseries.dataset.irregular import PhysioNet2012
    ds = PhysioNet2012(root=str(tmp_path), download=False)

    assert len(ds) == 3
    assert ds.num_features == 41
    assert ds.num_classes == 2
    assert ds.labels is not None
    assert len(ds.labels) == 3
    assert ds.labels.tolist() == [0, 1, 0]
    # each sample is (T_i, 41)
    assert ds.samples[0].ndim == 2
    assert ds.samples[0].shape[1] == 41
    # times: (T_i,)
    assert ds.times[0].ndim == 1
    assert len(ds.times[0]) == ds.samples[0].shape[0]
    # masks: (T_i, 41), binary
    assert ds.masks[0].shape == ds.samples[0].shape
    assert set(np.unique(ds.masks[0])).issubset({0, 1})
    # HR is observed → at least one mask entry is 1
    assert ds.masks[0].sum() > 0


def _write_fake_physionet2019(tmp_path, patient_ids, labels):
    """Create minimal PhysioNet 2019 set-A directory."""
    set_a = tmp_path / "physionet2019" / "training_setA"
    set_a.mkdir(parents=True)
    P19_HEADER = (
        "HR|O2Sat|Temp|SBP|MAP|DBP|Resp|EtCO2|BaseExcess|HCO3|FiO2|pH|"
        "PaCO2|SaO2|AST|BUN|Alkalinephos|Calcium|Chloride|Creatinine|"
        "Bilirubin_direct|Glucose|Lactate|Magnesium|Phosphate|Potassium|"
        "Bilirubin_total|TroponinI|Hct|Hgb|PTT|WBC|Fibrinogen|Platelets|"
        "Age|Gender|Unit1|Unit2|HospAdmTime|ICULOS|SepsisLabel"
    )
    for pid, label in zip(patient_ids, labels):
        lines = [P19_HEADER]
        lines.append("80|98|37.0|120|80|60|16|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|40|NaN|NaN|NaN|NaN|NaN|65|0|0|0|-5|1|0")
        lines.append(f"85|97|37.2|118|78|58|18|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|40|NaN|NaN|NaN|NaN|NaN|65|0|0|0|-5|2|{label}")
        (set_a / f"p{pid:06d}.psv").write_text("\n".join(lines))


def test_physionet2019_loads(tmp_path):
    _write_fake_physionet2019(tmp_path, [1, 2, 3], [0, 1, 0])
    from torch_timeseries.dataset.irregular import PhysioNet2019
    ds = PhysioNet2019(root=str(tmp_path), download=False)

    assert len(ds) == 3
    assert ds.num_features == 40
    assert ds.num_classes == 2
    assert ds.labels is not None
    assert len(ds.labels) == 3
    assert ds.labels.tolist() == [0, 1, 0]
    assert ds.samples[0].ndim == 2
    assert ds.samples[0].shape[1] == 40
    assert ds.times[0].ndim == 1
    assert ds.masks[0].shape == ds.samples[0].shape
    assert set(np.unique(ds.masks[0])).issubset({0, 1})
