from typing import Optional, Tuple


def resolve_split_ratios(
    train_ratio: float,
    test_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Return validated train/val/test ratios.

    If ``val_ratio`` is provided, ``test_ratio`` is derived for backward
    compatibility with constructors whose ``test_ratio`` has a default value.
    """
    train_ratio = float(train_ratio)
    if val_ratio is not None:
        val_ratio = float(val_ratio)
        test_ratio = 1.0 - train_ratio - val_ratio
    elif test_ratio is not None:
        test_ratio = float(test_ratio)
        val_ratio = 1.0 - train_ratio - test_ratio
    else:
        raise ValueError("Either val_ratio or test_ratio must be provided.")

    ratios = tuple(0.0 if abs(ratio) < 1e-12 else ratio for ratio in (train_ratio, val_ratio, test_ratio))
    train_ratio, val_ratio, test_ratio = ratios
    if any(ratio < 0 or ratio > 1 for ratio in ratios):
        raise ValueError(
            "Split ratios must each be between 0 and 1 and sum to 1.0; "
            f"got train={train_ratio}, val={val_ratio}, test={test_ratio}."
        )
    if abs(sum(ratios) - 1.0) >= 1e-6:
        raise ValueError(
            "Split ratios must sum to 1.0; "
            f"got train={train_ratio}, val={val_ratio}, test={test_ratio}."
        )
    return train_ratio, val_ratio, test_ratio
