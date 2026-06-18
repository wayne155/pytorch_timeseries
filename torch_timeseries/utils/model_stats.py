from __future__ import annotations

from typing import Dict

from prettytable import PrettyTable
import torch


def count_parameters(model: torch.nn.Module):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    return table, total_params


def model_summary(model: torch.nn.Module) -> Dict[str, object]:
    """Return a compact summary of a model's parameter counts and size.

    Args:
        model: Any ``nn.Module``.

    Returns:
        dict with keys:

        - ``total_params`` (int): all parameters
        - ``trainable_params`` (int): requires-grad parameters
        - ``non_trainable_params`` (int): frozen parameters
        - ``size_mb`` (float): total parameter storage in megabytes (float32)
        - ``param_table`` (PrettyTable): per-module breakdown (trainable only)

    Example::

        from torch_timeseries.utils import model_summary
        info = model_summary(model)
        print(f"Trainable: {info['trainable_params']:,}  ({info['size_mb']:.2f} MB)")
    """
    table = PrettyTable(["Module", "Shape", "Parameters", "Trainable"])
    total = 0
    trainable = 0
    non_trainable = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
        else:
            non_trainable += n
        table.add_row([name, list(param.shape), n, param.requires_grad])
    # estimate size assuming float32 (4 bytes)
    size_mb = total * 4 / 1024 / 1024
    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "size_mb": size_mb,
        "param_table": table,
    }
