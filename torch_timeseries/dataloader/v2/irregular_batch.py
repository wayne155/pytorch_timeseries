"""Structured batch type for irregular time-series tasks.

Unlike ``TSBatch``, which assumes uniform observation grids, ``IrregularTSBatch``
carries explicit observation timestamps and a per-feature mask to handle:
  - Non-uniform sampling rates
  - Per-feature missingness (e.g. sensor drop-out)
  - Variable-length sequences within a mini-batch (padded by ``collate_irregular``)

Padding conventions (applied by ``collate_irregular``)
------------------------------------------------------
* ``x``         : padded with **0**
* ``mask``      : padded with **0**  (marks padded positions as unobserved)
* ``t``         : padded with **1.0** (outside the normalized [0, 1] range)
* ``t_query``   : padded with **1.0**
* ``query_mask``: padded with **0**
* ``y``         : padded with **0** when it is a variable-length target tensor;
                  stacked directly when it is a scalar label
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor


@dataclass
class IrregularTSBatch:
    x: Tensor                               # (T, F) per-sample  OR  (B, T, F) batched
    t: Tensor                               # (T,)   elapsed time, normalized [0, 1]
    mask: Tensor                            # (T, F) 1=observed, 0=missing/padded
    x_time: Optional[Tensor] = None        # (T, C)  calendar features at obs times
    y: Optional[Tensor] = None             # scalar class label  OR  (Tq, F) query targets
    t_query: Optional[Tensor] = None       # (Tq,)  query times (interp / forecast)
    query_mask: Optional[Tensor] = None    # (Tq, F) which queries to evaluate
    t_query_time: Optional[Tensor] = None  # (Tq, C) calendar features at query times


def collate_irregular(samples: List[IrregularTSBatch]) -> IrregularTSBatch:
    """Pad variable-length sequences to the longest sequence in the batch.

    All per-sample ``IrregularTSBatch`` objects are assumed to carry 2-D ``x``
    tensors of shape ``(T_i, F)``.  The returned batch has 3-D tensors with
    leading batch dimension ``B``.

    Padding conventions:
      - ``x``, ``x_time``, ``y`` (when tensor), ``t_query_time``: padded with 0
      - ``mask``, ``query_mask``:  padded with 0  (marks as unobserved)
      - ``t``, ``t_query``:        padded with 1.0 (sentinel outside [0, 1])
    """
    max_T = max(s.x.shape[0] for s in samples)
    F = samples[0].x.shape[1]

    # ------------------------------------------------------------------
    # Core fields: x, t, mask
    # ------------------------------------------------------------------
    xs, ts, masks = [], [], []
    for s in samples:
        T_i = s.x.shape[0]

        x_pad = torch.zeros(max_T, F)
        x_pad[:T_i] = s.x
        xs.append(x_pad)

        t_pad = torch.ones(max_T)       # 1.0 sentinel for padded positions
        t_pad[:T_i] = s.t
        ts.append(t_pad)

        m_pad = torch.zeros(max_T, F)  # 0 = unobserved / padded
        m_pad[:T_i] = s.mask
        masks.append(m_pad)

    x_batch = torch.stack(xs, dim=0)
    t_batch = torch.stack(ts, dim=0)
    mask_batch = torch.stack(masks, dim=0)

    # ------------------------------------------------------------------
    # x_time  (calendar features at observation times)
    # ------------------------------------------------------------------
    x_time_batch: Optional[Tensor] = None
    if samples[0].x_time is not None:
        C = samples[0].x_time.shape[1]
        x_times = []
        for s in samples:
            T_i = s.x_time.shape[0]
            xt = torch.zeros(max_T, C)
            xt[:T_i] = s.x_time
            x_times.append(xt)
        x_time_batch = torch.stack(x_times, dim=0)

    # ------------------------------------------------------------------
    # y  (scalar classification label  OR  variable-length target tensor)
    # ------------------------------------------------------------------
    y_batch: Optional[Tensor] = None
    if samples[0].y is not None:
        if samples[0].y.dim() == 0:
            # scalar labels → simple stack → (B,)
            y_batch = torch.stack([s.y for s in samples], dim=0)
        else:
            # variable-length query targets → pad like x → (B, max_Tq_y, Fy)
            max_Tq_y = max(s.y.shape[0] for s in samples)
            Fy = samples[0].y.shape[1]
            ys = []
            for s in samples:
                Tq_i = s.y.shape[0]
                yp = torch.zeros(max_Tq_y, Fy)
                yp[:Tq_i] = s.y
                ys.append(yp)
            y_batch = torch.stack(ys, dim=0)

    # ------------------------------------------------------------------
    # Query times  (interpolation / forecast targets)
    # ------------------------------------------------------------------
    t_query_batch: Optional[Tensor] = None
    query_mask_batch: Optional[Tensor] = None
    t_query_time_batch: Optional[Tensor] = None

    if samples[0].t_query is not None:
        max_Tq = max(s.t_query.shape[0] for s in samples)
        Fq = samples[0].query_mask.shape[1] if samples[0].query_mask is not None else F

        tqs: List[Tensor] = []
        qms: List[Tensor] = []
        for s in samples:
            Tq_i = s.t_query.shape[0]

            tq = torch.ones(max_Tq)    # 1.0 sentinel for padded query times
            tq[:Tq_i] = s.t_query
            tqs.append(tq)

            if s.query_mask is not None:
                qm = torch.zeros(max_Tq, Fq)
                qm[:Tq_i] = s.query_mask
                qms.append(qm)

        t_query_batch = torch.stack(tqs, dim=0)
        if qms:
            query_mask_batch = torch.stack(qms, dim=0)

        if samples[0].t_query_time is not None:
            C2 = samples[0].t_query_time.shape[1]
            tqts: List[Tensor] = []
            for s in samples:
                Tq_i = s.t_query_time.shape[0]
                tqt = torch.zeros(max_Tq, C2)
                tqt[:Tq_i] = s.t_query_time
                tqts.append(tqt)
            t_query_time_batch = torch.stack(tqts, dim=0)

    return IrregularTSBatch(
        x=x_batch,
        t=t_batch,
        mask=mask_batch,
        x_time=x_time_batch,
        y=y_batch,
        t_query=t_query_batch,
        query_mask=query_mask_batch,
        t_query_time=t_query_time_batch,
    )
