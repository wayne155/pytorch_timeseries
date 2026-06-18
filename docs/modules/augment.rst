torch_timeseries.augment
========================

Composable, stochastic augmentations for time series.  All transforms are
pure-function callables that accept ``(B, T, C)`` or ``(T, C)`` float
tensors and return a tensor of the same shape.  Chain them with
:class:`Compose` and apply inside a training loop or dataset
``__getitem__``.

.. code-block:: python

   from torch_timeseries.augment import Compose, Jitter, Scale, MagnitudeWarp

   aug = Compose([Jitter(sigma=0.03), Scale(sigma=0.1), MagnitudeWarp(sigma=0.2)])
   x_aug = aug(x)   # x: (B, T, C)

----

Transforms
----------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Class
     - Description
   * - ``Compose``
     - Apply a list of transforms sequentially.  Useful for building a
       single augmentation pipeline from primitives.
   * - ``Jitter``
     - Add independent Gaussian noise *N(0, σ²)* to every element.
       Controlled by ``sigma`` (default 0.05).
   * - ``Scale``
     - Multiply each sample by a per-channel random scalar drawn from
       *N(1, σ²)*. Controlled by ``sigma`` (default 0.1).
   * - ``MagnitudeWarp``
     - Warp amplitudes by a smooth random curve (interpolated from
       ``n_knots`` Gaussian-perturbed control points). Controlled by
       ``sigma`` and ``n_knots``.
   * - ``TimeWarp``
     - Stretch and compress local time regions by resampling along a
       smooth random index mapping.  Controlled by ``sigma`` and
       ``n_knots``.
   * - ``WindowSlice``
     - Extract a contiguous sub-window (length = ``crop_ratio × T``) and
       interpolate it back to the original length. Controlled by
       ``crop_ratio`` (default 0.9).
   * - ``Permute``
     - Divide the sequence into ``n_segments`` equal-length blocks and
       shuffle them randomly.  Controlled by ``n_segments`` (default 4).
   * - ``Flip``
     - Reverse the time axis.  Deterministic — no parameters.
   * - ``RandomMask``
     - Zero-out each time step independently with probability ``p``
       (masking is broadcast across channels). Controlled by ``p``
       (default 0.1).

.. currentmodule:: torch_timeseries.augment

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   Compose
   Jitter
   Scale
   MagnitudeWarp
   TimeWarp
   WindowSlice
   Permute
   Flip
   RandomMask

----

References
----------

* Um et al., *Data Augmentation of Wearable Sensor Data for Parkinson's
  Disease Monitoring using Convolutional Neural Networks*, 2017.
* Iwana & Uchida, *An empirical survey of data augmentation for time
  series classification with neural networks*, PLOS ONE 2021.
