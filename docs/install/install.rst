Installation
============

``torch-timeseries`` requires Python ≥ 3.8 and a compatible PyTorch installation.

Install via pip
---------------

Install PyTorch for your CUDA version first (see `pytorch.org <https://pytorch.org/get-started/locally/>`_),
then install the package:

.. code-block:: bash

   pip install torch-timeseries

Optional dependencies for graph-based models:

.. code-block:: bash

   pip install torch_geometric torch_scatter

Development install
-------------------

.. code-block:: bash

   git clone https://github.com/wayne155/pytorch_timeseries
   cd pytorch_timeseries
   pip install -e ".[dev]"
