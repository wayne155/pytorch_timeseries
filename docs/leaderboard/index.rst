Leaderboard
===========

The leaderboard is documentation-first and machine-readable. It reads local
``RunResult`` JSON files plus curated reference entries, ranks methods by
task-specific metrics, and writes Markdown, CSV, JSON, and webapp data.

Inputs
------

Local experiment results are stored under ``results/records`` when experiments
run with a local backend:

.. code-block:: text

   results/records/{model}/{task}/{dataset}/{config_hash}/seed{seed}.json

Curated external entries live in YAML files:

.. code-block:: text

   leaderboard/entries/*.yaml

Each curated entry should include source metadata so external numbers remain
auditable.

Generate Static Outputs
-----------------------

Use the CLI command:

.. code-block:: bash

   pytexp leaderboard \
     --results_dir ./results \
     --entries_dir leaderboard/entries \
     --output_dir results/leaderboard \
     --docs_dir docs/leaderboard

The generated files include:

.. code-block:: text

   docs/leaderboard/*.md
   results/leaderboard/leaderboard.csv
   results/leaderboard/leaderboard.json

Web Leaderboard
---------------

The deployable web leaderboard reads generated JSON data. The usual flow is:

.. code-block:: bash

   python leaderboard/build_leaderboard.py
   cd webapp
   npm install
   npm run build

For local development, use the leaderboard server script if available:

.. code-block:: bash

   python leaderboard/serve_leaderboard.py

Ranking Metrics
---------------

Default primary metrics are:

- ``Forecast``: ``mse`` lower is better;
- ``Imputation``: ``mse`` lower is better;
- ``AnomalyDetection``: ``F-score`` higher is better;
- ``UEAClassification``: ``accuracy`` or ``MulticlassAccuracy`` higher is
  better.

Configuration-Aware Rows
------------------------

Rows are grouped by model, task, dataset, and run configuration. This means the
leaderboard is not limited to a fixed set of hand-picked settings. Any stored
configuration can be displayed, filtered, and aggregated as long as its
``RunResult`` records are available.
