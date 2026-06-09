from __future__ import annotations

from .loaders import load_all_entries
from .ranking import aggregate_and_rank
from .render import write_leaderboard_outputs


def leaderboard(
    results_dir: str = "./results",
    entries_dir: str = "leaderboard/entries",
    output_dir: str = "results/leaderboard",
    docs_dir: str = "docs/leaderboard",
):
    entries = load_all_entries(results_dir=results_dir, entries_dir=entries_dir)
    ranked = aggregate_and_rank(entries)
    write_leaderboard_outputs(ranked, output_dir=output_dir, docs_dir=docs_dir)
    return (
        f"Wrote leaderboard with {len(ranked)} rows to "
        f"{output_dir}/leaderboard.csv and {docs_dir}/index.md"
    )
