# Results and Benchmark Reports Are Separate Boundaries

Status: accepted

The Experiment Engine emits Result Records, and Result Backends store those records. Benchmark Reports read stored Result Records and curated reference entries; task and model code should not know about leaderboard rendering or specific tracking tools such as W&B.
