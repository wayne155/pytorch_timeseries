# Preserve Public Experiment Names With Compatibility Shims

Status: accepted

Existing public experiment names remain importable during the architecture migration, but they should delegate to the canonical Experiment Engine, Task Contracts, and Model Contracts. New features should target the canonical architecture; Compatibility Shims exist only to prevent immediate breakage and can be removed in a future release.
