# Separate Experiment Engine, Task Contracts, and Model Contracts

Status: accepted

Experiments are organized around three separate concepts: the Experiment Engine owns runtime behavior, a Task Contract owns task-specific data/loss/metric/evaluation semantics, and a Model Contract owns model construction and model hyperparameters. This replaces handwritten `(model × task)` experiment classes as the target architecture while preserving compatibility shims during migration.
