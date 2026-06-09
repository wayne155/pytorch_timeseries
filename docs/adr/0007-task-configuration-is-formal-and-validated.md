# Task Configuration Is Formal and Validated

Status: accepted

Task-shape settings such as windows, prediction length, horizon, column selection, mask ratio, and task-specific controls are formal Task Configuration, not scattered keyword arguments. The public Experiment API may remain flat and ergonomic, but internally each task builds and validates its Task Configuration before constructing the Task DataModule or model.
