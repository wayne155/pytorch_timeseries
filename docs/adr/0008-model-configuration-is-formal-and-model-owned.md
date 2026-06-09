# Model Configuration Is Formal and Model-Owned

Status: accepted

Model-specific architecture settings are formal Model Configuration and are validated by the Model Contract that owns them. The public Experiment API can accept flat keyword arguments, but internally task-owned, model-owned, runtime, and result settings should be separated before constructing data, models, or run state.
