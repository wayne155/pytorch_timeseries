# Experiment Configuration Is Strictly Validated

Status: accepted

Experiment configuration is strict by default: unknown or irrelevant settings fail before data or model construction. This prevents silent benchmark corruption when a user passes a hyperparameter that the selected task or model does not actually consume.
