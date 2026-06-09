# Named Batches Are Canonical

Status: accepted

Task DataModules emit named batch objects as the canonical boundary between data loading and experiment execution. Regular Batch and Irregular Batch objects make task inputs explicit; positional tuple batches are Legacy Loader output only and should not be used by new Task Contracts or Model Contracts.
