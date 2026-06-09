# v2 Task DataModules Are Canonical

Status: accepted

v2 Task DataModules are the canonical data-loading architecture for new code because they return explicit batch objects and centralize task-level split, scaling, and loader behavior. Legacy Loaders remain importable for compatibility, but they should not receive new features and are expected to be removed in a future release.
