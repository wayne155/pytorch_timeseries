from dataclasses import asdict, fields


def asdict_exc(instance, exc_class):
    """Convert a dataclass instance to a dictionary, excluding Parent2 fields."""
    parent2_fields = {f.name for f in fields(exc_class)}
    return {k: v for k, v in asdict(instance).items() if k not in parent2_fields}
