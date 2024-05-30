
from typing import Type, Union


def parse_type(str_or_type: Union[Type, str], globals) -> Type:
    if isinstance(str_or_type, str):
        return eval(str_or_type, globals)
    elif isinstance(str_or_type, type):
        return str_or_type
    else:
        raise RuntimeError(f"{str_or_type} should be string or type")
