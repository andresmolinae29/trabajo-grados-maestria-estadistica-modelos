import os
from functools import lru_cache


@lru_cache
def get_main_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
