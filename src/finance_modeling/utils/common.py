import os
from functools import lru_cache


@lru_cache
def get_main_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def validate_file_exists(file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")