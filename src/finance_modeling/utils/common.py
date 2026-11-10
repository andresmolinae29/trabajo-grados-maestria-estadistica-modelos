import os
from functools import lru_cache


@lru_cache
def get_main_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def validate_file_exists(file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")


def create_experiment_directory(output_dir: str, experiment_name: str) -> str:

    path = os.path.join(
        output_dir,
        experiment_name
    )
    os.makedirs(path, exist_ok=True)
    return path
