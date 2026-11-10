import logging
import os
from datetime import datetime
from multiprocessing import current_process
from .common import get_main_root


main_root = get_main_root()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# Los procesos auxiliares de CEEMDAN importan este módulo; solo el principal abre el archivo.
if current_process().name == "MainProcess":
    os.makedirs(os.path.join(main_root, "logs"), exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(main_root, "logs", f"{datetime.now():%Y%m%d_%H%M%S}.log"), encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
