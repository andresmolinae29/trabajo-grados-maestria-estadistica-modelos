import logging
import os
from datetime import datetime
from .common import get_main_root


main_root = get_main_root()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(
    os.path.join(main_root, "logs", f'{datetime.now().strftime("%Y%m%d")}.log')
)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
