# src/utils/logger.py

import logging
import os

LOG_FILE = os.path.join("logs", "app.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app_logger = logging.getLogger()
