# src/logger/logger.py

import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir='logs'):
        # Generate timestamped log file name
        timestamp = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
        log_file = f"{timestamp}.log"

        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)

        # Create logger
        self.logger = logging.getLogger("CustomerLTV")
        self.logger.setLevel(logging.INFO)

        # Prevent adding handlers multiple times
        if not self.logger.handlers:
            # File handler
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.INFO)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Formatter
            formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add handlers
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
