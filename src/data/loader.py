# src/data/loader.py

import pandas as pd
import os
from src.utils.logger import Logger

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.logger = Logger().get_logger()

    def load_data(self):
        try:
            self.logger.info(f"Attempting to load data from: {self.data_path}")

            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"File not found: {self.data_path}")

            if self.data_path.endswith(".csv"):
                data = pd.read_csv(self.data_path, encoding='ISO-8859-1')
            elif self.data_path.endswith(".xlsx") or self.data_path.endswith(".xls"):
                data = pd.read_excel(self.data_path)
            else:
                raise ValueError("Unsupported file format. Only .csv and .xlsx are allowed.")

            self.logger.info(f"Data loaded successfully. Shape: {data.shape}")
            return data

        except Exception as e:
            self.logger.error(f"Error while loading data: {str(e)}")
            raise e
