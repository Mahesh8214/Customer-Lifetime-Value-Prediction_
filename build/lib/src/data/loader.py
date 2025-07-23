import pandas as pd
import os
from src.data import app_logger
import logging

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_data(self) -> pd.DataFrame:
        try:
            app_logger.info(f"Loading data from {self.filepath}")
            df = pd.read_excel(self.filepath)
            app_logger.info(f"Data shape: {df.shape}")
            return df
        except FileNotFoundError as e:
            app_logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            app_logger.error(f"Unexpected error while loading data: {e}")
            raise
