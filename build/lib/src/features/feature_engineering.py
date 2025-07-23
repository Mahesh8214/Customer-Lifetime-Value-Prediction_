import pandas as pd
from datetime import datetime
from src.utils.logger import app_logger

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def clean_data(self) -> pd.DataFrame:
        app_logger.info("Cleaning data...")
        self.df.dropna(subset=['Customer ID'], inplace=True)
        self.df = self.df[self.df['Quantity'] > 0]
        self.df['TotalPrice'] = self.df['Quantity'] * self.df['Price']
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        return self.df

    def compute_rfm(self, ref_date: str = "2011-12-10") -> pd.DataFrame:
        app_logger.info("Computing RFM features...")
        ref_date = pd.to_datetime(ref_date)
        rfm = self.df.groupby('Customer ID').agg({
            'InvoiceDate': lambda x: (ref_date - x.max()).days,
            'Invoice': 'nunique',
            'TotalPrice': 'sum'
        }).reset_index()

        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        app_logger.info("RFM calculation complete.")
        return rfm
