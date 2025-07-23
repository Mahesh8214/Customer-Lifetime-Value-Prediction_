# src/features/feature_engineering.py

import pandas as pd #
import numpy as np # type: ignore
from datetime import datetime
from src.utils.logger import logging


class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def preprocess(self):
        try:
            logging.info("Starting preprocessing...")
            
            # Drop NA values
            self.df.dropna(inplace=True)

            # Remove canceled orders
            self.df = self.df[~self.df['Invoice'].astype(str).str.startswith('C')]

            # Convert InvoiceDate to datetime
            self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])

            # Create TotalPrice feature
            self.df['TotalPrice'] = self.df['Quantity'] * self.df['Price']

            logging.info("Preprocessing completed successfully.")
            return self.df
        except Exception as e:
            logging.error(f"Preprocessing failed: {str(e)}")
            raise

    def generate_rfm_features(self, reference_date=None):
        try:
            logging.info("Starting RFM feature generation...")

            if reference_date is None:
                reference_date = self.df['InvoiceDate'].max() + pd.Timedelta(days=1)

            rfm = self.df.groupby('Customer ID').agg({
                'InvoiceDate': lambda x: (reference_date - x.max()).days,
                'Invoice': 'nunique',
                'TotalPrice': 'sum'
            }).rename(columns={
                'InvoiceDate': 'Recency',
                'Invoice': 'Frequency',
                'TotalPrice': 'Monetary'
            })

            rfm = rfm[rfm['Monetary'] > 0]

            logging.info("RFM feature generation completed.")
            return rfm.reset_index()
        except Exception as e:
            logging.error(f"RFM feature generation failed: {str(e)}")
            raise
