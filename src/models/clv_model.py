# src/models/clv_model.py

import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.utils.logger import logging


class CLVModel:
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        logging.info(f"Initialized CLVModel with {self.n_clusters} clusters.")

    def train(self, df: pd.DataFrame):
        try:
            logging.info("Starting model training...")

            # Drop CustomerID and all non-numeric columns
            features = df.drop(columns=['Customer ID'], errors='ignore').select_dtypes(include=['number'])
            
            logging.info(f"Columns used for training: {features.columns.tolist()}")
            logging.info(f"Feature types:\n{features.dtypes}")

            # Standardize features
            X_scaled = self.scaler.fit_transform(features)

            # Train KMeans model
            self.model.fit(X_scaled)

            # Attach labels back to original data
            df['Segment'] = self.model.labels_

            logging.info("Model training completed successfully.")
            return df
        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            raise

    def save_model(self, model_path='saved_models/kmeans_model.pkl', scaler_path='saved_models/scaler.pkl'):
        try:
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            logging.info(f"Model and scaler saved to {model_path} and {scaler_path}.")
        except Exception as e:
            logging.error(f"Saving model failed: {str(e)}")
            raise
