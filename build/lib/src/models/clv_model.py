import pandas as pd
from sklearn.cluster import KMeans
import joblib
from src.utils.logger import app_logger

class CLVModel:
    def __init__(self, n_clusters: int = 4):
        self.model = KMeans(n_clusters=n_clusters, random_state=42)

    def train(self, X: pd.DataFrame) -> pd.DataFrame:
        app_logger.info("Training KMeans model...")
        self.model.fit(X)
        X['Cluster'] = self.model.labels_
        return X

    def save_model(self, path: str):
        app_logger.info(f"Saving model to {path}")
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        app_logger.info(f"Loading model from {path}")
        self.model = joblib.load(path)
