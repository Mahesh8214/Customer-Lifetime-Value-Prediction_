# src/main.py

from src.data.loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.clv_model import CLVModel
from src.config import Config

def main():
    # Load data
    loader = DataLoader(Config.RAW_DATA_PATH)
    df = loader.load_data()

    # Clean and transform
    fe = FeatureEngineer(df)
    clean_df = fe.clean_data()
    rfm_df = fe.compute_rfm(Config.REF_DATE)

    # Train model
    model = CLVModel(n_clusters=Config.N_CLUSTERS)
    rfm_clustered = model.train(rfm_df)
    model.save_model(Config.MODEL_PATH)

    print("Pipeline complete. Model trained and saved.")

if __name__ == "__main__":
    main()
