# src/main.py

import os
from src.data.loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.clv_model import CLVModel
from src.utils.logger import logging
from src.config import Config

import matplotlib.pyplot as plt
import seaborn as sns

def main():
    try:
        logging.info("Pipeline execution started.")

        # 1. Load data
        data_path = Config.RAW_DATA_PATH
        loader = DataLoader(data_path)
        df = loader.load_data()

        # 2. Feature Engineering
        fe = FeatureEngineer(df)

        print("Before preprocessing")
        features_df = fe.preprocess()
        print("After preprocessing")
        print(features_df.columns) 

        logging.info(f"Feature engineering complete: {features_df.shape}")
        print(features_df.head())

        rfm_df = fe.generate_rfm_features()
        print("\n\nRFM Features:")
        print(rfm_df.head())

        # Save processed features
        os.makedirs("data/processed", exist_ok=True)
        features_df.to_csv("data/processed/processed_features.csv", index=False)
        logging.info("Processed features saved to data/processed/processed_features.csv")


        # 3. Model Training
        model = CLVModel(n_clusters=4)
        clustered_df = model.train(features_df)
        print("\n\nClustered DF:")
        print(clustered_df.head())
        logging.info(f"Model training complete: {clustered_df.shape}")

        # 4. Save model
        os.makedirs("saved_models", exist_ok=True)
        model.save_model()

        # 5. Save segmented data
        clustered_df.to_csv("data/segmented_customers.csv", index=False)
        logging.info("Segmented customer data saved.")
        
         # 6. Segment insights
        segment_counts = clustered_df['Segment'].value_counts().sort_index()
        logging.info(f"Customer counts by segment:\n{segment_counts}")

        segment_summary = clustered_df.groupby('Segment').mean(numeric_only=True)
        logging.info(f"Segment summary:\n{segment_summary}")

        logging.info("Pipeline execution finished successfully.")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")

# def visualize_segments(rfm_df):
#     plt.figure(figsize=(8, 5))
#     sns.countplot(x='Segment', data=rfm_df)
#     plt.title("Customer Count per Segment")
#     plt.xlabel("Segment")
#     plt.ylabel("Number of Customers")
#     plt.tight_layout()
#     plt.savefig("outputs/segment_distribution.png")
#     plt.show()

if __name__ == "__main__":
    main()
