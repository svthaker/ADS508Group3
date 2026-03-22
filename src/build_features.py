import pandas as pd
from pathlib import Path

from .config import PROCESSED_DATA_DIR
from .preprocess_data import (
    clean_food_access,
    clean_census_data,
    clean_places_data,
    clean_food_environment_data,
)

PROCESSED_DIR = Path(PROCESSED_DATA_DIR)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_features() -> pd.DataFrame:
    food_access_df = clean_food_access()
    census_df = clean_census_data()
    places_df = clean_places_data()
    food_environment_df = clean_food_environment_data()

    features_df = (
        food_access_df
        .merge(census_df, on=["county", "state"], how="left")
        .merge(places_df, on=["county", "state"], how="left")
        .merge(food_environment_df, on=["county", "state"], how="left")
    )

    # Save CSV 
    csv_path = PROCESSED_DIR / "nutriaccess_features.csv"
    features_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Save PARQUET 
    parquet_path = PROCESSED_DIR / "nutriaccess_features.parquet"
    features_df.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet: {parquet_path}")    

    return features_df