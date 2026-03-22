import pandas as pd
from pathlib import Path

from .config import PROCESSED_DATA_DIR, S3_BUCKET
from .ingest_data import upload_file_to_s3

PROCESSED_DIR = Path(PROCESSED_DATA_DIR)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def standardize_keys(df: pd.DataFrame, county_col: str, state_col: str) -> pd.DataFrame:
    """
    Standardize county and state columns for merging.
    """
    df[county_col] = df[county_col].astype(str).str.lower().str.strip()
    df[state_col] = df[state_col].astype(str).str.lower().str.strip()
    return df


def build_features() -> pd.DataFrame:
    """
    Build the final California county-level NutriAccess feature table from processed datasets,
    save it locally as CSV and Parquet, and upload the Parquet file to S3.
    """
    # Read processed datasets
    food_access_df = pd.read_csv(PROCESSED_DIR / "food_access_county_clean.csv")
    census_df = pd.read_csv(PROCESSED_DIR / "census_features_clean.csv")
    places_df = pd.read_csv(PROCESSED_DIR / "places_ca_health_clean.csv")
    food_environment_df = pd.read_csv(PROCESSED_DIR / "food_environment_clean.csv")

    # Standardize merge keys
    food_access_df = standardize_keys(food_access_df, "county", "state")
    census_df = standardize_keys(census_df, "county", "state")
    places_df = standardize_keys(places_df, "county", "state")
    food_environment_df = standardize_keys(food_environment_df, "county", "state")

    # Restrict to California so geographic scope matches across datasets
    food_access_df = food_access_df[food_access_df["state"] == "california"].copy()
    census_df = census_df[census_df["state"] == "california"].copy()
    places_df = places_df[places_df["state"] == "california"].copy()
    food_environment_df = food_environment_df[food_environment_df["state"] == "california"].copy()

    # Merge datasets
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

    # Save Parquet
    parquet_path = PROCESSED_DIR / "nutriaccess_features.parquet"
    features_df.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet: {parquet_path}")

    # Upload Parquet to S3
    upload_file_to_s3(
        str(parquet_path),
        S3_BUCKET,
        "processedData/nutriaccess_features.parquet"
    )
    print(f"Uploaded Parquet to s3://{S3_BUCKET}/processedData/nutriaccess_features.parquet")

    return features_df