from src.ingest_data import download_raw_files
from src.preprocess_data import run_all_cleaning
from src.build_features import build_features  # ⭐ ADD THIS
from src.config import S3_BUCKET, S3_PREFIX, LOCAL_DATA_DIR


def main():
    print("Starting pipeline...")

    # Step 1: Ingest raw data
    download_raw_files(S3_BUCKET, S3_PREFIX, LOCAL_DATA_DIR)

    # Step 2: Clean + preprocess datasets
    run_all_cleaning()

    # Step 3: Build final feature dataset (CSV + PARQUET)
    build_features()   # CREATES MODELING DATASET

    print("Pipeline finished.")


if __name__ == "__main__":
    main()