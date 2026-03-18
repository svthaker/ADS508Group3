from src.ingest_data import download_raw_files
from src.preprocess_data import run_all_cleaning
from src.config import S3_BUCKET, S3_PREFIX, LOCAL_DATA_DIR


def main():
    print("Starting pipeline...")

    download_raw_files(S3_BUCKET, S3_PREFIX, LOCAL_DATA_DIR)
    run_all_cleaning()

    print("Pipeline finished.")


if __name__ == "__main__":
    main()