from pathlib import Path
import boto3
import pandas as pd

from .config import AWS_REGION, S3_BUCKET, S3_PREFIX, LOCAL_DATA_DIR

SKIP_TERMS = ["readme", "variablelookup", "variablelist"]


def get_s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def list_s3_files(bucket: str, prefix: str) -> list[str]:
    """
    List all object keys under an S3 prefix.
    """
    if not bucket:
        raise ValueError("S3 bucket is not set. Check your .env file.")

    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    files = []
    for page in pages:
        for obj in page.get("Contents", []):
            files.append(obj["Key"])

    return files


def clean_dataset_name(key: str) -> str:
    """
    Convert an S3 file path into a cleaner dataset name.
    """
    name = Path(key).name.replace(".csv", "")
    name = (
        name.replace(" ", "_")
        .replace(",", "")
        .replace("-", "_")
        .replace(".", "_")
    )
    return name


def is_useful_csv(key: str, skip_terms: list[str] | None = None) -> bool:
    """
    Return True if the S3 object is a useful CSV dataset to load.
    """
    skip_terms = skip_terms or SKIP_TERMS
    lower_key = key.lower()

    return key.endswith(".csv") and not any(term in lower_key for term in skip_terms)


def load_csv_datasets_from_s3(bucket: str, prefix: str) -> dict[str, pd.DataFrame]:
    """
    Load useful CSV datasets from an S3 prefix into a dictionary of DataFrames.
    """
    files = list_s3_files(bucket, prefix)

    if not files:
        print(f"No files found in s3://{bucket}/{prefix}")
        return {}

    datasets = {}

    for key in files:
        if is_useful_csv(key):
            s3_path = f"s3://{bucket}/{key}"
            dataset_name = clean_dataset_name(key)

            print(f"Loading {s3_path}")
            datasets[dataset_name] = pd.read_csv(s3_path, low_memory=False)

    return datasets


def download_raw_files(bucket: str, prefix: str, local_dir: str, overwrite: bool = False) -> None:
    """
    Download all raw files from S3 to a local directory while preserving structure.
    """
    files = list_s3_files(bucket, prefix)

    if not files:
        print(f"No files found in s3://{bucket}/{prefix}")
        return

    s3 = get_s3_client()
    local_base = Path(local_dir)
    local_base.mkdir(parents=True, exist_ok=True)

    for key in files:
        if key.endswith("/"):
            continue

        relative_path = key.replace(prefix, "", 1)
        local_path = local_base / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists() and not overwrite:
            print(f"Skipping existing file: {local_path}")
            continue

        print(f"Downloading {key} -> {local_path}")
        s3.download_file(bucket, key, str(local_path))

    print("Raw file download complete.")