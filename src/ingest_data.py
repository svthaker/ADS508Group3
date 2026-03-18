from pathlib import Path
import boto3
from config import AWS_REGION, S3_BUCKET, S3_PREFIX, LOCAL_DATA_DIR


def download_data():
    """
    Download files from the configured S3 bucket/prefix
    into the local raw data directory.
    """
    if not S3_BUCKET:
        raise ValueError("S3_BUCKET is not set. Check your .env file.")

    s3 = boto3.client("s3", region_name=AWS_REGION)

    local_dir = Path(LOCAL_DATA_DIR)
    local_dir.mkdir(parents=True, exist_ok=True)

    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)

    if "Contents" not in response:
        print(f"No files found in s3://{S3_BUCKET}/{S3_PREFIX}")
        return

    for obj in response["Contents"]:
        key = obj["Key"]

        if key.endswith("/"):
            continue

        relative_path = key.replace(S3_PREFIX, "", 1)
        local_path = local_dir / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {key} -> {local_path}")
        s3.download_file(S3_BUCKET, key, str(local_path))

    print("Download complete.")