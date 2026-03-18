import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "rawData/")
LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR", "data/raw")