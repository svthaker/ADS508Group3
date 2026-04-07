"""
Central configuration for AWS, S3, Athena, and local project paths.
Loads values from .env and validates required settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# AWS / S3 config
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "rawData/")
PROCESSED_S3_PREFIX = os.getenv("PROCESSED_S3_PREFIX", "processedData/")

# Athena config
ATHENA_DATABASE = os.getenv("ATHENA_DATABASE", "nutriaccess_db")
ATHENA_TABLE = os.getenv("ATHENA_TABLE", "nutriaccess_features")
ATHENA_OUTPUT_PREFIX = os.getenv("ATHENA_OUTPUT_PREFIX", "athena-results/query-results/")

# Derived paths
ATHENA_OUTPUT = f"s3://{S3_BUCKET}/{ATHENA_OUTPUT_PREFIX}" if S3_BUCKET else None

# Local paths
LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")

# Validation
if not S3_BUCKET:
    raise ValueError("S3_BUCKET is not set. Please check your .env file.")

if not ATHENA_DATABASE:
    raise ValueError("ATHENA_DATABASE is not set. Please check your .env file.")

if not ATHENA_TABLE:
    raise ValueError("ATHENA_TABLE is not set. Please check your .env file.")

if not ATHENA_OUTPUT:
    raise ValueError("ATHENA_OUTPUT could not be constructed. Check S3_BUCKET and ATHENA_OUTPUT_PREFIX.")