"""
Utility functions for running Athena queries and loading results into pandas.
"""

import time
import boto3
import pandas as pd

from src.config import AWS_REGION, ATHENA_DATABASE, ATHENA_OUTPUT


def run_athena_query(
    query,
    database=ATHENA_DATABASE,
    output_location=ATHENA_OUTPUT,
    region=AWS_REGION,
    poll_interval=2,
    timeout=60
):
    """
    Run a SQL query in Athena and return the results as a pandas DataFrame.

    Args:
        query (str): SQL query string to execute.
        database (str): Athena database name.
        output_location (str): S3 path where Athena query results are stored.
        region (str): AWS region.
        poll_interval (int): Seconds between query status checks.
        timeout (int): Maximum time in seconds to wait for query completion.

    Returns:
        tuple:
            - df (pd.DataFrame): Query results
            - query_execution_id (str): Athena query execution ID
            - result_path (str): S3 path to result CSV
    """
    if not query.strip():
        raise ValueError("Query string is empty.")

    if not output_location:
        raise ValueError("Athena output location is not set.")

    if not output_location.endswith("/"):
        output_location = f"{output_location}/"

    athena = boto3.client("athena", region_name=region)

    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_location},
    )

    query_execution_id = response["QueryExecutionId"]

    start_time = time.time()

    while True:
        result = athena.get_query_execution(QueryExecutionId=query_execution_id)
        status = result["QueryExecution"]["Status"]["State"]

        if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            break

        if time.time() - start_time > timeout:
            raise TimeoutError(f"Athena query timed out after {timeout} seconds.")

        time.sleep(poll_interval)

    if status != "SUCCEEDED":
        reason = result["QueryExecution"]["Status"].get("StateChangeReason", "Unknown error")
        raise RuntimeError(f"Athena query failed: {status} - {reason}")

    result_path = f"{output_location}{query_execution_id}.csv"

    # Wait for file to appear in S3
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            df = pd.read_csv(result_path)
            break
        except FileNotFoundError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(2)

    return df, query_execution_id, result_path