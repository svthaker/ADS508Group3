from pathlib import Path
import pandas as pd
import geopandas as gpd
import boto3

from .config import AWS_REGION, S3_BUCKET, LOCAL_DATA_DIR, PROCESSED_DATA_DIR

RAW_DIR = Path(LOCAL_DATA_DIR)
PROCESSED_DIR = Path(PROCESSED_DATA_DIR)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def upload_file_to_s3(local_path: str, bucket: str, s3_key: str) -> None:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.upload_file(local_path, bucket, s3_key)
    print(f"Uploaded to s3://{bucket}/{s3_key}")


def clean_food_access():
    raw_path = (
        RAW_DIR
        / "FoodAccess"
        / "2019_Food_Access_Research_Atlas_Data"
        / "Food Access Research Atlas.csv"
    )

    df = pd.read_csv(raw_path)

    selected_columns = [
        "CensusTract",
        "State",
        "County",
        "Urban",
        "Pop2010",
        "PovertyRate",
        "MedianFamilyIncome",
        "LowIncomeTracts",
        "LILATracts_1And10",
        "LILATracts_halfAnd10",
        "LILATracts_1And20",
        "TractHUNV",
        "TractSNAP",
    ]

    clean_df = df[selected_columns].copy()

    numeric_fill_cols = [
        "PovertyRate",
        "MedianFamilyIncome",
        "TractHUNV",
        "TractSNAP",
    ]

    for col in numeric_fill_cols:
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    output_path = PROCESSED_DIR / "food_access_clean.csv"
    clean_df.to_csv(output_path, index=False)
    print(f"Saved local file: {output_path}")

    upload_file_to_s3(
        str(output_path),
        S3_BUCKET,
        "processedData/food_access_clean.csv"
    )


def inspect_acs_files():
    dp05_path = RAW_DIR / "ACSDP1Y2024.DP05-2026-03-13T140903.csv"
    s1701_path = RAW_DIR / "ACSST1Y2024.S1701-2026-03-13T140807.csv"
    s1901_path = RAW_DIR / "ACSST1Y2024.S1901-2026-03-13T140835.csv"

    dp05_df = pd.read_csv(dp05_path)
    s1701_df = pd.read_csv(s1701_path)
    s1901_df = pd.read_csv(s1901_path)

    print(
        "ACS files were ingested successfully, but they appear to be "
        "summary tables rather than tract-level files, so they are not "
        "processed further in this stage."
    )

    return {
        "dp05": dp05_df,
        "s1701": s1701_df,
        "s1901": s1901_df,
    }


def clean_places_data():
    raw_path = RAW_DIR / "PLACES__Local_Data_for_Better_Health,_County_Data,_2025_release_20260313.csv"

    df = pd.read_csv(raw_path, low_memory=False)

    ca_df = df[df["StateAbbr"] == "CA"].copy()

    selected_measures = ["OBESITY", "DIABETES", "LPA", "CSMOKING", "BPHIGH"]
    ca_df = ca_df[ca_df["MeasureId"].isin(selected_measures)].copy()

    ca_df = ca_df[
        ["StateAbbr", "LocationName", "MeasureId", "Measure", "Data_Value"]
    ].copy()

    ca_wide = ca_df.pivot_table(
        index=["StateAbbr", "LocationName"],
        columns="MeasureId",
        values="Data_Value",
        aggfunc="mean"
    ).reset_index()

    ca_wide.columns.name = None
    ca_wide.columns = [str(col).strip().lower() for col in ca_wide.columns]
    ca_wide = ca_wide.rename(columns={"locationname": "county_name"})

    output_path = PROCESSED_DIR / "places_ca_health_clean.csv"
    ca_wide.to_csv(output_path, index=False)
    print(f"Saved local file: {output_path}")

    upload_file_to_s3(
        str(output_path),
        S3_BUCKET,
        "processedData/places_ca_health_clean.csv"
    )


def clean_food_environment_data():
    raw_path = (
        RAW_DIR
        / "FoodEnvironment"
        / "2025-food-environment-atlas-data"
        / "StateAndCountyData.csv"
    )

    df_main = pd.read_csv(raw_path, low_memory=False)

    selected_codes = [
        "POVRATE21",
        "PCT_DIABETES_ADULTS19",
        "PCT_OBESE_ADULTS22",
    ]

    filtered = df_main[df_main["Variable_Code"].isin(selected_codes)].copy()

    wide_df = filtered.pivot_table(
        index=["State", "County"],
        columns="Variable_Code",
        values="Value",
        aggfunc="first"
    ).reset_index()

    wide_df.columns.name = None
    wide_df.columns = [str(col).strip().lower() for col in wide_df.columns]

    output_path = PROCESSED_DIR / "food_environment_clean.csv"
    wide_df.to_csv(output_path, index=False)
    print(f"Saved local file: {output_path}")

    upload_file_to_s3(
        str(output_path),
        S3_BUCKET,
        "processedData/food_environment_clean.csv"
    )


def clean_retail_food_locations():
    norcal_path = (
        RAW_DIR
        / "geofabrik_NorCal"
        / "norcal-260312-free.shp"
        / "gis_osm_pois_free_1.shp"
    )

    socal_path = (
        RAW_DIR
        / "geofabrik_SoCal"
        / "socal-260312-free.shp"
        / "gis_osm_pois_free_1.shp"
    )

    pois_norcal = gpd.read_file(norcal_path)
    pois_socal = gpd.read_file(socal_path)

    pois_ca = pd.concat([pois_norcal, pois_socal], ignore_index=True)
    pois_ca = gpd.GeoDataFrame(pois_ca, geometry="geometry", crs=pois_norcal.crs)

    pois_ca["fclass"] = pois_ca["fclass"].astype(str).str.lower()

    store_classes = [
        "supermarket",
        "convenience",
        "greengrocer",
        "general",
        "department_store",
        "market_place",
    ]

    food_stores = pois_ca[pois_ca["fclass"].isin(store_classes)].copy()

    print("Food store class counts:")
    print(food_stores["fclass"].value_counts())
    print(f"Total stores: {food_stores.shape[0]}")

    geojson_path = PROCESSED_DIR / "california_food_store_locations.geojson"
    csv_path = PROCESSED_DIR / "california_food_store_locations.csv"

    food_stores.to_file(geojson_path, driver="GeoJSON")
    food_stores.to_csv(csv_path, index=False)

    print(f"Saved local file: {geojson_path}")
    print(f"Saved local file: {csv_path}")

    upload_file_to_s3(
        str(csv_path),
        S3_BUCKET,
        "processedData/california_food_store_locations.csv",
    )

    upload_file_to_s3(
        str(geojson_path),
        S3_BUCKET,
        "processedData/california_food_store_locations.geojson",
    )


def run_all_cleaning():
    clean_food_access()
    clean_places_data()
    clean_food_environment_data()
    clean_retail_food_locations()