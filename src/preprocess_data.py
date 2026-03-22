from pathlib import Path
import pandas as pd
import geopandas as gpd
import boto3

from .config import AWS_REGION, S3_BUCKET, LOCAL_DATA_DIR, PROCESSED_DATA_DIR

RAW_DIR = Path(LOCAL_DATA_DIR)
PROCESSED_DIR = Path(PROCESSED_DATA_DIR)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def upload_file_to_s3(local_path: str, bucket: str, s3_key: str) -> None:
    """
    Upload a local file to S3.
    """
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.upload_file(local_path, bucket, s3_key)
    print(f"Uploaded to s3://{bucket}/{s3_key}")


def _save_and_upload(df: pd.DataFrame, filename: str) -> None:
    """
    Save a processed dataframe locally and upload it to S3.
    """
    output_path = PROCESSED_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved local file: {output_path}")

    upload_file_to_s3(
        str(output_path),
        S3_BUCKET,
        f"processedData/{filename}"
    )


def _split_county_state(df: pd.DataFrame, county_col: str = "county_name") -> pd.DataFrame:
    """
    Split 'County, State' into separate standardized county and state columns.
    """
    df[["county", "state"]] = df[county_col].str.split(", ", expand=True)
    df["county"] = (
        df["county"]
        .str.replace(" County", "", regex=False)
        .str.strip()
        .str.lower()
    )
    df["state"] = df["state"].str.strip().str.lower()
    return df


def _standardize_county_state_columns(
    df: pd.DataFrame,
    state_col: str = "state",
    county_col: str = "county"
) -> pd.DataFrame:
    """
    Standardize existing county and state columns for merging.
    """
    df[state_col] = df[state_col].astype(str).str.lower().str.strip()
    df[county_col] = (
        df[county_col]
        .astype(str)
        .str.replace(" County", "", regex=False)
        .str.strip()
        .str.lower()
    )
    return df


def clean_food_access() -> pd.DataFrame:
    """
    Clean Food Access Research Atlas data, save tract-level output for EDA,
    aggregate to county level for modeling, and upload both outputs.
    """
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
        "Pop2010",
        "LowIncomeTracts",
        "LILATracts_1And10",
        "LILATracts_halfAnd10",
        "LILATracts_1And20",
    ]

    for col in numeric_fill_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    # Save tract-level cleaned file for EDA
    _save_and_upload(clean_df, "food_access_clean.csv")

    # Aggregate to county level for modeling
    food_access_county = clean_df.groupby(["State", "County"], as_index=False).agg({
        "Urban": "mean",
        "Pop2010": "sum",
        "PovertyRate": "mean",
        "MedianFamilyIncome": "mean",
        "LowIncomeTracts": "sum",
        "LILATracts_1And10": "sum",
        "LILATracts_halfAnd10": "sum",
        "LILATracts_1And20": "sum",
        "TractHUNV": "sum",
        "TractSNAP": "sum",
        "CensusTract": "count",
    })

    food_access_county = food_access_county.rename(columns={
        "State": "state",
        "County": "county",
        "Urban": "urban_share",
        "Pop2010": "population_2010",
        "PovertyRate": "avg_poverty_rate",
        "MedianFamilyIncome": "avg_median_family_income",
        "LowIncomeTracts": "low_income_tracts_count",
        "LILATracts_1And10": "lila_1_10_count",
        "LILATracts_halfAnd10": "lila_half_10_count",
        "LILATracts_1And20": "lila_1_20_count",
        "TractHUNV": "total_housing_units_no_vehicle",
        "TractSNAP": "total_snap_households",
        "CensusTract": "tract_count",
    })

    food_access_county = _standardize_county_state_columns(
        food_access_county,
        state_col="state",
        county_col="county"
    )

    _save_and_upload(food_access_county, "food_access_county_clean.csv")
    return food_access_county


def clean_census_data() -> pd.DataFrame:
    """
    Clean ACS DP05, S1701, and S1901 files, save individual outputs,
    merge them into a county-level census feature table, and upload results.
    """
    dp05_path = RAW_DIR / "Censue_Dataset" / "ACSDP5Y2023.DP05-Data.csv"
    s1701_path = RAW_DIR / "Censue_Dataset" / "ACSST5Y2023.S1701-Data.csv"
    s1901_path = RAW_DIR / "Censue_Dataset" / "ACSST5Y2023.S1901-Data.csv"

    dp05_df = pd.read_csv(dp05_path, low_memory=False)
    s1701_df = pd.read_csv(s1701_path, low_memory=False)
    s1901_df = pd.read_csv(s1901_path, low_memory=False)

    print("DP05 columns:", dp05_df.columns.tolist()[:10])
    print("S1701 columns:", s1701_df.columns.tolist()[:10])
    print("S1901 columns:", s1901_df.columns.tolist()[:10])

    # Remove ACS metadata row
    dp05_df = dp05_df.iloc[1:].reset_index(drop=True)
    s1701_df = s1701_df.iloc[1:].reset_index(drop=True)
    s1901_df = s1901_df.iloc[1:].reset_index(drop=True)

    # Clean DP05
    dp05_clean = dp05_df[[
        "GEO_ID",
        "NAME",
        "DP05_0001E",
        "DP05_0002E",
        "DP05_0003E",
        "DP05_0018E",
    ]].copy()

    dp05_clean.columns = [
        "geo_id",
        "county_name",
        "total_population",
        "male_population",
        "female_population",
        "median_age",
    ]

    for col in ["total_population", "male_population", "female_population", "median_age"]:
        dp05_clean[col] = pd.to_numeric(dp05_clean[col], errors="coerce")

    dp05_clean = _split_county_state(dp05_clean)

    # Clean S1701
    s1701_clean = s1701_df[[
        "GEO_ID",
        "NAME",
        "S1701_C03_001E",
    ]].copy()

    s1701_clean.columns = [
        "geo_id",
        "county_name",
        "poverty_rate",
    ]

    s1701_clean["poverty_rate"] = pd.to_numeric(
        s1701_clean["poverty_rate"], errors="coerce"
    )

    s1701_clean = _split_county_state(s1701_clean)

    # Clean S1901
    s1901_clean = s1901_df[[
        "GEO_ID",
        "NAME",
        "S1901_C01_012E",
    ]].copy()

    s1901_clean.columns = [
        "geo_id",
        "county_name",
        "median_income",
    ]

    s1901_clean["median_income"] = pd.to_numeric(
        s1901_clean["median_income"], errors="coerce"
    )

    s1901_clean = _split_county_state(s1901_clean)

    # Save individual cleaned files
    _save_and_upload(dp05_clean, "dp05_clean.csv")
    _save_and_upload(s1701_clean, "s1701_clean.csv")
    _save_and_upload(s1901_clean, "s1901_clean.csv")

    # Merge to one county-level census feature table
    census_clean = dp05_clean.merge(
        s1701_clean[["county", "state", "poverty_rate"]],
        on=["county", "state"],
        how="left"
    ).merge(
        s1901_clean[["county", "state", "median_income"]],
        on=["county", "state"],
        how="left"
    )

    _save_and_upload(census_clean, "census_features_clean.csv")
    return census_clean


def clean_places_data() -> pd.DataFrame:
    """
    Clean PLACES county-level health data for California and upload output.
    """
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
    ca_wide = ca_wide.rename(columns={
        "stateabbr": "state",
        "locationname": "county"
    })

    ca_wide = _standardize_county_state_columns(
        ca_wide,
        state_col="state",
        county_col="county"
    )

    _save_and_upload(ca_wide, "places_ca_health_clean.csv")
    return ca_wide


def clean_food_environment_data() -> pd.DataFrame:
    """
    Clean county-level food environment data and upload output.
    """
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
    wide_df = wide_df.rename(columns={
        "state": "state",
        "county": "county"
    })

    wide_df = _standardize_county_state_columns(
        wide_df,
        state_col="state",
        county_col="county"
    )

    _save_and_upload(wide_df, "food_environment_clean.csv")
    return wide_df


def clean_retail_food_locations() -> gpd.GeoDataFrame:
    """
    Clean and combine Northern and Southern California retail food location shapefiles.
    """
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

    return food_stores


def run_all_cleaning() -> dict[str, pd.DataFrame]:
    """
    Run all cleaning steps and return processed datasets.
    """
    food_access_df = clean_food_access()
    census_df = clean_census_data()
    places_df = clean_places_data()
    food_environment_df = clean_food_environment_data()
    retail_food_df = clean_retail_food_locations()

    return {
        "food_access": food_access_df,
        "census": census_df,
        "places": places_df,
        "food_environment": food_environment_df,
        "retail_food_locations": retail_food_df,
    }