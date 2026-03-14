# NutriAccess Food Environment Analysis

## Project Overview

This project analyzes multiple datasets on food access, socioeconomic conditions, and health indicators to better understand the factors associated with food deserts and limited access to nutrition.

The project integrates datasets from several sources, including:

    American Community Survey (ACS)

    USDA Food Access Research Atlas

    USDA Food Environment Atlas

    CDC PLACES health dataset

    OpenStreetMap (OSM) geographic data

The analysis pipeline uses AWS cloud storage and Python-based tools to ingest, process, and explore these datasets.

## Data Storage

All raw datasets are stored in an Amazon S3 bucket, allowing the team to access a shared data source without downloading files locally.

### S3 bucket location:

```
s3://sagemaker-us-east-1-274099962196/rawData/
```

Datasets are organized into folders within the bucket to maintain a consistent structure.

Example structure:

```
rawData/
    ACSDP1Y2024.DP05...
    ACSST1Y2024.S1701...
    ACSST1Y2024.S1901...
    FoodAccess/
    FoodEnvironment/
    PLACES__Local_Data...
    geofabrik_NorCal/
    geofabrik_SoCal/
```

### Tools Used

The project uses the following tools:

    AWS Services

    Amazon S3 for dataset storage

    SageMaker Studio for data ingestion and exploration

    Python Libraries

    pandas for tabular data processing

    geopandas for geospatial datasets

    boto3 for interacting with AWS S3

#### Development Tools

    Jupyter notebooks for analysis

    GitHub for version control and collaboration

## Data Ingestion

Datasets are ingested directly from S3 using Python within a SageMaker Studio notebook.

Example ingestion code:

```
import pandas as pd

BUCKET = "sagemaker-us-east-1-274099962196"
RAW_PREFIX = "rawData/"

df = pd.read_csv(
    f"s3://{BUCKET}/{RAW_PREFIX}ACSDP1Y2024.DP05-2026-03-13T140903.csv"
)
```

To ensure reproducibility, the notebook automatically discovers and loads CSV files from the raw data folder.

### Geospatial Data Processing

OpenStreetMap POI datasets were used to extract food retail locations, including:

    supermarkets

    grocery stores

    convenience stores

The following categories were filtered from the OSM dataset:

    convenience
    greengrocer
    general
    department_store
    market_place

These locations were combined into a statewide dataset representing food store access points.

## Processed Datasets

Derived datasets are stored in the project repository under:

```
data/processed/

Current processed outputs include:

california_food_store_locations.geojson
california_food_store_locations.csv
```

These files contain filtered grocery and food retail locations extracted from OpenStreetMap POI datasets.

## Repository Structure

ADS508Group3
│
├── data
│   ├── processed
│   │   ├── california_food_store_locations.geojson
│   │   └── california_food_store_locations.csv
│
├── NutriAccessAnalytics.ipynb
├── dependency_setup.py
├── README.md

## Data Exploration

During the exploration phase, the datasets are evaluated for:

    missing values

    data quality issues

    key fields and join variables

    variable data types

    potential biases or inconsistencies

    relationships between socioeconomic and food access indicators

Exploration is conducted using pandas and geopandas within the project notebook.

## Running the Project

To reproduce the analysis:

    Clone the repository

```
git clone https://github.com/svthaker/ADS508Group3
```

    Open the SageMaker notebook environment

    Run the dependency setup file if required

    Execute the notebook cells to ingest and explore the datasets

    Datasets will load directly from the S3 bucket.

## Future Work

Planned next steps include:

    cleaning and standardizing socioeconomic datasets

    integrating food access indicators with demographic variables

    spatial analysis of food store density

    identifying potential food desert regions

    developing visualizations and maps of food accessibility