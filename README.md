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

## Project Setup

### Data Storage

Project data is stored in Amazon S3. Depending on team workflow, each teammate may use their own bucket, so the bucket name should be configured locally in the `.env` file rather than hardcoded in project scripts.

Example `.env` configuration:

```env
AWS_REGION=us-east-1
S3_BUCKET=BucketName
S3_PREFIX=rawData/
LOCAL_DATA_DIR=data/raw
```

Datasets are organized into folders within the bucket to maintain a consistent structure.

Example structure:

```
rawData/
    ACSDP1Y2024.DP05...
    ACSST1Y2024.S1701...
    ACSST1Y2024.S1901...
    Censue_Dataset/
    FoodAccess/
    FoodEnvironment/
    PLACES__Local_Data...
    geofabrik_NorCal/
    geofabrik_SoCal/
```

### Environment Setup

Copy the example environment file and update it with your local configuration:

```
cp .env.example .env
```

Install project dependencies:

```
pip install -r requirements.txt
```

### Run the Project

```
python src/main.py
```
This will:

    load configuration from .env

    connect to AWS S3

    download raw datasets to the local environment

    prepare data for downstream analysis

### Installation and Environment Validation

The `installation/` folder contains setup notebooks used to prepare and validate the SageMaker Studio lab environment. These notebooks document project setup steps such as:

    - installing or checking required dependencies
    - validating the SageMaker Studio environment
    - creating or verifying S3 bucket access
    - checking IAM role configuration

These steps are not required for daily pipeline execution, but are useful for initial setup and debugging.

### Tools Used

#### AWS Services

    Amazon S3 (data storage)

    SageMaker Studio (development environment)

#### Python Libraries

    pandas (data processing)

    geopandas (geospatial analysis)

    boto3 (AWS interaction)

    python-dotenv (environment configuration)

#### Development Tools

    Jupyter notebooks (exploration and validation)

    GitHub (version control and collaboration)

#### Development Tools

    Jupyter notebooks for analysis

    GitHub for version control and collaboration

## Data Ingestion

Data ingestion is handled programmatically through the pipeline using boto3.

The pipeline:

    reads configuration from .env

    connects to S3

    iterates through objects in the specified prefix

    downloads datasets to a local directory

### Example structure:

```
src/
    ingest_data.py
```

This replaces manual ingestion steps previously performed in notebooks and ensures reproducibility across team members.

### Geospatial Data Processing

OpenStreetMap POI datasets were used to extract food retail locations, including:

    supermarkets

    grocery stores

    convenience stores

Filtered categories include:

    convenience

    greengrocer

    general

    department_store

    market_place

These are combined into a statewide dataset representing food access points.

## Processed Datasets

Derived datasets are stored in the project repository under:

```
data/processed/
```
Current processed outputs include:

california_food_store_locations.geojson
california_food_store_locations.csv

## Repository Structure

ADS508Group3/
│
├── src/
│   ├── main.py
│   ├── config.py
│   ├── ingest_data.py
│
├── installation/
│   ├── 01_setup_dependencies.ipynb
│   ├── 02_check_environment.ipynb
│   ├── 03_validate_s3_bucket.ipynb
│   ├── 04_validate_iam_role.ipynb
│
├── data/
├── documents/
│
├── .env              ❌ (NOT tracked)
├── .env.example      ✅ (shared template)
├── .gitignore
├── requirements.txt
├── README.md

## Data Exploration

Exploratory analysis is conducted in notebooks using pandas and geopandas.

This includes:

    missing value analysis

    data quality checks

    identifying key join fields

    detecting inconsistencies or bias

    exploring relationships between datasets

## Running the Project (End-to-End)

To reproduce the project:

```
git clone https://github.com/svthaker/ADS508Group3
cd ADS508Group3
cp .env.example .env
pip install -r requirements.txt
python src/main.py
```

## Future Work