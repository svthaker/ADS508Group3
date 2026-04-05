# 🥗 NutriAccess Food Environment Analysis

## 📌 Project Overview

NutriAccess is a data science pipeline designed to analyze food access, socioeconomic conditions, and health indicators to identify communities at risk of limited access to healthy food.

This project integrates multiple public datasets to answer a key question:

Can we identify which counties are most at risk of limited food access, and how severe that risk is?

The pipeline combines data engineering, exploratory analysis, and machine learning to support data-driven decision-making for food accessibility.

### 🎯 Business Problem

Access to healthy and affordable food is a critical public health issue. Many communities face barriers such as distance, income, and limited infrastructure, resulting in food deserts.

This project aims to:

Identify high-risk counties (classification)
Estimate the severity of food access risk (regression)
Provide insights to support policymakers, retailers, and public health organizations

### 📊 Data Sources

The pipeline integrates datasets from:

USDA Food Access Research Atlas
USDA Food Environment Atlas
U.S. Census (ACS) – demographics and income
CDC PLACES – health indicators
OpenStreetMap (OSM) – food retail locations

## ⚙️ Pipeline Architecture

The pipeline follows a structured workflow:

1. Data Ingestion
   
    - Load datasets from AWS S3 using boto3
    - Standardize file structure and naming
      
2. Data Processing & Feature Engineering

    - Clean and merge datasets
    - Aggregate data to the county level
    - Create key features such as:
    - lila_rate
    - high_food_access_risk
      
3. Modeling

    - Classification models to identify high-risk counties
    - Regression models to estimate severity of food access risk
      
4. Evaluation & Visualization

    - Compare model performance
    - Generate insights and visualizations

## 🧠 Modeling Workflow
**Classification Models**

Used to predict high_food_access_risk:

    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting ✅ (Selected Model)

**Regression Models**

Used to predict lila_rate:

    - Linear Regression
    - Random Forest Regressor ✅ (Selected Model)
    - Gradient Boosting Regressor
**Model Selection**

Gradient Boosting (Classification): Selected for strong performance and ability to capture nonlinear relationships

Random Forest Regressor (Regression): Selected for best balance of accuracy and generalization (R² ≈ 0.60)

### 📈 Key Results

Classification models successfully identified high-risk counties

Regression models captured variation in food access severity

Nonlinear ensemble models outperformed simpler models

Results highlight geographic disparities in food access

⚠️ Note: *Some models achieved near-perfect performance due to small dataset size and potential overfitting. Results should be interpreted as directional insights.*

## 📁 Repository Structure

```bash
ADS508Group3/
│
├── src/
│   
│   ├── config.py
│   ├── ingest_data.py
│   ├── preprocess_data.py
│   ├── build_features.py
│   ├── modeling/
│   │   ├── classification.py
│   │   ├── regression.py
│   │   ├── data_prep.py
│   │   ├── regression_prep.py
│   │   ├── evaluation.py
│   │   ├── regression_evaluation.py
│
├── notebooks/
│   ├── gradient_boosting.ipynb
│   ├── LogisticRegressionModel
│   ├── Random_Forest.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── final/
│
├── installation/
│
├── main.py
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
```

### 🛠️ Environment Setup

Copy the example environment file:

```bash
cp .env.example .env
```

Update with your AWS configuration: 

```env
AWS_REGION=us-east-1
S3_BUCKET=BucketName
S3_PREFIX=rawData/
LOCAL_DATA_DIR=data/raw
```

Install project dependencies:

```
pip install -r requirements.txt
```

## ▶️ Running the Pipeline

Run the full pipeline 
```
python main.py
```

This will:

    load configuration from .env

    connect to AWS S3

    download raw datasets to the local environment

    prepare data for downstream analysis

## 📊 Outputs & Artifacts
Processed datasets stored in:

``` 
data/processed/
```

Final modeling dataset:
```
data/final/
```

### Installation and Environment Validation

The `installation/` folder contains setup notebooks used to prepare and validate the SageMaker Studio lab environment. These notebooks document project setup steps such as:

    - installing or checking required dependencies
    - validating the SageMaker Studio environment
    - creating or verifying S3 bucket access
    - checking IAM role configuration

These steps are not required for daily pipeline execution, but are useful for initial setup and debugging.

## Running the Project (End-to-End)

The project uses a Python-based preprocessing pipeline located in the src/ directory.

Run the pipeline with:

```
git clone https://github.com/svthaker/ADS508Group3
cd ADS508Group3
cp .env.example .env
pip install -r requirements.txt
python main.py
```

## Future Work

This pipeline could be extended into a decision-support tool for identifying food deserts in real time.

Key areas for future improvement include:

    Expanding the size, coverage, and granularity of the dataset
    Strengthening the modeling framework through more rigorous validation and hyperparameter tuning
    Extending the pipeline into a more operational, user-facing decision-support system

## 🌎 Impact

This project demonstrates how data-driven approaches can help identify and prioritize communities most in need, supporting more equitable access to healthy food.

## 👥 Team

Shery Thaker
James Shoenhair
Nancy Walker
