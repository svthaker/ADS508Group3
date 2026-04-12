#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 00:10:26 2026

@author: jamesshoenhair
"""

from __future__ import annotations
import json
from pathlib import Path
import boto3
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

AWS_REGION        = "us-east-1"
S3_BUCKET         = "projects-sagemaker-ml-pipeline-787172416632-us-east-1-an"
S3_PREFIX         = "food-access-pipeline"
LOCAL_DATA_DIR    = "data"
PROCESSED_DATA_DIR = "data/processed"
 
PROCESSED_DIR = Path(PROCESSED_DATA_DIR)
MODELS_DIR = Path(LOCAL_DATA_DIR) / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
 
RANDOM_SEED = 42

#Helper

def _upload_to_s3(local_path: str, s3_key: str) -> None:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    print(f"Uploaded → s3://{S3_BUCKET}/{s3_key}")
    
    
#Load Cleaned Datastes(s) by County

def load_master_dataset() -> pd.DataFrame:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for fname in [
        "food_access_county_clean.csv",
        "census_features_clean.csv",
        "places_ca_health_clean.csv",
        "food_environment_clean.csv",
        "california_food_store_locations.csv",
    ]:
        s3.download_file(
            Bucket=S3_BUCKET,
            Key=f"{S3_PREFIX}/data/processed/{fname}",
            Filename=str(PROCESSED_DIR / fname),
        )
    print("[S3] All processed CSVs downloaded to data/processed")

    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        df["county"] = (
            df["county"].astype(str)
            .str.replace(" County", "", regex=False)
            .str.lower().str.strip()
        )
        df["state"] = df["state"].astype(str).str.lower().str.strip()
        return df
    
        food_access = pd.read_csv(PROCESSED_DIR / "food_access_county_clean.csv")
        census      = pd.read_csv(PROCESSED_DIR / "census_features_clean.csv")
        places      = pd.read_csv(PROCESSED_DIR / "places_ca_health_clean.csv")
        food_env    = pd.read_csv(PROCESSED_DIR / "food_environment_clean.csv")
        stores_raw  = pd.read_csv(PROCESSED_DIR / "california_food_store_locations.csv")

    
#Aggregates Food Retail Stores by Count

if "county" in stores_raw.columns:
        store_agg = (
            stores_raw
            .assign(county=lambda d: (
                d["county"].astype(str)
                .str.replace(" County", "", regex=False)
                .str.lower().str.strip()
            ))
            .groupby("county")
            .agg(
                total_stores      = ("fclass", "count"),
                supermarket_count = ("fclass", lambda x: (x == "supermarket").sum()),
                convenience_count = ("fclass", lambda x: (x == "convenience").sum()),
                greengrocer_count = ("fclass", lambda x: (x == "greengrocer").sum()),
            )
            .reset_index()
        )
else:
        print(
            "WARNING: retail store locations have no 'county' column. "
            "Skipping store density features. Run spatial join first."
        )
        store_agg = None
     

# Standardize joins
food_access["county"] = food_access["county"].astype(str).str.replace(" County", "", regex=False).str.lower().str.strip()
food_access["state"] = food_access["state"].astype(str).str.lower().str.strip()

census["county"] = census["county"].astype(str).str.replace(" County", "", regex=False).str.lower().str.strip()
census["state"] = census["state"].astype(str).str.lower().str.strip()

food_env["county"] = food_env["county"].astype(str).str.replace(" County", "", regex=False).str.lower().str.strip()
food_env["state"] = food_env["state"].astype(str).str.lower().str.strip()

places["state"] = "california"
places["county"] = places["county"].astype(str).str.replace(" County", "", regex=False).str.lower().str.strip()
    
#Left Join for  Merging into Master Dataset

    master = food_access.merge(
        census.drop(columns=["geo_id", "county_name",
                              "male_population", "female_population"], errors="ignore"),
        on=["county", "state"], how="left", suffixes=("", "_census"),
    ).merge(
        places,
        on=["county", "state"], how="left", suffixes=("", "_places"),
    ).merge(
        food_env,
        on=["county", "state"], how="left", suffixes=("", "_env"),
    )

    return master

        
        
#Feature Engineering

FEATURES = [
# Food access
    "urban_share",
    "avg_poverty_rate",
    "avg_median_family_income",
    "low_income_tracts_count",
    "lila_half_10_count",        # alternate LILA threshold (useful predictor)
    "lila_1_20_count",
    "total_housing_units_no_vehicle",
    "total_snap_households",
    "tract_count",
    
# Census demographics
    "total_population",
    "median_age",
    "poverty_rate",              
    "median_income",             

# Health outcomes (PLACES)
    "obesity",
    "diabetes",
    "lpa",                       
    "csmoking",
    "bphigh",
    
# Food environment
    "povrate21",
    "pct_diabetes_adults19",
    "pct_obese_adults22",
    # Engineered features (added below)
    "snap_per_capita",
    "no_vehicle_per_capita",
    "lila_density",              
]
 
# Store features added only when spatial join is present
STORE_FEATURES = [
    "store_density",
    "supermarket_share",         
]
 
 
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived ratio features and the regression target.
    All per-capita rates use population_2010 (consistent with food_access source).
    """
    pop = df["population_2010"].replace(0, np.nan)
 
# Target: proportion of tracts that are LILA (1-mile / 10-mile definition)
    df["lila_rate"] = df["lila_1_10_count"] / df["tract_count"].replace(0, np.nan)
 
# Per-capita ratios
    df["snap_per_capita"]       = df["total_snap_households"] / pop
    df["no_vehicle_per_capita"] = df["total_housing_units_no_vehicle"] / pop
 
# LILA density (redundant with target but useful as a cross-threshold feature)
    df["lila_density"] = df["lila_1_10_count"] / df["tract_count"].replace(0, np.nan)
 
# Store density features (only when retail join succeeded)
    if "total_stores" in df.columns:
        df["store_density"]      = (df["total_stores"] / pop) * 1_000
        df["supermarket_share"]  = (
            df["supermarket_count"] / df["total_stores"].replace(0, np.nan)
        )
 
    return df

#Feature Matrix

def build_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select features, impute remaining nulls with column medians, return X, y.
    """
    active_features = FEATURES.copy()
    if "store_density" in df.columns:
        active_features += STORE_FEATURES
 
# Keep only columns that actually exist post-merge
    active_features = [f for f in active_features if f in df.columns]
    missing = set(FEATURES + STORE_FEATURES) - set(active_features)
    if missing:
        print(f"INFO: features not found in master and skipped: {missing}")
 
    X = df[active_features].copy()
    y = df["lila_rate"].copy()
 
# Drop rows where target is null (shouldn't happen but guard anyway)
    valid_mask = y.notna()
    X, y = X[valid_mask], y[valid_mask]
 
# Median imputation for any remaining feature nulls
    X = X.fillna(X.median(numeric_only=True))
 
    print(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"Target range: [{y.min():.3f}, {y.max():.3f}]  mean={y.mean():.3f}")
    return X, y


#Training - Stratification based on 70% Training, 15% Validation, 15% Testing

def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.70,
    val_size: float = 0.15,
    n_bins: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series,    pd.Series,    pd.Series]:
    """
    Split into train / validation / test sets.
 
    Stratification:  lila_rate is binned into `n_bins` quartiles so that the
    full distribution of food desert severity is represented in every split.
    This is critical at N≈58 where a pure random split can easily produce
    folds that miss high-severity counties entirely.
 
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    strat_bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
 
# First cut: train vs temp (val + test)
    X_train, X_temp, y_train, y_temp, bins_train, bins_temp = train_test_split(
        X, y, strat_bins,
        test_size=(1.0 - train_size),
        random_state=RANDOM_SEED,
        stratify=strat_bins,
    )
 
#Second cut: val vs test (equal halves of temp → 15 / 15)
    val_fraction_of_temp = val_size / (1.0 - train_size)   # 0.15 / 0.30 = 0.50
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=RANDOM_SEED,
        stratify=bins_temp,
    )
 
    print(
        f"\nSplit sizes — "
        f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}"
    )
    for name, split_y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        print(f"  {name} lila_rate: mean={split_y.mean():.3f}  "
              f"std={split_y.std():.3f}  "
              f"[{split_y.min():.3f}, {split_y.max():.3f}]")
 
    return X_train, X_val, X_test, y_train, y_val, y_test


#Scale Features Included in Training

def scale_features(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit StandardScaler on train only; apply to val and test.
    Scaler is returned for persistence and inference use.
 
    Note: Random Forest is scale-invariant, but scaling is included so the
    same pipeline can be reused for linear baselines (Ridge, Lasso) without
    changes.
    """
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_val_s = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns, index=X_val.index
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_s, X_val_s, X_test_s, scaler


#Random Forest Analysis (Pre-Hypterparameter Tuning)

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
) -> RandomForestRegressor:
    """
    Fit a baseline Random Forest on the training set and evaluate on validation.
 
    Hyperparameters here are sensible defaults for N≈40 training rows.
    - n_estimators=500: more trees are always safe; diminishing returns after ~300
    - max_features="sqrt": standard for regression forests
    - min_samples_leaf=2: prevents overfitting on tiny N
    - max_depth=None: let trees grow fully; min_samples_leaf controls overfit
    - oob_score=True: free out-of-bag R² estimate on training data
 
    Tuning (GridSearchCV / RandomizedSearchCV over val set) is the next step —
    see the Random Forest analysis plan at the bottom of this file.
    """
    rf = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        max_depth=None,
        oob_score=True,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    rf.fit(X_train, y_train)
 
    val_preds = rf.predict(X_val)
    metrics = {
        "oob_r2":  round(rf.oob_score_, 4),
        "val_r2":  round(r2_score(y_val, val_preds), 4),
        "val_mae": round(mean_absolute_error(y_val, val_preds), 4),
        "val_rmse": round(mean_squared_error(y_val, val_preds) ** 0.5, 4),
    }
 
    print("\nBaseline Random Forest — validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
 
    return rf, metrics


#Random Forest After Hyperparameter-Tuning 

def evaluate_on_test(
    rf: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Run final evaluation on the test set.
    Call this ONCE after all hyperparameter decisions are locked.
    """
    test_preds = rf.predict(X_test)
    metrics = {
        "test_r2":   round(r2_score(y_test, test_preds), 4),
        "test_mae":  round(mean_absolute_error(y_test, test_preds), 4),
        "test_rmse": round(mean_squared_error(y_test, test_preds) ** 0.5, 4),
    }
 
    print("\nFinal test-set metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
 
    return metrics


#Sort Random Forest Feature Importance

def feature_importance_report(
    rf: RandomForestRegressor,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Return a sorted DataFrame of MDI feature importances.
    MDI can favour high-cardinality features; interpret alongside permutation
    importance (see Random Forest analysis plan).
    """
    imp = pd.DataFrame({
        "feature":    feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
 
    print("\nTop 10 features (MDI importance):")
    print(imp.head(10).to_string(index=False))
    return imp


#Save Analysis & Artifacts to S3 Bucket

def save_artefacts(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series,    y_val: pd.Series,    y_test: pd.Series,
    scaler:  StandardScaler,
    rf:      RandomForestRegressor,
    val_metrics:  dict,
    test_metrics: dict,
    importance_df: pd.DataFrame,
) -> None:
    """
    Save splits, scaler, model, metrics, and feature importances locally and
    upload to S3.
    """
#Split CSVs
    for split_name, X_split, y_split in [
        ("train", X_train, y_train),
        ("val",   X_val,   y_val),
        ("test",  X_test,  y_test),
    ]:
        out = X_split.copy()
        out["lila_rate"] = y_split.values
        path = PROCESSED_DIR / f"{split_name}_split.csv"
        out.to_csv(path, index=False)
        _upload_to_s3(str(path),        f"{S3_PREFIX}/data/processed/{split_name}_split.csv")
 
    #Scaler
    scaler_path = MODELS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    _upload_to_s3(str(scaler_path),  f"{S3_PREFIX}/models/scaler.joblib")
 
    #Model
    model_path = MODELS_DIR / "random_forest_baseline.joblib"
    joblib.dump(rf, model_path)
    _upload_to_s3(str(model_path),   f"{S3_PREFIX}/models/random_forest_baseline.joblib")
 
    #Metrics
    all_metrics = {"validation": val_metrics, "test": test_metrics}
    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    _upload_to_s3(str(metrics_path), f"{S3_PREFIX}/models/metrics.json")
 
    #Feature importance
    imp_path = MODELS_DIR / "feature_importances.csv"
    importance_df.to_csv(imp_path, index=False)
    _upload_to_s3(str(imp_path),     f"{S3_PREFIX}/models/feature_importances.csv")
 
    print("\nAll artefacts saved and uploaded.")
    
