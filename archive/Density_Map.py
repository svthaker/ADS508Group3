#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 11:08:58 2026

@author: jamesshoenhair
"""
import boto3
import sagemaker
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# Using Preprocessing Pipeline
from preprocessing import prepare_data, load_and_scrub_data, select_features, create_features

S3_BUCKET = "projects-sagemaker-ml-pipeline-787172416632-us-east-1-an"
S3_PREFIX = "food-access-pipeline"

PROCESSED_DIR = f"s3://{S3_BUCKET}/{S3_PREFIX}/data/processed"
SHAPEFILE_DIR  = f"s3://{S3_BUCKET}/{S3_PREFIX}/data/shapefiles"  # California county boundaries

#Load Data

def load_spatial_data() -> gpd.GeoDataFrame:         
    s3 = boto3.client("s3")                  
    
#Download processed CSVs from S3
    local_processed_dir = Path("data/processed")
    local_processed_dir.mkdir(parents=True, exist_ok=True)

    for fname in [
        "food_access_county_clean.csv",
        "census_features_clean.csv",
        "places_ca_health_clean.csv",
    ]:
        s3.download_file(
            Bucket=S3_BUCKET,
            Key=f"{S3_PREFIX}/data/processed/{fname}",
            Filename=str(local_processed_dir / fname),
        )
    print("[S3] Processed CSVs downloaded to data/processed")

#Download shapefile components          
    local_shp_dir = Path("/tmp/shapefiles")        
    local_shp_dir.mkdir(exist_ok=True)
    
    for ext in ["shp", "shx", "dbf", "prj"]:      
        s3.download_file(
            Bucket=S3_BUCKET,
            Key=f"{S3_PREFIX}/shapefiles/ca_counties.{ext}",
            Filename=str(local_shp_dir / f"ca_counties.{ext}")
        )
    
    ca_counties = gpd.read_file(local_shp_dir / "ca_counties.shp")

# Pull raw merged data (pre-split, with county identifiers intact)
    df_raw = load_and_scrub_data()
    df_selected = select_features(df_raw)
    df_model = create_features(df_raw, df_selected)

# Retain county identifier for spatial join
    df_model["county"] = df_raw["county"]

# Load CA county shapefile (e.g., from Census TIGER/Line)
    ca_counties["county"] = ca_counties["NAME"].str.lower().str.strip()
    df_model["county"]    = df_model["county"].str.lower().str.strip()

# Spatial join
    gdf = ca_counties.merge(df_model, on="county", how="left")
    gdf = gdf.set_crs("EPSG:4326").to_crs("EPSG:3310")  # CA Albers projection

    print(f"[Spatial] GeoDataFrame shape: {gdf.shape}")
    return gdf


#Density Metrics

def compute_density_metrics(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute per-area density metrics for choropleth mapping.
    Extends your existing engineered features with spatial density.
    """

    gdf["area_km2"] = gdf.geometry.area / 1e6  # Convert m² to km²

# Population density
    gdf["pop_density"]          = gdf["total_population"] / gdf["area_km2"]

# SNAP household density (extends your snap_per_capita feature)
    gdf["snap_density"]         = gdf["total_snap_households"] / gdf["area_km2"]

# No-vehicle household density (extends your hunv_per_capita feature)
    gdf["hunv_density"]         = gdf["total_housing_units_no_vehicle"] / gdf["area_km2"]

# LILA tract density (low income + low access)
    gdf["lila_density"]         = gdf["lila_1_10_count"] / gdf["area_km2"]

# Composite food access risk score (normalized 0–1)
    risk_cols = ["poverty_rate", "snap_per_capita", "hunv_per_capita", "lila_density"]
    gdf["composite_risk_score"] = (
        gdf[risk_cols]
        .apply(lambda col: (col - col.min()) / (col.max() - col.min()))
        .mean(axis=1)
    )

    return gdf


#cloropleth Plot

def plot_choropleth(
    gdf:        gpd.GeoDataFrame,
    column:     str,
    title:      str,
    cmap:       str   = "YlOrRd",
    output_path: Path = None,
):
    """
    Render a single-metric choropleth density map of California counties.
    """

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    gdf.plot(
        column=column,
        ax=ax,
        legend=True,
        cmap=cmap,
        edgecolor="white",
        linewidth=0.5,
        legend_kwds={
            "label": title,
            "orientation": "horizontal",
            "shrink": 0.6,
            "pad": 0.02,
        },
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )

# Annotate high-risk counties
    for _, row in gdf.iterrows():
        if row[column] > gdf[column].quantile(0.85):
            ax.annotate(
                text=row["county"].title(),
                xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                fontsize=6,
                ha="center",
                color="black",
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_axis_off()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {output_path}")

    plt.show()


#CA Dashboard/Map Visualization

def plot_dashboard(gdf: gpd.GeoDataFrame, output_path: Path = None):
    """
    Side-by-side choropleth panels for all key density metrics,
    plus the binary high_food_access_risk target from your pipeline.
    """

    metrics = [
        ("pop_density",          "Population Density\n(per km²)",         "Blues"),
        ("snap_density",         "SNAP Household Density\n(per km²)",      "Greens"),
        ("hunv_density",         "No-Vehicle HH Density\n(per km²)",       "Oranges"),
        ("poverty_rate",         "Poverty Rate (%)",                       "Reds"),
        ("composite_risk_score", "Composite Food Access\nRisk Score",      "YlOrRd"),
        ("high_food_access_risk","High Food Access Risk\n(Binary Target)", "RdYlGn_r"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 14))
    fig.suptitle(
        "California County Food Access Risk — Density Analysis",
        fontsize=16, fontweight="bold", y=1.01
    )

    for ax, (col, label, cmap) in zip(axes.flatten(), metrics):
        gdf.plot(
            column=col,
            ax=ax,
            cmap=cmap,
            edgecolor="white",
            linewidth=0.4,
            legend=True,
            missing_kwds={"color": "lightgrey"},
        )
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_axis_off()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[Dashboard] Saved → {output_path}")

    plt.show()


#CA Health Outcomes

def plot_health_overlay(gdf: gpd.GeoDataFrame, output_path: Path = None):
    """
    Scatter-style bubble map overlaying health outcomes (obesity, diabetes)
    from your PLACES features on top of the composite risk choropleth.
    Bubble size = obesity prevalence; color = diabetes prevalence.
    """

    fig, ax = plt.subplots(figsize=(10, 12))

# Base layer — composite risk choropleth
    gdf.plot(
        column="composite_risk_score",
        ax=ax,
        cmap="YlOrRd",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.75,
        legend=False,
    )

# Bubble overlay — health outcomes
    centroids = gdf.geometry.centroid
    scatter = ax.scatter(
        x=centroids.x,
        y=centroids.y,
        s=gdf["obesity"] * 8,            
        c=gdf["diabetes"],               
        cmap="coolwarm",
        alpha=0.65,
        edgecolors="black",
        linewidths=0.4,
    )

    plt.colorbar(scatter, ax=ax, label="Diabetes Prevalence (%)", shrink=0.5)
    ax.set_title(
        "Health Outcome Overlay on Food Access Risk\n"
        "(Bubble size = Obesity Rate)",
        fontsize=13, fontweight="bold"
    )
    ax.set_axis_off()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.show()


#Density Analysis (By County)

def run_density_analysis():

    OUTPUT_DIR = Path("outputs/maps")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#Spatial data
    gdf = load_spatial_data()

#Density metrics
    gdf = compute_density_metrics(gdf)

#Individual choropleths
    plot_choropleth(
        gdf, "composite_risk_score",
        "Composite Food Access Risk Score — California Counties",
        output_path=OUTPUT_DIR / "composite_risk.png"
    )

#Full dashboard
    plot_dashboard(gdf, output_path=OUTPUT_DIR / "dashboard.png")

#Health outcome overlay
    plot_health_overlay(gdf, output_path=OUTPUT_DIR / "health_overlay.png")

    print("\n[Analysis] Density map analysis complete.")


if __name__ == "__main__":
    run_density_analysis()