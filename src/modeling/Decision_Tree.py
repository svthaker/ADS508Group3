#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 16:59:01 2026

@author: jamesshoenhair
"""

import sagemaker, boto3
from sagemaker import Session
from sagemaker.sklearn import SKLearn
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, CategoricalParameter

# --- S3 Upload ---
sess   = Session()
bucket = "projects-sagemaker-ml-pipeline-787172416632-us-east-1-an"
prefix = "food-access-pipeline"

sess.upload_data("/tmp/X_train.csv", bucket=bucket, key_prefix=f"{prefix}/train")
sess.upload_data("/tmp/y_train.csv", bucket=bucket, key_prefix=f"{prefix}/train")
sess.upload_data("/tmp/X_val.csv",   bucket=bucket, key_prefix=f"{prefix}/validation")
sess.upload_data("/tmp/y_val.csv",   bucket=bucket, key_prefix=f"{prefix}/validation")
sess.upload_data("/tmp/X_test.csv",  bucket=bucket, key_prefix=f"{prefix}/test")
sess.upload_data("/tmp/y_test.csv",  bucket=bucket, key_prefix=f"{prefix}/test")

train_s3 = f"s3://{bucket}/{prefix}/train"
val_s3   = f"s3://{bucket}/{prefix}/validation"
test_s3  = f"s3://{bucket}/{prefix}/test"

# --- Estimator ---
estimator = SKLearn(
    entry_point="train.py",
    framework_version="1.2-1",
    instance_type="ml.m5.large",
    instance_count=1,
    role=sagemaker.get_execution_role(),
    hyperparameters={
        "max_depth": 5, "min_samples_split": 5,
        "min_samples_leaf": 2, "class_weight": "balanced"
    },
)

# --- Tuner ---
tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name="validation:f1",
    metric_definitions=[{"Name": "validation:f1",
                         "Regex": "validation:f1=([0-9\\.]+)"}],
    hyperparameter_ranges={
        "max_depth":          IntegerParameter(2, 10),
        "min_samples_split":  IntegerParameter(2, 20),
        "min_samples_leaf":   IntegerParameter(1, 10),
        "class_weight":       CategoricalParameter(["None", "balanced"]),
    },
    max_jobs=20,
    max_parallel_jobs=4,
    objective_type="Maximize",
)

tuner.fit({
    "train":      train_s3,
    "validation": val_s3,
    "test":       test_s3,
})