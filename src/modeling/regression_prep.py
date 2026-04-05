import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_regression_data(features):
    features = features.copy()
    features.columns = features.columns.str.strip()

    required_cols = [
        "state",
        "county",
        "lila_rate",
        "low_income_tracts_count",
        "lila_1_10_count"
    ]
    missing = [col for col in required_cols if col not in features.columns]
    if missing:
        raise ValueError(f"Missing required columns for regression: {missing}")

    X = features.drop(columns=[
        "state",
        "county",
        "high_food_access_risk",   # drop if present
        "lila_rate",
        "low_income_tracts_count",
        "lila_1_10_count"
    ], errors="ignore")

    y = features["lila_rate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }