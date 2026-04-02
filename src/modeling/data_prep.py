from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_model_data(features):
    X = features.drop(columns=[
        "state",
        "county",
        "high_food_access_risk",
        "low_income_tracts_count",
        "lila_1_10_count"
    ])
    y = features["high_food_access_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler
    }