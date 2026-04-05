import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regressor(model, X_test, y_test, feature_names=None):
    y_pred = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
    }

    print("Regression Metrics")
    print("------------------")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

    importance_df = None
    if feature_names is not None:
        if hasattr(model, "coef_"):
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Value": model.coef_
            }).sort_values(by="Value", ascending=False)

            print("\nTop Coefficients:")
            print(importance_df.head(10))

        elif hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Value": model.feature_importances_
            }).sort_values(by="Value", ascending=False)

            print("\nTop Feature Importances:")
            print(importance_df.head(10))

    return metrics, importance_df