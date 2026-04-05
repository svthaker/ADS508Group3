from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest_regressor(X_train, y_train, tuned=True, **kwargs):
    if tuned:
        model = RandomForestRegressor(
            random_state=42,
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            max_depth=None,
            n_jobs=-1,
            **kwargs
        )
    else:
        model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **kwargs
        )

    model.fit(X_train, y_train)
    return model


def train_gradient_boosting_regressor(X_train, y_train, tuned=True, **kwargs):
    if tuned:
        model = GradientBoostingRegressor(
            random_state=42,
            learning_rate=0.1,
            max_depth=2,
            n_estimators=100
        )
    else:
        model = GradientBoostingRegressor(
            random_state=42
        )

    model.fit(X_train, y_train)
    return model