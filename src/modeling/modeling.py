from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


def train_logistic_regression(X_train, y_train, **kwargs):
    # Use scaled features
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight="balanced",
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train, **kwargs):
    # Use unscaled features
    model = GradientBoostingClassifier(
        random_state=42,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model