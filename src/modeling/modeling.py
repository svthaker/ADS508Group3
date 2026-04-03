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

def train_gradient_boosting(X_train, y_train, tuned=True):
    if tuned:
        model = GradientBoostingClassifier(
            random_state=42,
            learning_rate=0.1,
            max_depth=2,
            min_samples_leaf=3,
            min_samples_split=2,
            n_estimators=50
        )
    else:
        model = GradientBoostingClassifier(
            random_state=42
        )

    model.fit(X_train, y_train)
    return model