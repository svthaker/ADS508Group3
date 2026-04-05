from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def train_logistic_regression(X_train, y_train, **kwargs):
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight="balanced",
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, **kwargs):
    model = DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced",
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, tuned=True, **kwargs):
    if tuned:
        model = RandomForestClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            **kwargs
        )
    else:
        model = RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
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