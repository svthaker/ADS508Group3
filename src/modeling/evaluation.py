import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

def evaluate_classifier(model, X_test, y_test, feature_names=None, positive_label=1):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision_class_1": precision_score(y_test, y_pred, pos_label=positive_label, zero_division=0),
        "recall_class_1": recall_score(y_test, y_pred, pos_label=positive_label, zero_division=0),
        "f1_class_1": f1_score(y_test, y_pred, pos_label=positive_label, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }

    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Low Risk", "High Risk"]
    )
    plt.title("Confusion Matrix")
    plt.show()

    coef_df = None
    if feature_names is not None and hasattr(model, "coef_"):
        coef_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": model.coef_[0]
        }).sort_values(by="Coefficient", ascending=False)

    return metrics, coef_df