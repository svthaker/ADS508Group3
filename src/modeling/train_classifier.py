import os
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier

if __name__ == "__main__":
    input_path = os.environ["SM_CHANNEL_TRAIN"]
    print("Training input path:", input_path)
    print("Files in training channel:", os.listdir(input_path))

    file_name = os.listdir(input_path)[0]
    file_path = os.path.join(input_path, file_name)

    df = pd.read_csv(file_path)
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)

    # Drop non-numeric ID/text columns if they are still present
    drop_cols = [col for col in ["state", "county"] if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    target = "high_food_access_risk"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in training data.")

    X = df.drop(columns=[target])
    y = df[target]

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)

    model_dir = os.environ["SM_MODEL_DIR"]
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))

    print("Model training complete.")