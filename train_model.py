import pandas as pd 
from . import TAGS, MODALITIES 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

TAGS = ["greeting", "food", "travel", "shopping", "office"]
MODALITIES = ["text", "audio", "speech"]

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    for tag in TAGS:
        df[f"tag_{tag}"] = df[f"accuracy_by_tag.{tag}"]
    for mod in MODALITIES:
        df[f"modality_{mod}"] = df[f"accuracy_by_modality.{mod}"]

    df["ق_error"] = df["pronunciation_errors.ق"].astype(int)
    df["ج_error"] = df["pronunciation_errors.ج"].astype(int)

    feature_cols = (
        ["overall_accuracy", "phoneme_mismatch_rate"] +
        [f"tag_{tag}" for tag in TAGS] +
        [f"modality_{mod}" for mod in MODALITIES] +
        ["ق_error", "ج_error"]
    )
    X = df[feature_cols]
    y = df["label"]
    return X, y

def train_model(X, y):
    categorical_features = ["ق_error", "ج_error"]
    numerical_features = [col for col in X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(), categorical_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X, y)
    return pipeline

# Fixing the predict_learning_plan in train_model.py to ensure safe data types

def predict_learning_plan(model, single_input_dict):

    df_input = pd.json_normalize([single_input_dict])

    # Add default 0.5 values if keys are missing
    for tag in TAGS:
        df_input[f"tag_{tag}"] = single_input_dict.get("accuracy_by_tag", {}).get(tag, 0.5)
    for mod in MODALITIES:
        df_input[f"modality_{mod}"] = single_input_dict.get("accuracy_by_modality", {}).get(mod, 0.5)

    # Safely convert phoneme error values to integers (0 or 1)
    try:
        df_input["ق_error"] = int(single_input_dict.get("pronunciation_errors", {}).get("ق", 0))
    except (ValueError, TypeError):
        df_input["ق_error"] = 0

    try:
        df_input["ج_error"] = int(single_input_dict.get("pronunciation_errors", {}).get("ج", 0))
    except (ValueError, TypeError):
        df_input["ج_error"] = 0

    feature_cols = (
        ["overall_accuracy", "phoneme_mismatch_rate"] +
        [f"tag_{tag}" for tag in TAGS] +
        [f"modality_{mod}" for mod in MODALITIES] +
        ["ق_error", "ج_error"]
    )

    return model.predict(df_input[feature_cols])[0]



# Run this script manually to retrain and save model
if __name__ == "__main__":
    csv_path = "synthetic_data.csv"  # replace with your actual CSV path
    X, y = load_and_prepare_data(csv_path)
    print("✅ Data loaded. Number of samples:", len(X))

    model = train_model(X, y)
    print("✅ Model trained.")

    joblib.dump(model, "personalisation_model.joblib")
    print("✅ Model saved to personalisation_model.joblib.")