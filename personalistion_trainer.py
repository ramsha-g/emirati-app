import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

TAGS = ["greeting", "food", "travel", "shopping", "office"]
MODALITIES = ["text", "audio", "speech"]
PHONEME_VARIANTS = {
    "ق": ["g", "q"],
    "ج": ["ch", "j", "y"]
}

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    for tag in TAGS:
        df[f"tag_{tag}"] = df[f"accuracy_by_tag.{tag}"]
    for mod in MODALITIES:
        df[f"modality_{mod}"] = df[f"accuracy_by_modality.{mod}"]

    df["ق_error"] = df["pronunciation_errors.ق"]
    df["ج_error"] = df["pronunciation_errors.ج"]

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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),
            ("cat", OneHotEncoder(), categorical_features)
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X, y)
    return pipeline

def predict_learning_plan(model, single_input_dict):
    df_input = pd.json_normalize([single_input_dict])
    
    for tag in TAGS:
        df_input[f"tag_{tag}"] = df_input.get(f"accuracy_by_tag.{tag}", 0.5)
    for mod in MODALITIES:
        df_input[f"modality_{mod}"] = df_input.get(f"accuracy_by_modality.{mod}", 0.5)
    
    df_input["ق_error"] = int(df_input.get("pronunciation_errors.ق", 0))
    df_input["ج_error"] = int(df_input.get("pronunciation_errors.ج", 0))

    feature_cols = (
        ["overall_accuracy", "phoneme_mismatch_rate"] +
        [f"tag_{tag}" for tag in TAGS] +
        [f"modality_{mod}" for mod in MODALITIES] +
        ["ق_error", "ج_error"]
    )

    return model.predict(df_input[feature_cols])[0]

# ✅ Example usage
csv_path = "/Users/rmg/Documents/Transcribe/personlisation/synthetic_data.csv"
X, y = load_and_prepare_data(csv_path)
print("✅ Data loaded. Number of samples:", len(X))

model = train_model(X, y)
print("✅ Model trained.")

joblib.dump(model, "personalisation_model.joblib")
print("✅ Model saved to personalisation_model.joblib.")
