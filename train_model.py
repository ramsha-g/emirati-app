import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Constants
MODALITIES = ["text", "audio", "speech"]
TAGS = ["greeting", "food", "travel", "shopping", "office"]  # kept for future

# -------------------------------
# Load and prepare training data
# -------------------------------
def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    # ✅ Optional: keep tag-based columns for future but drop from features
    for tag in TAGS:
        df[f"tag_{tag}"] = df[f"accuracy_by_tag.{tag}"]

    for mod in MODALITIES:
        df[f"modality_{mod}"] = df[f"accuracy_by_modality.{mod}"]

    # ✅ Map string errors to binary flags
    df["ق_error"] = df["pronunciation_errors.ق"].apply(lambda x: 'yes' if str(x).strip().lower() in ['g', 'q'] else 'no')
    df["ج_error"] = df["pronunciation_errors.ج"].apply(lambda x: 'yes' if str(x).strip().lower() in ['j', 'ch', 'y'] else 'no')

    # ✅ Map labels to plan names
    LABEL_MAPPING = {
        "start_from_pronunciation_basics": "Pearl Seeker",
        "focus_on_greeting": "Reef Explorer",
        "focus_on_food": "Open Sea Navigator"
    }
    df["label"] = df["label"].map(LABEL_MAPPING)

    # ✅ Add dummy data to cover edge cases
    dummy = pd.DataFrame([
        {
            "overall_accuracy": 0.0,
            "phoneme_mismatch_rate": 0.9,
            "modality_text": 0.2,
            "modality_audio": 0.3,
            "modality_speech": 0.2,
            "ق_error": "yes",
            "ج_error": "yes",
            "label": "Pearl Seeker"
        },
        {
            "overall_accuracy": 1.0,
            "phoneme_mismatch_rate": 0.0,
            "modality_text": 0.9,
            "modality_audio": 0.9,
            "modality_speech": 0.95,
            "ق_error": "no",
            "ج_error": "no",
            "label": "Open Sea Navigator"
        }
    ])
    df = pd.concat([df, dummy], ignore_index=True)

    # ✅ Use only modality and phoneme features
    feature_cols = (
        ["overall_accuracy", "phoneme_mismatch_rate"] +
        [f"modality_{mod}" for mod in MODALITIES] +
        ["ق_error", "ج_error"]
    )
    X = df[feature_cols]
    y = df["label"]
    return X, y

# -------------------
# Train model
# -------------------
def train_model(X, y):
    categorical_features = ["ق_error", "ج_error"]
    numerical_features = [col for col in X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X, y)
    return pipeline

# -------------------
# Predict single input
# -------------------
def predict_learning_plan(model, input_dict):
    sanitized = {
        'overall_accuracy': float(input_dict.get('overall_accuracy', 0)),
        'phoneme_mismatch_rate': float(input_dict.get('phoneme_mismatch_rate', 0)),
        **{f'modality_{mod}': float(input_dict.get('accuracy_by_modality', {}).get(mod, 0))
            for mod in MODALITIES},
        'ق_error': 'yes' if input_dict.get('pronunciation_errors', {}).get('ق', 0) else 'no',
        'ج_error': 'yes' if input_dict.get('pronunciation_errors', {}).get('ج', 0) else 'no'
    }

    df = pd.DataFrame([sanitized])

    expected_cols = [
        'overall_accuracy',
        'phoneme_mismatch_rate',
        'modality_text',
        'modality_audio',
        'modality_speech',
        'ق_error',
        'ج_error'
    ]

    print("✅ Sanitized input types:", {k: type(v) for k, v in sanitized.items()})
    print("✅ DataFrame dtypes:\n", df.dtypes)

    try:
        return model.predict(df[expected_cols])[0]
    except Exception as e:
        print(f"❌ Model prediction failed. Input:\n{df[expected_cols]}")
        raise ValueError(f"Prediction failed: {str(e)}") from e

# -------------------
# Run training
# -------------------
if __name__ == "__main__":
    csv_path = "/Users/rmg/Documents/Transcribe/personlisation/synthetic_data.csv"
    X, y = load_and_prepare_data(csv_path)
    print("✅ Data loaded. Number of samples:", len(X))

    model = train_model(X, y)
    print("✅ Model trained.")

    joblib.dump(model, "personalisation_model.joblib")
    print("✅ Model saved to personalisation_model.joblib.")
