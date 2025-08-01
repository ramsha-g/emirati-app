import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Constants
TAGS = ["greeting", "food", "travel", "shopping", "office"]
MODALITIES = ["text", "audio", "speech"]

# -------------------------------
# Load and prepare training data
# -------------------------------
def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    # Flatten nested metrics
    for tag in TAGS:
        df[f"tag_{tag}"] = df[f"accuracy_by_tag.{tag}"]
    for mod in MODALITIES:
        df[f"modality_{mod}"] = df[f"accuracy_by_modality.{mod}"]

    # Ensure errors are integer type
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

# -------------------
# Train model
# -------------------
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

# -------------------
# Predict single input
# -------------------
def predict_learning_plan(model, single_input_dict):
    # Helper to safely convert to float
    def safe_float(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0

    # Defensive copy of input
    input_dict = dict(single_input_dict)

    # Convert top-level fields
    overall_accuracy = safe_float(input_dict.get('overall_accuracy', 0))
    phoneme_mismatch_rate = safe_float(input_dict.get('phoneme_mismatch_rate', 0))

    # Extract nested tag accuracy
    tag_accuracy = {
        f"tag_{tag}": safe_float(input_dict.get('accuracy_by_tag', {}).get(tag, 0.5))
        for tag in TAGS
    }

    # Extract nested modality accuracy
    modality_accuracy = {
        f"modality_{mod}": safe_float(input_dict.get('accuracy_by_modality', {}).get(mod, 0.5))
        for mod in MODALITIES
    }

    # Extract pronunciation errors
    ق_error = int(input_dict.get('pronunciation_errors', {}).get('ق', 0))
    ج_error = int(input_dict.get('pronunciation_errors', {}).get('ج', 0))

    # Create input DataFrame
    df_input = pd.DataFrame([{
        'overall_accuracy': overall_accuracy,
        'phoneme_mismatch_rate': phoneme_mismatch_rate,
        **tag_accuracy,
        **modality_accuracy,
        'ق_error': ق_error,
        'ج_error': ج_error
    }])

    # Ensure all columns match
    feature_cols = (
        ["overall_accuracy", "phoneme_mismatch_rate"] +
        list(tag_accuracy.keys()) +
        list(modality_accuracy.keys()) +
        ["ق_error", "ج_error"]
    )
    for col in feature_cols:
        if col not in df_input:
            df_input[col] = 0.0

    # Predict
    try:
        prediction = model.predict(df_input[feature_cols])[0]
        return prediction
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        print("Input DataFrame:\n", df_input[feature_cols])
        raise

# -------------------
# Run training
# -------------------
if __name__ == "__main__":
    csv_path = "synthetic_data.csv"
    X, y = load_and_prepare_data(csv_path)
    print("✅ Data loaded. Number of samples:", len(X))

    model = train_model(X, y)
    print("✅ Model trained.")

    joblib.dump(model, "personalisation_model.joblib")
    print("✅ Model saved to personalisation_model.joblib.")
