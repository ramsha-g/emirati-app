# Fixing the predict_learning_plan in train_model.py to ensure safe data types

import pandas as pd 
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
def predict_learning_plan(model, single_input_dict):
    # Create a defensive copy to avoid modifying original
    input_dict = {k: v for k, v in single_input_dict.items()}
    
    # Type conversion with error handling
    def safe_float(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0  # Default value if conversion fails
    
    # Convert numeric fields
    input_dict['overall_accuracy'] = safe_float(input_dict.get('overall_accuracy', 0))
    input_dict['phoneme_mismatch_rate'] = safe_float(input_dict.get('phoneme_mismatch_rate', 0))
    
    # Convert nested dictionaries
    for tag in TAGS:
        input_dict['accuracy_by_tag'][tag] = safe_float(
            input_dict.get('accuracy_by_tag', {}).get(tag, 0.5)
    
    for mod in MODALITIES:
        input_dict['accuracy_by_modality'][mod] = safe_float(
            input_dict.get('accuracy_by_modality', {}).get(mod, 0.5))
    
    # Prepare DataFrame with explicit type conversion
    df_input = pd.DataFrame({
        'overall_accuracy': [input_dict['overall_accuracy']],
        'phoneme_mismatch_rate': [input_dict['phoneme_mismatch_rate']],
        **{f'tag_{tag}': [input_dict['accuracy_by_tag'][tag]] for tag in TAGS},
        **{f'modality_{mod}': [input_dict['accuracy_by_modality'][mod]] for mod in MODALITIES},
        'ق_error': [int(input_dict.get('pronunciation_errors', {}).get('ق', 0))],
        'ج_error': [int(input_dict.get('pronunciation_errors', {}).get('ج', 0))]
    })
    
    # Ensure all feature columns exist
    feature_cols = [
        'overall_accuracy',
        'phoneme_mismatch_rate',
        *[f'tag_{tag}' for tag in TAGS],
        *[f'modality_{mod}' for mod in MODALITIES],
        'ق_error',
        'ج_error'
    ]
    
    # Verify feature columns match training
    missing_cols = set(feature_cols) - set(df_input.columns)
    if missing_cols:
        for col in missing_cols:
            df_input[col] = 0.0  # Add missing columns with default value
    
    try:
        return model.predict(df_input[feature_cols])[0]
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        print("Input DataFrame:\n", df_input[feature_cols])
        raise

if __name__ == "__main__":
    csv_path = "synthetic_data.csv"
    X, y = load_and_prepare_data(csv_path)
    print("✅ Data loaded. Number of samples:", len(X))

    model = train_model(X, y)
    print("✅ Model trained.")

    joblib.dump(model, "personalisation_model.joblib")
    print("✅ Model saved to personalisation_model.joblib.")
