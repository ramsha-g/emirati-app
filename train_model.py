import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

MODALITIES = ["text", "audio", "speech"]
TAGS = ["greeting", "food", "travel", "shopping", "office"]
SKILLS = [
    "vocabulary", "grammar", "pronunciation", "translation", "comprehension",
    "emotion_recognition", "politeness_register", "cultural_expression"
]

def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    original_rows = len(df)

    for tag in TAGS:
        df[f"tag_{tag}"] = df.get(f"accuracy_by_tag.{tag}", 0)
    for mod in MODALITIES:
        df[f"modality_{mod}"] = df.get(f"accuracy_by_modality.{mod}", 0)

    def safe_float(x):
        try:
            return float(x)
        except:
            return 0.0

    df["ق_error"] = df["pronunciation_errors.ق"].apply(lambda x: 'yes' if safe_float(x) > 0.3 else 'no')
    df["ج_error"] = df["pronunciation_errors.ج"].apply(lambda x: 'yes' if safe_float(x) > 0.3 else 'no')

    # 💡 Define feature_cols BEFORE using it
    feature_cols = [
        "overall_accuracy",
        "phoneme_mismatch_rate",
        "modality_text",
        "modality_audio",
        "modality_speech"
    ] + [f"accuracy_{skill}" for skill in SKILLS] + ["ق_error", "ج_error"]

    # Add dummy data
    dummy = pd.DataFrame([...])  # (same as before)
    df = pd.concat([df, dummy], ignore_index=True)

    final_rows = len(df)
    df = df.dropna(subset=["label"])
    rows_used = len(df)
    skipped = final_rows - rows_used

    print(f"✅ Original rows: {original_rows}")
    print(f"✅ Added dummy rows: {len(dummy)}")
    print(f"❌ Skipped rows (missing label): {skipped}")
    print(f"✅ Final rows used for training: {rows_used}")

    X = df[feature_cols]
    y = df["label"]
    return X, y, feature_cols

def train_model(X, y, feature_cols):
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
    return pipeline, feature_cols

def predict_learning_plan(model_bundle, input_dict):
    model = model_bundle["model"]
    expected_cols = model_bundle["meta"]["features"]

    sanitized = {
        'overall_accuracy': float(input_dict.get('overall_accuracy', 0)),
        'phoneme_mismatch_rate': float(input_dict.get('phoneme_mismatch_rate', 0)),
        **{f'modality_{mod}': float(input_dict.get('accuracy_by_modality', {}).get(mod, 0)) for mod in MODALITIES},
        **{f'accuracy_{skill}': float(input_dict.get('accuracy_by_skill', {}).get(skill, 0)) for skill in SKILLS},
        'ق_error': 'yes' if input_dict.get('pronunciation_errors', {}).get('ق', 0) > 0.3 else 'no',
        'ج_error': 'yes' if input_dict.get('pronunciation_errors', {}).get('ج', 0) > 0.3 else 'no'
    }

    df = pd.DataFrame([sanitized])
    return model.predict(df[expected_cols])[0]

if __name__ == "__main__":
    csv_path = "/Users/rmg/Documents/Transcribe/personlisation/synthetic_data_with_skills.csv"  # ✅ Update this path if needed
    X, y, feature_cols = load_and_prepare_data(csv_path)
    model, feature_cols = train_model(X, y, feature_cols)

    model_bundle = {
        "model": model,
        "meta": {
            "version": "1.0",
            "features": feature_cols
        }
    }

    joblib.dump(model_bundle, "personalisation_model.joblib")
    print("✅ Model trained and saved as personalisation_model.joblib")
