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

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    for tag in TAGS:
        df[f"tag_{tag}"] = df.get(f"accuracy_by_tag.{tag}", 0)
    for mod in MODALITIES:
        df[f"modality_{mod}"] = df.get(f"accuracy_by_modality.{mod}", 0)

    df["ق_error"] = df["pronunciation_errors.ق"].apply(lambda x: 'yes' if str(x).strip().lower() in ['g', 'q'] else 'no')
    df["ج_error"] = df["pronunciation_errors.ج"].apply(lambda x: 'yes' if str(x).strip().lower() in ['j', 'ch', 'y'] else 'no')

    feature_cols = [
        "overall_accuracy",
        "phoneme_mismatch_rate",
        "modality_text",
        "modality_audio",
        "modality_speech"
    ] + [f"accuracy_{skill}" for skill in SKILLS] + ["ق_error", "ج_error"]

    dummy = pd.DataFrame([
        {
            "overall_accuracy": 0.1,
            "phoneme_mismatch_rate": 0.9,
            "modality_text": 0.9,
            "modality_audio": 0.1,
            "modality_speech": 0.2,
            "accuracy_vocabulary": 0.2,
            "accuracy_grammar": 0.3,
            "accuracy_pronunciation": 0.4,
            "accuracy_translation": 0.2,
            "accuracy_comprehension": 0.3,
            "accuracy_emotion_recognition": 0.5,
            "accuracy_politeness_register": 0.2,
            "accuracy_cultural_expression": 0.3,
            "ق_error": "yes",
            "ج_error": "yes",
            "label": "Pearl Seeker"
        },
        {
            "overall_accuracy": 0.7,
            "phoneme_mismatch_rate": 0.1,
            "modality_text": 0.3,
            "modality_audio": 0.8,
            "modality_speech": 0.2,
            "accuracy_vocabulary": 0.6,
            "accuracy_grammar": 0.5,
            "accuracy_pronunciation": 0.6,
            "accuracy_translation": 0.7,
            "accuracy_comprehension": 0.5,
            "accuracy_emotion_recognition": 0.6,
            "accuracy_politeness_register": 0.6,
            "accuracy_cultural_expression": 0.6,
            "ق_error": "no",
            "ج_error": "no",
            "label": "Reef Explorer"
        },
        {
            "overall_accuracy": 0.9,
            "phoneme_mismatch_rate": 0.05,
            "modality_text": 0.2,
            "modality_audio": 0.6,
            "modality_speech": 0.95,
            "accuracy_vocabulary": 0.9,
            "accuracy_grammar": 0.8,
            "accuracy_pronunciation": 0.95,
            "accuracy_translation": 0.85,
            "accuracy_comprehension": 0.8,
            "accuracy_emotion_recognition": 0.7,
            "accuracy_politeness_register": 0.9,
            "accuracy_cultural_expression": 0.9,
            "ق_error": "no",
            "ج_error": "no",
            "label": "Open Sea Navigator"
        }
    ])
    df = pd.concat([df, dummy], ignore_index=True)
    df = df.dropna(subset=["label"])

    X = df[feature_cols]
    y = df["label"]
    return X, y

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

def predict_learning_plan(model, input_dict):
    sanitized = {
        'overall_accuracy': float(input_dict.get('overall_accuracy', 0)),
        'phoneme_mismatch_rate': float(input_dict.get('phoneme_mismatch_rate', 0)),
        **{f'modality_{mod}': float(input_dict.get('accuracy_by_modality', {}).get(mod, 0)) for mod in MODALITIES},
        **{f'accuracy_{skill}': float(input_dict.get('accuracy_by_skill', {}).get(skill, 0)) for skill in SKILLS},
        'ق_error': 'yes' if input_dict.get('pronunciation_errors', {}).get('ق', 0) else 'no',
        'ج_error': 'yes' if input_dict.get('pronunciation_errors', {}).get('ج', 0) else 'no'
    }

    df = pd.DataFrame([sanitized])
    expected_cols = list(sanitized.keys())

    return model.predict(df[expected_cols])[0]

if __name__ == "__main__":
    csv_path = "/Users/rmg/Documents/Project/scripts/personalisation-ai/synthetic_data.csv"
    X, y = load_and_prepare_data(csv_path)
    model = train_model(X, y)
    joblib.dump(model, "personalisation_model.joblib")
    print("Model trained")
