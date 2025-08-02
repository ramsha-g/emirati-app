from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from train_model import predict_learning_plan
from fastapi.middleware.cors import CORSMiddleware
import json
import os

app = FastAPI()

# Load Learning Plan Definitions
try:
    TRACKS_PATH = os.path.join(os.path.dirname(__file__), "learning_tracks.json")
    with open(TRACKS_PATH, "r", encoding="utf-8") as f:
        LEARNING_TRACKS = json.load(f)
    print("✅ Learning tracks loaded.")
except Exception as e:
    print("❌ Failed to load learning tracks:", str(e))
    LEARNING_TRACKS = {}

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["*"],
)

# Load Trained Model
try:
    model = joblib.load("personalisation_model.joblib")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Failed to load model:", str(e))
    raise

# ------------------------
# Define Input Schema
# ------------------------
class QuizData(BaseModel):
    overall_accuracy: float
    phoneme_mismatch_rate: float
    accuracy_by_tag: dict  # Optional: kept for future use
    accuracy_by_modality: dict
    pronunciation_errors: dict

# ------------------------
# Prediction Endpoint
# ------------------------
@app.post("/predict")
def get_plan(data: QuizData):
    try:
        input_dict = data.dict()

        # ✅ Coerce modality values (text/audio/speech)
        for mod in ['text', 'audio', 'speech']:
            input_dict['accuracy_by_modality'][mod] = float(input_dict['accuracy_by_modality'].get(mod, 0.0))

        # ✅ Safe phoneme defaults
        input_dict['pronunciation_errors']['ق'] = input_dict['pronunciation_errors'].get('ق', "")
        input_dict['pronunciation_errors']['ج'] = input_dict['pronunciation_errors'].get('ج', "")

        print("✅ Processed input:", input_dict)

        label = predict_learning_plan(model, input_dict)
        print("✅ Predicted track:", label)

        return {
            "learning_plan": LEARNING_TRACKS.get(
                label,
                {"track": label, "level": "A1", "steps": ["Coming soon..."]}
            )
        }

    except Exception as e:
        print("❌ Full error:", str(e))
        raise HTTPException(status_code=400, detail=str(e))

# ------------------------
# Health & Root
# ------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health"
        }
    }

@app.get("/")
def root():
    return {"message": "API is running", "docs": "/docs"}
