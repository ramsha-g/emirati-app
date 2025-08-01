from fastapi import FastAPI, HTTPException  # Added HTTPException
from pydantic import BaseModel
import joblib
from train_model import predict_learning_plan
from fastapi.middleware.cors import CORSMiddleware
import json
import os

app = FastAPI()

try:
    TRACKS_PATH = os.path.join(os.path.dirname(__file__), "learning_tracks.json")
    with open(TRACKS_PATH, "r", encoding="utf-8") as f:
        LEARNING_TRACKS = json.load(f)
    print("✅ Learning tracks loaded.")
except Exception as e:
    print("❌ Failed to load learning tracks:", str(e))
    LEARNING_TRACKS = {}

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS", "GET"],  # Added GET for health check
    allow_headers=["*"],
)

# Model Loading
try:
    model = joblib.load("personalisation_model.joblib")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Failed to load model:", str(e))
    raise

# Data Models
class QuizData(BaseModel):
    overall_accuracy: float
    phoneme_mismatch_rate: float
    accuracy_by_tag: dict
    accuracy_by_modality: dict
    pronunciation_errors: dict

# Routes
@app.post("/predict")
def get_plan(data: QuizData):
    try:
        input_dict = data.dict()
        
        # Convert all numeric inputs to float explicitly
        input_dict['overall_accuracy'] = float(input_dict['overall_accuracy'])
        input_dict['phoneme_mismatch_rate'] = float(input_dict['phoneme_mismatch_rate'])
        
        # Convert nested dictionaries
        for tag in ['greeting', 'food', 'travel', 'shopping', 'office']:
            input_dict['accuracy_by_tag'][tag] = float(input_dict['accuracy_by_tag'].get(tag, 0))
            
        for mod in ['text', 'audio', 'speech']:
            input_dict['accuracy_by_modality'][mod] = float(input_dict['accuracy_by_modality'].get(mod, 0))
            
        print("✅ Processed input:", input_dict)  # Debug log
        
        label = predict_learning_plan(model, input_dict)
        print("✅ Predicted track:", label)  # Debug
        return {
            "learning_plan": LEARNING_TRACKS.get(
                label,
                {"track": label, "level": "A1", "steps": ["Coming soon..."]}
            )
        }

        
    except Exception as e:
        print("❌ Full error:", str(e))
        raise HTTPException(status_code=400, detail=str(e))

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

# Add proper root endpoint
@app.get("/")
def root():
    return {"message": "API is running", "docs": "/docs"}

