from fastapi import FastAPI, HTTPException  # Added HTTPException
from pydantic import BaseModel
import joblib
from train_model import predict_learning_plan
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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
        print("📥 Received input:", input_dict)
        
        # For debugging, you can uncomment this to test without the model:
        # return {"plan": "focus_on_greeting"}
        
        label = predict_learning_plan(model, input_dict)
        return {"plan": label}
    except Exception as e:
        print("❌ Prediction error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

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