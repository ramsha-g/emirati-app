from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
from train_model import predict_learning_plan
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from fastapi.staticfiles import StaticFiles
from typing import List


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


try:
    TRACKS_PATH = os.path.join(os.path.dirname(__file__), "learning_tracks.json")
    with open(TRACKS_PATH, "r", encoding="utf-8") as f:
        LEARNING_TRACKS = json.load(f)
    print("Learning tracks loaded.")
except Exception as e:
    print("Failed to load learning tracks:", str(e))
    LEARNING_TRACKS = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["*"],
)

try:
    model = joblib.load("personalisation_model.joblib")
    print("Model loaded successfully.")
except Exception as e:
    print("Failed to load model:", str(e))
    raise

class QuizData(BaseModel):
    overall_accuracy: float
    phoneme_mismatch_rate: float
    accuracy_by_tag: dict
    accuracy_by_modality: dict
    pronunciation_errors: dict
    accuracy_by_skill: dict

@app.post("/predict")
def get_plan(data: QuizData):
    try:
        input_dict = data.dict()

        for mod in ['text', 'audio', 'speech']:
            input_dict['accuracy_by_modality'][mod] = float(input_dict['accuracy_by_modality'].get(mod, 0.0))

        input_dict['pronunciation_errors']['ق'] = input_dict['pronunciation_errors'].get('ق', "")
        input_dict['pronunciation_errors']['ج'] = input_dict['pronunciation_errors'].get('ج', "")

        label = predict_learning_plan(model, input_dict)

        return {
            "learning_plan": LEARNING_TRACKS.get(
                label,
                {"track": label, "level": "A1", "steps": ["Coming soon..."]}
            )
        }

    except Exception as e:
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

@app.get("/")
def root():
    return {"message": "API is running", "docs": "/docs"}


class AnswerEntry(BaseModel):
    id: str
    user_answer: str = None
    filename: str = None

@app.post("/upload-answers")
async def upload_answers(answers: List[AnswerEntry], request: Request):
    try:
        # Create folder if it doesn't exist
        os.makedirs("user_answers", exist_ok=True)

        # Create unique filename
        client_ip = request.client.host.replace(":", "_")
        output_path = f"user_answers/answers_{client_ip}.json"

        # Save as JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([a.dict() for a in answers], f, ensure_ascii=False, indent=2)

        return {"message": "Answers saved successfully", "file": output_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
