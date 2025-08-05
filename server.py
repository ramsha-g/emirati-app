from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import tempfile
import json
import os
import joblib
import whisper

from train_model import predict_learning_plan

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load learning track plans
try:
    TRACKS_PATH = os.path.join(os.path.dirname(__file__), "learning_tracks.json")
    with open(TRACKS_PATH, "r", encoding="utf-8") as f:
        LEARNING_TRACKS = json.load(f)
    print("✅ Learning tracks loaded.")
except Exception as e:
    print("❌ Failed to load learning tracks:", str(e))
    LEARNING_TRACKS = {}

# Load model bundle (model + metadata)
try:
    model_bundle = joblib.load("personalisation_model.joblib")
    print("✅ Model bundle loaded successfully.")
except Exception as e:
    print("❌ Failed to load model bundle:", str(e))
    raise

# Load Whisper for transcription
model_whisper = whisper.load_model("base")  # or "small" if needed

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Request Body Schemas
# ------------------------------

class QuizData(BaseModel):
    overall_accuracy: float
    phoneme_mismatch_rate: float
    accuracy_by_tag: dict
    accuracy_by_modality: dict
    accuracy_by_skill: dict
    pronunciation_errors: dict

class AnswerEntry(BaseModel):
    id: str
    correct: bool
    selected: Optional[str]
    tag: str
    modality: str
    skill_targeted: str
    phoneme_error: Optional[str] = None

# ------------------------------
# API Endpoints
# ------------------------------

@app.post("/predict")
def get_plan(data: QuizData):
    try:
        input_dict = data.dict()

        # Sanitize input to ensure expected structure
        for mod in ['text', 'audio', 'speech']:
            input_dict['accuracy_by_modality'][mod] = float(input_dict['accuracy_by_modality'].get(mod, 0.0))
        for p in ['ق', 'ج']:
            input_dict['pronunciation_errors'][p] = float(input_dict['pronunciation_errors'].get(p, 0))

        # Predict using the trained model bundle
        label = predict_learning_plan(model_bundle, input_dict)

        return {
            "learning_plan": LEARNING_TRACKS.get(
                label,
                {"track": label, "level": "A1", "steps": ["Coming soon..."]}
            )
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        result = model_whisper.transcribe(tmp_path, language="ar")
        return {"transcript": result['text'].strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-answers")
async def upload_answers(answers: List[AnswerEntry], request: Request):
    try:
        os.makedirs("user_answers", exist_ok=True)
        client_ip = request.client.host.replace(":", "_")
        output_path = f"user_answers/answers_{client_ip}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([a.dict() for a in answers], f, ensure_ascii=False, indent=2)

        return {"message": "Answers saved successfully", "file": output_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_bundle is not None,
        "endpoints": {
            "predict": "POST /predict",
            "transcribe": "POST /transcribe-audio",
            "upload_answers": "POST /upload-answers",
            "health": "GET /health"
        }
    }


@app.get("/")
def root():
    return {"message": "API is running", "docs": "/docs"}
