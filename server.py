from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from train_model import predict_learning_plan

app = FastAPI()
model = joblib.load("personalisation_model.joblib")

class QuizData(BaseModel):
    overall_accuracy: float
    phoneme_mismatch_rate: float
    accuracy_by_tag: dict
    accuracy_by_modality: dict
    pronunciation_errors: dict

@app.post("/predict")
def get_plan(data: QuizData):
    input_dict = data.dict()
    label = predict_learning_plan(model, input_dict)
    return {"plan": label}
