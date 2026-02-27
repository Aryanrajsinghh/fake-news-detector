from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="Fake News Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path("models/bert-fake-news")

if not MODEL_DIR.exists():
    raise RuntimeError(
        f"Model directory '{MODEL_DIR}' not found. Run train_bert.py first."
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


class NewsInput(BaseModel):
    text: str


@app.get("/")
def root():
    return {"status": "API running", "device": str(device)}


@app.post("/predict")
def predict_news(input_data: NewsInput):
    text = input_data.text.strip()

    if not text:
        return {
            "label": "ERROR",
            "confidence": 0,
            "fake_probability": 0,
            "real_probability": 0,
        }

    if len(text.split()) < 3:
        return {
            "label": "UNCERTAIN",
            "confidence": 0,
            "fake_probability": 0,
            "real_probability": 0,
            "warning": "Text too short for reliable prediction.",
        }

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    # Matches train_bert.py convention: 0 = REAL, 1 = FAKE
    real_prob = float(probs[0])
    fake_prob = float(probs[1])

    label = "FAKE" if fake_prob > real_prob else "REAL"
    confidence = max(fake_prob, real_prob)

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "fake_probability": round(fake_prob * 100, 2),
        "real_probability": round(real_prob * 100, 2),
    }