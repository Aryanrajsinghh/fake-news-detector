from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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


# Serve frontend
FRONTEND_DIST = Path("Frontend/dist")
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/")
def serve_frontend():
    return FileResponse("Frontend/dist/index.html")


@app.get("/health")
def health():
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

    # probs[0]=FAKE, probs[1]=REAL based on your training
    fake_prob = float(probs[0])
    real_prob = float(probs[1])

    is_fake = fake_prob > real_prob
    label = "FAKE" if is_fake else "REAL"

    # Frontend does its own *100 multiplication AND determines label
    # from whichever of fake_probability/real_probability is higher.
    # So we send raw decimals (0.0-1.0) and let frontend handle display.
    # We also ensure fake_probability > real_probability when FAKE
    # so the frontend's internal label logic matches reality.
    return {
        "label": label,
        "confidence": round(fake_prob if is_fake else real_prob, 4),
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(real_prob, 4),
    }