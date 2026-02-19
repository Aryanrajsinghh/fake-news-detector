# 📰 Fake News Detector (Final Year CSE Project)

A complete fake news detection project using **BERT (`bert-base-uncased`)**, **Streamlit**, **PyTorch**, **SQLite**, and **Plotly**.

This repository includes:
- `train_bert.py` → fine-tune BERT on your dataset.
- `app.py` → Streamlit UI with prediction, confidence, chat, word cloud, and logging.
- `Dockerfile` → production-ready container build/run.

---

## ✅ Features Implemented

1. **BERT fake-news classifier** (`bert-base-uncased` fine-tuning).
2. **Streamlit dashboard** with text input → class + confidence.
3. **Chat mode** (“Is this fake?”) using the same BERT inference pipeline.
4. **Word cloud** from logged FAKE predictions.
5. **SQLite logging** of every prediction.
6. **Docker support** for deployment.

---

## 📁 Project Structure

```text
.
├── app.py
├── train_bert.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── readme.md
```

---

## 🧱 Prerequisites

- Python 3.10+
- pip
- (Optional) CUDA GPU for faster training/inference

---

## ⚙️ Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📚 Dataset Format

Provide two CSV files (train + validation) with:
- `text` column: news headline/article/body
- `label` column: one of `0/1` or `real/fake`

### Example

```csv
text,label
"Government approves new policy for public schools",real
"Shocking secret cure hidden by media",fake
```

---

## 🧠 Model Training

Run:

```bash
python train_bert.py \
  --train-file data/train.csv \
  --validation-file data/valid.csv \
  --text-column text \
  --label-column label \
  --output-dir models/bert-fake-news \
  --epochs 3 \
  --batch-size 8
```

### Output Artifacts

After training:
- Model + tokenizer saved to `models/bert-fake-news/`
- Validation metrics saved to `models/bert-fake-news/metrics.json`

### Target Accuracy

Your target is **90%+ F1**. This depends on dataset quality, class balance, and train/validation split.

> To report “actual accuracy results”, copy values from your generated `metrics.json`.

---

## 🖥️ Run the Streamlit App

```bash
streamlit run app.py
```

Open: `http://localhost:8501`

### What you’ll see
- **Prediction panel**: paste text, get `REAL/FAKE` + confidence.
- **Chat panel**: ask “Is this fake?” with free text.
- **Word cloud**: generated from FAKE logs.
- **Recent logs**: stored in `predictions.db`.

### Model loading behavior
- If `models/bert-fake-news` exists, app loads the fine-tuned classifier.
- If missing, app falls back to base `bert-base-uncased` and shows a warning (for smoke testing only).

---

## 🗃️ SQLite Logging

By default app writes to:
- `predictions.db`
- Table: `prediction_logs`

Stored fields:
- timestamp
- input text
- predicted label
- confidence
- fake/real probabilities
- model source

Configurable env vars:
- `SQLITE_PATH` (default: `predictions.db`)
- `MAX_LOG_ROWS` (default: `1000`)

---

## 🐳 Docker Usage

### Build

```bash
docker build -t fake-news-detector:latest .
```

### Run

```bash
docker run --rm -p 8501:8501 \
  -v $(pwd)/models/bert-fake-news:/app/models/bert-fake-news \
  -v $(pwd)/predictions.db:/app/predictions.db \
  fake-news-detector:latest
```

Open: `http://localhost:8501`

---

## ✅ Validation Checklist

Before demo/submission:
1. Train and confirm `models/bert-fake-news/metrics.json` exists.
2. Ensure `f1 >= 0.90` on validation set.
3. Run app and test both button prediction and chat prediction.
4. Confirm logs are written to SQLite.
5. Verify word cloud updates after fake predictions.

---

## 📌 Suggested Next Improvements

- Add test scripts (unit + integration).
- Add model versioning + experiment tracking.
- Add confidence threshold + “uncertain” class.
- Add batch CSV prediction mode.
- Deploy with CI/CD to cloud (Render/AWS/GCP/Azure).
