<div align="center">

# 📰 Fake News Detector
### Final Year CSE Project

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)

<br/>

> **BERT-powered fake news classification system with a custom web frontend, REST API, Streamlit dashboard, and SQLite logging.**

<br/>

![-----](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

</div>

<br/>

## 🧠 How It Works

```
 News Text  ──▶  BERT Tokenizer  ──▶  Fine-tuned BERT  ──▶  FAKE / REAL + Confidence
                  (bert-base-uncased)    (2-class classifier)
```

The system fine-tunes `bert-base-uncased` on a labelled fake/real news corpus, achieving **92.3% accuracy** and **0.925 F1-score** on the validation set.

<br/>

## ✨ Features

| Interface | Features |
|-----------|----------|
| 🌐 **HTML Frontend** | Glassmorphism dark UI · BERT prediction · Confidence display · Connected to FastAPI |
| 📊 **Streamlit Dashboard** | Classification panel · Chat assistant · Word cloud · SQLite logs · Metrics sidebar |
| ⚡ **FastAPI Backend** | `/predict` REST endpoint · Auto docs · CORS enabled · Short-text validation |
| 🐳 **Docker** | Production-ready containerized deployment |

<br/>

## 📁 Project Structure

```
fake-news-detector/
│
├── 🌐 Frontend/
│   ├── index.html          # Custom web interface
│   ├── script.js           # FastAPI fetch calls
│   └── styles.css          # Glassmorphism styling
│
├── ⚡ api.py               # FastAPI inference server
├── 📊 app.py               # Streamlit dashboard
├── 🧠 train_bert.py        # BERT fine-tuning script
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── readme.md
```

<br/>

## 📊 Model Performance

<div align="center">

| Metric | Score |
|:------:|:-----:|
| ✅ Accuracy | **0.923** |
| 🎯 Precision | **0.918** |
| 🔍 Recall | **0.933** |
| ⭐ F1-Score | **0.925** |

> 🏆 Target achieved: **90%+ F1-Score**

</div>

<br/>

## 🚀 Quick Start

### 1 — Clone & Install

```bash
git clone https://github.com/Aryanrajsinghh/fake-news-detector.git
cd fake-news-detector

python -m venv .venv311
.venv311\Scripts\activate        # Windows
source .venv311/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### 2 — Train the Model

Prepare a CSV with columns: `title`, `source_domain`, `real` (1 = REAL, 0 = FAKE)

```bash
python train_bert.py \
  --train-file data/train.csv \
  --validation-file data/valid.csv \
  --output-dir models/bert-fake-news \
  --epochs 3 \
  --batch-size 8
```

Outputs saved to `models/bert-fake-news/` including `metrics.json`

### 3 — Run the App

**Option A — HTML Frontend + FastAPI**
```bash
# Terminal 1
uvicorn api:app --reload --port 8000

# Then open Frontend/index.html in your browser
# API docs at http://localhost:8000/docs
```

**Option B — Streamlit Dashboard**
```bash
streamlit run app.py
# Open http://localhost:8501
```

<br/>

## 🐳 Docker Deployment

```bash
# Build
docker build -t fake-news-detector:latest .

# Run
docker run --rm -p 8501:8501 \
  -v $(pwd)/models/bert-fake-news:/app/models/bert-fake-news \
  fake-news-detector:latest
```

Open: `http://localhost:8501`

<br/>

## 🔒 Production Notes

- 🔐 Add authentication to `/predict` before public deployment
- 📦 Version model artifacts via a model registry
- 🔄 Rotate `predictions.db` in long-running environments
- ⚡ For high traffic: serve FastAPI separately with TorchServe or Triton

<br/>

![-----](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<div align="center">

Made with ❤️ as a Final Year CSE Project

</div>