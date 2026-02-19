import os
import re
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from wordcloud import STOPWORDS, WordCloud

APP_TITLE = "📰 Fake News Detector"
MODEL_DIR = os.getenv("MODEL_DIR", "models/bert-fake-news")
BASE_MODEL = os.getenv("BASE_MODEL", "bert-base-uncased")
DB_PATH = os.getenv("SQLITE_PATH", "predictions.db")
MAX_LOG_ROWS = int(os.getenv("MAX_LOG_ROWS", "1000"))

FAKE_INDICATOR_WORDS = {
    "shocking",
    "secret",
    "hoax",
    "conspiracy",
    "exposed",
    "click here",
    "miracle",
    "unbelievable",
    "urgent",
    "they don't want you to know",
    "viral",
    "breaking",
}


def model_source() -> str:
    model_path = Path(MODEL_DIR)
    return str(model_path) if model_path.exists() else BASE_MODEL


@st.cache_resource(show_spinner=False)
def load_model() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device, str]:
    source = model_source()
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)

    if model.config.num_labels != 2:
        raise ValueError(
            "Model must be a 2-class classifier. Train with train_bert.py or provide a compatible checkpoint."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device, source


def init_db() -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                text TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                fake_probability REAL NOT NULL,
                real_probability REAL NOT NULL,
                model_source TEXT NOT NULL
            )
            """
        )
        conn.commit()


def log_prediction(
    text: str,
    label: str,
    confidence: float,
    fake_p: float,
    real_p: float,
    source: str,
) -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            """
            INSERT INTO prediction_logs (
                timestamp, text, predicted_label, confidence,
                fake_probability, real_probability, model_source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                text[:4000],
                label,
                float(confidence),
                float(fake_p),
                float(real_p),
                source,
            ),
        )
        conn.execute(
            """
            DELETE FROM prediction_logs
            WHERE id NOT IN (
                SELECT id FROM prediction_logs ORDER BY id DESC LIMIT ?
            )
            """,
            (MAX_LOG_ROWS,),
        )
        conn.commit()


def fetch_logs(limit: int = 100) -> pd.DataFrame:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        return pd.read_sql_query(
            """
            SELECT timestamp, text, predicted_label, confidence,
                   fake_probability, real_probability, model_source
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )


def preprocess_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _probabilities(text: str, tokenizer, model, device) -> List[float]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0].tolist()
    return probs


def predict_news(text: str, tokenizer, model, device) -> Dict[str, float]:
    probs = _probabilities(text, tokenizer, model, device)

    # By convention from train_bert.py: id 0 => REAL, id 1 => FAKE.
    # If model labels are available, use them as authority.
    id2label = {int(k): str(v).upper() for k, v in model.config.id2label.items()} if model.config.id2label else {}
    if id2label.get(0) == "REAL" and id2label.get(1) == "FAKE":
        real_p, fake_p = float(probs[0]), float(probs[1])
    elif id2label.get(0) == "FAKE" and id2label.get(1) == "REAL":
        fake_p, real_p = float(probs[0]), float(probs[1])
    else:
        real_p, fake_p = float(probs[0]), float(probs[1])

    label = "FAKE" if fake_p >= real_p else "REAL"
    confidence = max(fake_p, real_p)
    return {
        "label": label,
        "confidence": confidence,
        "fake_probability": fake_p,
        "real_probability": real_p,
    }


def token_candidates(texts: List[str]) -> str:
    corpus = " ".join(texts).lower()
    for phrase in FAKE_INDICATOR_WORDS:
        corpus = corpus.replace(phrase, phrase.replace(" ", "_"))
    words = re.findall(r"\b[a-z_]{4,}\b", corpus)
    filtered = [word.replace("_", " ") for word in words if word not in STOPWORDS]
    return " ".join(filtered)


def build_wordcloud(texts: List[str]):
    merged = token_candidates(texts)
    if not merged.strip():
        return None
    wc = WordCloud(width=1000, height=400, background_color="white", colormap="Reds", stopwords=STOPWORDS).generate(merged)
    fig = px.imshow(wc.to_array())
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis_visible=False, yaxis_visible=False)
    return fig


def chat_response(user_message: str, tokenizer, model, device) -> str:
    prediction = predict_news(user_message, tokenizer, model, device)
    return (
        f"BERT analysis: This text is **{prediction['label']}** with "
        f"**{prediction['confidence'] * 100:.2f}%** confidence.\n\n"
        f"Fake probability: {prediction['fake_probability'] * 100:.2f}% | "
        f"Real probability: {prediction['real_probability'] * 100:.2f}%"
    )


def main() -> None:
    st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
    st.title(APP_TITLE)
    st.caption("BERT-powered fake news classifier with chat analysis, word cloud visualization, and SQLite logs.")

    init_db()

    try:
        tokenizer, model, device, source = load_model()
    except Exception as exc:
        st.error(f"Unable to load model: {exc}")
        st.stop()

    source_path = Path(source)
    if source_path.exists() and source_path.name == "bert-fake-news":
        st.success(f"Loaded fine-tuned model from `{source}`")
    else:
        st.warning(
            f"Fine-tuned model was not found at `{MODEL_DIR}`. Loaded base model `{source}` for smoke usage only. "
            "Train with `train_bert.py` for meaningful fake-news predictions."
        )

    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("1) News Classification")
        input_text = st.text_area(
            "Paste a news headline or article excerpt",
            height=180,
            placeholder="Example: Government announces a miracle cure that media is hiding...",
        )

        if st.button("Analyze with BERT", use_container_width=True):
            cleaned = preprocess_text(input_text)
            if not cleaned:
                st.warning("Please enter text to analyze.")
            else:
                pred = predict_news(cleaned, tokenizer, model, device)
                log_prediction(
                    cleaned,
                    pred["label"],
                    pred["confidence"],
                    pred["fake_probability"],
                    pred["real_probability"],
                    source,
                )

                badge = "🚨 Likely FAKE" if pred["label"] == "FAKE" else "✅ Likely REAL"
                st.markdown(f"### {badge}")
                st.metric("Confidence", f"{pred['confidence'] * 100:.2f}%")

                probs_df = pd.DataFrame(
                    {
                        "Class": ["REAL", "FAKE"],
                        "Probability": [pred["real_probability"], pred["fake_probability"]],
                    }
                )
                bar = px.bar(probs_df, x="Class", y="Probability", color="Class", range_y=[0, 1])
                st.plotly_chart(bar, use_container_width=True)

        st.subheader("2) Chat: Is this fake?")
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        for role, msg in st.session_state.chat_messages:
            with st.chat_message(role):
                st.markdown(msg)

        user_prompt = st.chat_input("Ask: Is this fake? Paste text and press Enter...")
        if user_prompt:
            cleaned_prompt = preprocess_text(user_prompt)
            st.session_state.chat_messages.append(("user", cleaned_prompt))
            answer = chat_response(cleaned_prompt, tokenizer, model, device)
            st.session_state.chat_messages.append(("assistant", answer))
            pred = predict_news(cleaned_prompt, tokenizer, model, device)
            log_prediction(
                cleaned_prompt,
                pred["label"],
                pred["confidence"],
                pred["fake_probability"],
                pred["real_probability"],
                source,
            )
            st.rerun()

    with right_col:
        st.subheader("3) Fake Indicator Word Cloud")
        logs = fetch_logs(limit=300)
        fake_logs = logs[logs["predicted_label"] == "FAKE"]["text"].tolist() if not logs.empty else []
        wc_fig = build_wordcloud(fake_logs)
        if wc_fig is not None:
            st.plotly_chart(wc_fig, use_container_width=True)
        else:
            st.info("Need more FAKE predictions to generate the word cloud.")

        st.subheader("4) Recent Prediction Logs")
        if logs.empty:
            st.info("No logs yet. Run a prediction to populate SQLite logs.")
        else:
            st.dataframe(logs.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
