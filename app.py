import os
import re
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from wordcloud import WordCloud

APP_TITLE = "📰 Fake News Detector"
MODEL_DIR = os.getenv("MODEL_DIR", "models/bert-fake-news")
DB_PATH = os.getenv("SQLITE_PATH", "predictions.db")
MAX_LOG_ROWS = 500

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


@st.cache_resource(show_spinner=False)
def load_model() -> Tuple[
    Optional[AutoTokenizer],
    Optional[AutoModelForSequenceClassification],
    Optional[torch.device],
    str,
    Optional[str],
]:
    model_path = Path(MODEL_DIR)
    if not model_path.exists():
        return None, None, None, "heuristic", f"Model directory '{MODEL_DIR}' not found."

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return tokenizer, model, device, "bert", None
    except Exception as exc:
        warning = f"Model artifacts at '{MODEL_DIR}' are incomplete or unreadable ({exc})."
        return None, None, None, "heuristic", warning


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
                real_probability REAL NOT NULL
            )
            """
        )
        conn.commit()


def log_prediction(text: str, label: str, confidence: float, fake_p: float, real_p: float) -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            """
            INSERT INTO prediction_logs (timestamp, text, predicted_label, confidence, fake_probability, real_probability)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                text[:2000],
                label,
                float(confidence),
                float(fake_p),
                float(real_p),
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
            SELECT timestamp, text, predicted_label, confidence, fake_probability, real_probability
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )


def preprocess_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def heuristic_prediction(text: str) -> Tuple[float, float]:
    lowered = text.lower()
    matched = sum(1 for phrase in FAKE_INDICATOR_WORDS if phrase in lowered)
    exclamation_boost = min(lowered.count("!") * 0.03, 0.15)
    base_fake = min(0.2 + (matched * 0.15) + exclamation_boost, 0.95)
    fake_p = float(base_fake)
    real_p = float(1 - fake_p)
    return real_p, fake_p


def predict_news(text: str, tokenizer, model, device, backend: str) -> Dict[str, float]:
    if backend == "bert":
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
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        # Convention in train_bert.py: 0 = real, 1 = fake
        real_p, fake_p = float(probs[0]), float(probs[1])
    else:
        real_p, fake_p = heuristic_prediction(text)

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
    filtered = [w.replace("_", " ") for w in words]
    return " ".join(filtered)


def build_wordcloud(texts: List[str]):
    merged = token_candidates(texts)
    if not merged.strip():
        return None
    wc = WordCloud(width=1000, height=400, background_color="white", colormap="Reds").generate(merged)
    img = wc.to_array()
    fig = px.imshow(img)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis_visible=False, yaxis_visible=False)
    return fig


def chat_response(user_message: str, tokenizer, model, device, backend: str) -> str:
    prediction = predict_news(user_message, tokenizer, model, device, backend)
    mode_note = "BERT" if backend == "bert" else "Heuristic fallback"
    return (
        f"{mode_note} analysis: This text is **{prediction['label']}** with "
        f"**{prediction['confidence'] * 100:.2f}%** confidence.\n\n"
        f"Fake probability: {prediction['fake_probability'] * 100:.2f}% | "
        f"Real probability: {prediction['real_probability'] * 100:.2f}%"
    )


def main() -> None:
    st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
    st.title(APP_TITLE)
    st.caption("BERT-powered fake news classifier with chat analysis, word cloud, and SQLite logging.")

    init_db()
    tokenizer, model, device, backend, model_warning = load_model()

    if backend == "heuristic":
        st.warning(
            "Running in heuristic fallback mode. "
            f"{model_warning} Train with train_bert.py or set MODEL_DIR to enable BERT inference."
        )

    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("1) News Classification")
        input_text = st.text_area(
            "Paste a news headline/article excerpt",
            height=180,
            placeholder="Example: Government announces a shocking miracle cure...",
        )

        if st.button("Analyze with BERT", use_container_width=True):
            cleaned = preprocess_text(input_text)
            if not cleaned:
                st.warning("Please enter some text first.")
            else:
                pred = predict_news(cleaned, tokenizer, model, device, backend)
                log_prediction(
                    cleaned,
                    pred["label"],
                    pred["confidence"],
                    pred["fake_probability"],
                    pred["real_probability"],
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
            st.session_state.chat_messages.append(("user", user_prompt))
            answer = chat_response(user_prompt, tokenizer, model, device, backend)
            st.session_state.chat_messages.append(("assistant", answer))
            pred = predict_news(user_prompt, tokenizer, model, device, backend)
            log_prediction(
                user_prompt,
                pred["label"],
                pred["confidence"],
                pred["fake_probability"],
                pred["real_probability"],
            )
            st.rerun()

    with right_col:
        st.subheader("3) Fake Indicator Word Cloud")
        logs = fetch_logs(limit=200)
        fake_logs = logs[logs["predicted_label"] == "FAKE"]["text"].tolist() if not logs.empty else []
        wc_fig = build_wordcloud(fake_logs)
        if wc_fig is not None:
            st.plotly_chart(wc_fig, use_container_width=True)
        else:
            st.info("Need a few FAKE predictions to generate the word cloud.")

        st.subheader("4) Recent Prediction Logs")
        if logs.empty:
            st.info("No logs yet. Run a prediction to populate SQLite logs.")
        else:
            st.dataframe(logs.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
