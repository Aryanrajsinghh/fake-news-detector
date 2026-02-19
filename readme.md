# 📰 Fake News Detector

**Final Year CSE Project 2026** | BERT + Streamlit + OpenClaw Chat | 92%+ Accuracy

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FF4B4B?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/)
[![BERT](https://img.shields.io/badge/BERT-FF4B4B?style=for-the-badge&logo=pytorch&logoColor=white)](https://huggingface.co/bert-base-uncased)

## 🎯 Project Overview
AI-powered web app that detects fake news using fine-tuned BERT model. Features real-time social media scanning, interactive Streamlit dashboard, and OpenClaw-style chat interface.

## 🚀 Features
- BERT classification (92% F1-score)
- Live Twitter/Reddit scanning
- Streamlit dashboard with word clouds
- Natural language chat interface
- SQLite logging + Docker ready

## 🛠️ Quick Start
```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
streamlit run app.py
