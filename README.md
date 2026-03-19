# 🧠 SIH Project – Medical RAG Chatbot

## 📌 Overview

This project is an AI-powered chatbot developed for **Smart India Hackathon (SIH)**.
It uses **Retrieval-Augmented Generation (RAG)** to answer medical questions from a PDF dataset.

---

## 🚀 Features

* 📄 PDF-based knowledge system
* 🤖 AI chatbot
* 🔎 Semantic search
* 💬 Streamlit UI

---

## 🏗️ Tech Stack

* Python
* Streamlit
* LangChain
* ChromaDB
* OpenAI API

---

## ⚙️ Setup Instructions

### 1. Clone Repo

```bash
git clone https://github.com/your-username/SIH-RAG-Medical-Chatbot.git
cd SIH-RAG-Medical-Chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API Key

Create `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

---

## ▶️ Run Project

### Step 1: Ingest Data

```bash
python ingest_pdf.py
```

### Step 2: Run App

```bash
streamlit run streamlit_app.py
```

Open:

```
http://localhost:8501
```

---

## 📊 Use Cases

* Medical chatbot
* Document Q&A system
* Educational assistant

---

## 👥 Contributors

* Mohit Marwah
---

## 📜 License

MIT License
