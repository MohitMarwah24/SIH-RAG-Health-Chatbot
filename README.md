# 🧠 AI-Powered Medical Chatbot using RAG

### 🚀 Smart India Hackathon (SIH) Project

---

## 📌 Project Description

This project is an **AI-powered Medical Question Answering System** that uses **Retrieval-Augmented Generation (RAG)** to provide accurate and context-based responses from a medical dataset.

Traditional chatbots rely only on pre-trained knowledge, which can be outdated or incorrect. This system improves reliability by **retrieving real information from documents** before generating answers.

📄 The chatbot reads a medical PDF (`diseases.pdf`) and answers user queries based strictly on that content.

---

## ❗ Problem Statement

Many users struggle to find **reliable and understandable medical information** online. Existing systems:

* Provide generic answers
* Lack document-based accuracy
* Cannot handle domain-specific queries effectively

---

## 💡 Proposed Solution

We developed a **RAG-based chatbot** that:

* Extracts knowledge from medical documents
* Stores it in a vector database
* Retrieves relevant information for each query
* Generates accurate and contextual answers

This ensures:
✅ Better accuracy
✅ Context-aware responses
✅ Domain-specific knowledge

---

## 🧠 What is RAG (Retrieval-Augmented Generation)?

RAG is a technique that combines:

* 🔎 **Retrieval** → Fetch relevant data from a database
* 🤖 **Generation** → Use AI model to generate answers

Instead of guessing, the model answers using **real retrieved content**.

---

## 🏗️ System Architecture

The project follows a structured pipeline:

### 1️⃣ Data Ingestion

* Medical PDF is loaded
* Text is split into smaller chunks

### 2️⃣ Embedding Generation

* Each chunk is converted into vector embeddings

### 3️⃣ Vector Database Storage

* Embeddings are stored in **ChromaDB**

### 4️⃣ Query Processing

* User query is converted into embedding
* Similar content is retrieved from database

### 5️⃣ Answer Generation

* Retrieved content + query → passed to LLM
* Final answer is generated

---

## ⚙️ Tech Stack

### 🧑‍💻 Programming Language

* Python

### 🤖 AI / ML

* Retrieval-Augmented Generation (RAG)
* Embeddings
* Semantic Search

### 📚 Libraries & Frameworks

* **Streamlit** → Frontend UI
* **LangChain** → RAG pipeline
* **ChromaDB** → Vector database
* **PyPDF** → PDF processing
* **OpenAI API** → Response generation

### 🗄️ Database

* Chroma Vector Database

---

## 📂 Project Structure

```
SIH-RAG-Medical-Chatbot
│
├── streamlit_app.py       # Streamlit UI
├── query_rag.py          # RAG query logic
├── ingest_pdf.py         # PDF processing + embeddings
│
├── diseases.pdf          # Medical dataset
│
├── chroma_db/            # Vector database (generated)
│
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── LICENSE
```

---

## 🔄 Workflow Explanation

1. Run `ingest_pdf.py`
   → Converts PDF into embeddings and stores in database

2. Run `streamlit_app.py`
   → Starts chatbot UI

3. Ask a question
   → System retrieves relevant data
   → Generates answer using AI

---

## ⚙️ Installation & Setup Guide

### 📥 Step 1: Clone Repository

```bash
git clone https://github.com/your-username/SIH-RAG-Medical-Chatbot.git
cd SIH-RAG-Medical-Chatbot
```

---

### 🧪 Step 2: Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

---

### 📦 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🔐 Step 4: Setup Environment Variables

Create a `.env` file in the root folder:

```
OPENAI_API_KEY=your_openai_api_key_here
```

---

## ▶️ How to Run the Project

### Step 1: Process the PDF

```bash
python ingest_pdf.py
```

👉 This creates the vector database.

---

### Step 2: Start the Chatbot

```bash
streamlit run streamlit_app.py
```

---

### Step 3: Open in Browser

```
http://localhost:8501
```

---

## 💬 Example Queries

* "What are symptoms of diabetes?"
* "Explain causes of hypertension"
* "What is treatment for fever?"

---

## 📊 Applications

* 🏥 Medical information assistant
* 📚 Educational tool
* 🤖 AI chatbot systems
* 📄 Document-based Q&A

---

## 🚀 Future Enhancements

* Multi-document support
* Voice interaction
* Mobile app integration
* Better UI/UX
* Cloud deployment

---

## 👨‍💻 Contribution (Your Role)

* Implemented **RAG pipeline**
* Developed **PDF ingestion system**
* Integrated **ChromaDB vector storage**
* Built **Streamlit chatbot interface**

---

## 👥 Contributors

* Mohit Marwah

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Acknowledgement

Developed as part of **Smart India Hackathon (SIH)** to solve real-world problems using AI.

---
