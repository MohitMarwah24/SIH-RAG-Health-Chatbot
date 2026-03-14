# ingest_pdf.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# prefer new package if installed, fallback to community version
try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

def ingest(pdf_path="/Users/abcd/Desktop/SIH Project/diseases.pdf", persist_dir="./chroma_db"):
    # 1) Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2) Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # 3) Gemini embeddings (reads key from env)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.environ["GEMINI_API_KEY"]
    )

    # 4) Save to Chroma (Chroma auto-persists now)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print(f"Ingested {len(chunks)} chunks into {persist_dir}")

if __name__ == "__main__":
    ingest()
