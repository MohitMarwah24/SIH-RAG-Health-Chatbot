# streamlit_app.py
import os
import sys
import streamlit as st

# --- Debug: ensure correct Python environment ---
print("Python executable:", sys.executable)
print("Python path:", sys.path)

# --- Imports ---
from pypdf import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

# Try importing Gemini embeddings
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ModuleNotFoundError:
    st.warning("⚠️ langchain-google-genai not found! Using mock embeddings.")
    GEMINI_AVAILABLE = False
    class GoogleGenerativeAIEmbeddings(Embeddings):
        def embed_documents(self, texts): return [[0.0]*768 for _ in texts]
        def embed_query(self, text): return [0.0]*768
    class ChatGoogleGenerativeAI:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return {"result": "Mock response"}

# --- Page setup ---
st.set_page_config(
    page_title="PDF Multilingual Chatbot",
    page_icon="🧪",
    layout="wide"
)

# --- Load PDFs from current folder ---
def load_pdfs(folder_path="."):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            documents.append(Document(page_content=text, metadata={"name": filename}))
    if not documents:
        st.warning("No PDF documents found in current folder.")
    return documents

# --- Split documents ---
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    all_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for c in chunks:
            all_chunks.append(Document(page_content=c, metadata=doc.metadata))
    return all_chunks

# --- Get embeddings ---
def get_embeddings():
    if GEMINI_AVAILABLE:
        try:
            return GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=os.environ.get("GEMINI_API_KEY", "")
            )
        except Exception as e:
            st.warning(f"Failed to load Gemini embeddings: {e}\nUsing mock embeddings.")
    # fallback
    class MockEmbeddings(Embeddings):
        def embed_documents(self, texts): return [[0.0]*768 for _ in texts]
        def embed_query(self, text): return [0.0]*768
    return MockEmbeddings()

# --- Sidebar ---
st.sidebar.title("PDF Chatbot Settings")
folder_path = st.sidebar.text_input("PDF Folder Path", value=".")  # current folder by default

language_options = [
    "English", "Hindi", "Bengali", "Telugu", "Marathi", "Tamil", "Gujarati",
    "Kannada", "Malayalam", "Odia", "Punjabi", "Assamese", "Urdu", "Kashmiri",
    "Maithili", "Sanskrit", "Dogri", "Bodo", "Santhali", "Manipuri", "Same as question"
]

if "lang_select" not in st.session_state:
    st.session_state.lang_select = "English"

st.session_state.lang_select = st.sidebar.selectbox(
    "Select Language",
    language_options,
    index=language_options.index(st.session_state.lang_select)
)

# --- Load docs and vector DB ---
with st.spinner("Loading PDFs and embeddings..."):
    docs = load_pdfs(folder_path)
    chunks = split_documents(docs)

    if not chunks:
        vectordb = None
        st.warning("No PDFs found. Chatbot disabled until PDFs are added.")
    else:
        embeddings = get_embeddings()
        vectordb = Chroma.from_documents(
            chunks,
            embedding=embeddings,
            persist_directory=".chromadb"
        )

# --- UI text ---
ui_text = {
    "English": {"title": "Pulse AI 🧪", "placeholder": "Ask something about the PDF...",
                "examples": ["What are the symptoms of dengue?", "How is malaria transmitted?", "How can we prevent malaria?"]},
    "Hindi": {"title": "Pulse AI 🧪", "placeholder": "पीडीएफ के बारे में कुछ पूछें...",
              "examples": ["डेंगू के लक्षण क्या हैं?", "मलेरिया कैसे फैलता है?", "हम मलेरिया से कैसे बच सकते हैं?"]},
    "Bengali": {"title": "Pulse AI 🧪", "placeholder": "পিডিএফ সম্পর্কে কিছু জিজ্ঞাসা করুন...",
                "examples": ["ডেঙ্গুর লক্ষণ কী?", "ম্যালেরিয়া কীভাবে ছড়ায়?", "আমরা ম্যালেরিয়ার থেকে কিভাবে রক্ষা পাব?"]},
    "Telugu": {"title": "Pulse AI 🧪", "placeholder": "PDF గురించి ఏదైనా అడగండి...",
               "examples": ["డెంగ్యూ లక్షణాలు ఏమిటి?", "మలేరియా ఎలా ప్రసారం అవుతుంది?", "మనం మలేరియా నుండి ఎలా రక్షించగలం?"]},
    "Marathi": {"title": "Pulse AI 🧪", "placeholder": "PDF बद्दल काही विचारा...",
                "examples": ["डेंग्याचे लक्षणे कोणती आहेत?", "मलेरिया कसे पसरणे?", "आपण मलेरियापासून कसे वाचू शकतो?"]},
    "Tamil": {"title": "Pulse AI 🧪", "placeholder": "PDF பற்றி ஏதாவது கேளுங்கள்...",
              "examples": ["டெங்குவின் அறிகுறிகள் என்ன?", "மலேரியா எவ்வாறு பரவுகிறது?", "நாம் மலேரியாவிலிருந்து எப்படி பாதுகாப்பாக இருக்கலாம்?"]},
    "Gujarati": {"title": "Pulse AI 🧪", "placeholder": "PDF વિશે કંઈક પૂછો...",
                 "examples": ["ડેંગ્યુના લક્ષણો શું છે?", "મલેરિયા કેવી રીતે ફેલાય છે?", "અમે મલેરિયાથી કેવી રીતે બચી શકીએ?"]},
    "Kannada": {"title": "Pulse AI 🧪", "placeholder": "PDF ಬಗ್ಗೆ ಏನಾದರೂ ಕೇಳಿ...",
                "examples": ["ಡೆಂಗ್ಯೂ ಲಕ್ಷಣಗಳು ಯಾವುವು?", "ಮಲೆರಿಯಾ ಹೇಗೆ ಹರಡುತ್ತದೆ?", "ನಾವು ಮಲೆರಿಯಾ ಬಗ್ಗೆ ಹೇಗೆ ತಪ್ಪಿಸಿಕೊಳ್ಳಬಹುದು?"]},
    "Malayalam": {"title": "Pulse AI 🧪", "placeholder": "PDF സംബന്ധിച്ച ചില ചോദ്യങ്ങൾ ചോദിക്കൂ...",
                  "examples": ["ഡെംഗ്യു ലക്ഷണങ്ങൾ എന്തെല്ലാം?", "മലേറിയ എങ്ങനെ പടരും?", "മലേറിയയിൽ നിന്ന് നമ്മെ എങ്ങനെ രക്ഷിക്കാം?"]},
    "Odia": {"title": "Pulse AI 🧪", "placeholder": "PDF ବିଷୟରେ କିଛି ପଚାରନ୍ତୁ...",
             "examples": ["ଡେଙ୍ଗୁର ଲକ୍ଷଣ କଣ?", "ମାଲେରିଆ କେମିତି ପ୍ରସାରିତ ହୁଏ?", "ଆମେ କିପରି ମାଲେରିଆରୁ ବଚିପାରିବା?"]},
    "Punjabi": {"title": "Pulse AI 🧪", "placeholder": "PDF ਬਾਰੇ ਕੁਝ ਪੁੱਛੋ...",
                "examples": ["ਡੇਂਗੂ ਦੇ ਲੱਛਣ ਕੀ ਹਨ?", "ਮਲੇਰੀਆ ਕਿਵੇਂ ਫੈਲਦਾ ਹੈ?", "ਅਸੀਂ ਮਲੇਰੀਆ ਤੋਂ ਕਿਵੇਂ ਬਚ ਸਕਦੇ ਹਾਂ?"]},
    "Assamese": {"title": "Pulse AI 🧪", "placeholder": "PDF সম্পৰ্কে কিবা সুধক...",
                 "examples": ["ডেঙ্গুৰ লক্ষণ কি কি?", "মেলেৰিয়া কেনেকৈ ছড়ায়?", "আমি কেনেকৈ মেলেৰিয়াৰ পৰা বাচি থাকিব পাৰিম?"]},
    "Urdu": {"title": "Pulse AI 🧪", "placeholder": "PDF کے بارے میں کچھ پوچھیں...",
             "examples": ["ڈینگی کی علامات کیا ہیں؟", "ملیریا کیسے پھیلتا ہے؟", "ہم ملیریا سے کیسے بچ سکتے ہیں؟"]},
    "Kashmiri": {"title": "Pulse AI 🧪", "placeholder": "PDF بابت کچھ پوچھو...",
                  "examples": ["ڈینگی کی علامات کیا ہیں؟", "ملیریا کیسے پھیلتا ہے؟", "ہم ملیریا سے کیسے بچ سکتے ہیں؟"]},
    "Maithili": {"title": "Pulse AI 🧪", "placeholder": "PDF के बारे में कुछ पूछू...",
                 "examples": ["डेंगू के लक्षण की-की अछि?", "मलेरिया कतेक फैलैत अछि?", "हम मलेरियासँ कियेक बची?"]},
    "Sanskrit": {"title": "Pulse AI 🧪", "placeholder": "PDF विषये किमपि पृच्छतु...",
                 "examples": ["डेंगु लक्ष्णानि का?", "मलेरिया कथं प्रसारितः?", "कथं मलेरियात् रक्षामः?"]},
    "Dogri": {"title": "Pulse AI 🧪", "placeholder": "PDF बारे कुछ पूछो...",
              "examples": ["डेंगू के लक्षण क्या हैं?", "मलेरिया कैसे फैलता है?", "हम मलेरिया से कैसे बच सकते हैं?"]},
    "Bodo": {"title": "Pulse AI 🧪", "placeholder": "PDF आरो दाबा हागो...",
             "examples": ["डेंगू सिम्पटम्स फाइ?", "मलेरिया हागो फेलाय?", "आमी मलेरिया खों बचायो?"]},
    "Santhali": {"title": "Pulse AI 🧪", "placeholder": "PDF रे बारे में पूछु...",
                 "examples": ["डेंगू के लक्षण?", "मलेरिया कोना फैलाए?", "हम मलेरिया से कोना बचाए?"]},
    "Manipuri": {"title": "Pulse AI 🧪", "placeholder": "PDF বিষয়ে কিবা সুধক...",
                 "examples": ["ডেঙ্গুৰ লক্ষণ কি কি?", "মেলেৰিয়া কেনেকৈ ছড়ায়?", "আমি কেনেকৈ মেলেৰিয়াৰ পৰা বাচি থাকিব পাৰিম?"]},
    "Same as question": {"title": "Pulse AI 🧪", "placeholder": "Type question here...", "examples": ["Type your question directly."]}
}

st.markdown(f"<h1>{ui_text[st.session_state.lang_select]['title']}</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Session state for chat ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Query function ---
def query_db(user_query):
    if not user_query.strip():
        return "Please type a query."
    if vectordb is None:
        return "PDF database not loaded. Add PDFs to the folder first."
    results = vectordb.similarity_search(user_query, k=3)
    if not results:
        return "No relevant info found."
    return "\n\n".join([doc.page_content for doc in results])

# --- Chat input form ---
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Type your question here:",
        value="",
        placeholder=ui_text[st.session_state.lang_select]["placeholder"]
    )
    submit = st.form_submit_button("Send")
    if submit and user_input:
        bot_response = query_db(user_input)
        st.session_state.history.append({"user": user_input, "bot": bot_response})

# --- Display chat ---
st.markdown("<div style='max-height:500px; overflow-y:auto; padding:10px; border:1px solid #ddd; border-radius:10px; background-color:#f8f8f8;'>", unsafe_allow_html=True)
for chat in st.session_state.history:
    st.markdown(f"<div style='text-align:right; background-color:#DCF8C6; color:black; padding:10px; border-radius:10px; margin:5px; max-width:70%; margin-left:auto;'>{chat['user']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:left; background-color:#ECECEC; color:black; padding:10px; border-radius:10px; margin:5px; max-width:70%;'>{chat['bot']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
