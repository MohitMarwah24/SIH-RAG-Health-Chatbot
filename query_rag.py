# query_rag_openrouter.py
import os
from typing import Optional

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# --- Humanize symptom responses ---
def humanize_response(query: str, answer: str) -> str:
    symptoms = [
        "fever", "headache", "cold", "cough", "dizzy", "nausea", "not feeling well",
        "sick", "fatigue", "vomit", "diarrhea", "body pain", "chills", "tired",
        "unwell", "weak", "sore throat", "breathless", "pain", "loss of appetite"
    ]
    query_lower = query.lower()
    if any(symptom in query_lower for symptom in symptoms):
        return (
            "🩺 **It sounds like you're not feeling well. Here's what you can do:**\n\n"
            "- Get enough rest and avoid physical exertion.\n"
            "- Drink plenty of fluids (water, ORS, clear soups, herbal teas).\n"
            "- Stay in a calm, cool, and quiet environment.\n"
            "- Apply a cold compress if you have a headache or fever.\n"
            "- Avoid spicy or oily foods — eat light meals.\n"
            "- Monitor your symptoms. If they worsen, consult a healthcare provider.\n\n"
            "❤️ I'm here to help you with more information if you need. Just ask!"
        )
    return answer

# --- Build Chroma vector DB ---
def build_vectordb(persist_dir="./chroma_db"):
    embeddings = OpenAIEmbeddings(
        model="google/gemini-2.5-flash",
        openai_api_key=os.environ["GEMINI_API_KEY"],
        openai_api_base="https://openrouter.ai/api/v1"
    )
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    return vectordb

# --- Query the DB ---
def query_db(question: str, lang: Optional[str] = "English", persist_dir="./chroma_db") -> str:
    vectordb = build_vectordb(persist_dir=persist_dir)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="google/gemini-2.5-flash",
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2
    )

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    personality = (
        "You are a helpful, caring, and professional health assistant. "
        "When users mention symptoms like headache, fever, or weakness, respond empathetically "
        "and provide point-wise, clear health advice. Avoid emotional or overly personal language. "
        "Keep the tone kind, supportive, and respectful."
    )

    if lang is None or str(lang).strip().lower() in ("same", "auto", ""):
        final_question = f"{personality}\n\nQuestion: {question}"
    else:
        final_question = f"{personality}\n\nAnswer in {lang}.\nQuestion: {question}"

    out = qa.invoke({"query": final_question})
    answer = out.get("result", out)

    return humanize_response(question, answer)

# --- Main loop ---
if __name__ == "__main__":
    print("💡 Health Assistant Chat (type 'exit' to quit)")
    lang = input("Enter your language (e.g., English, Hindi): ").strip()
    if not lang:
        lang = "English"
    while True:
        q = input("\nEnter your question: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        ans = query_db(q, lang=lang)
        print(f"\n🤖 Answer ({lang}):\n{ans}\n")













# import os
# from typing import Optional

# try:
#     from langchain_chroma import Chroma
# except Exception:
#     from langchain_community.vectorstores import Chroma

# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA
# from deep_translator import GoogleTranslator

# # ✅ Language name → ISO code mapping
# LANG_CODE_MAP = {
#     "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te", "Bengali": "bn",
#     "Punjabi": "pa", "Marathi": "mr", "Gujarati": "gu", "Kannada": "kn",
#     "Malayalam": "ml", "Urdu": "ur", "Odia": "or", "Assamese": "as",
#     "Maithili": "mai", "Konkani": "kok", "Sindhi": "sd", "Santali": "sat",
#     "Kashmiri": "ks", "Dogri": "doi", "Manipuri": "mni", "Bodo": "brx", "Nepali": "ne"
# }

# def humanize_response(query: str, answer: str) -> str:
#     """Add empathetic health advice if symptoms or general wellness are detected."""
#     symptoms = [
#         "fever","headache","cold","cough","dizzy","nausea","vomit","diarrhea",
#         "body pain","chills","fatigue","tired","weak","sore throat","breathless",
#         "pain","loss of appetite","stomach ache","indigestion","rash","allergy",
#         "swelling","joint pain","back pain","anxiety","stress","insomnia",
#         "dehydration","skin irritation"
#     ]
#     general_health = [
#         "diet","exercise","sleep","healthy","nutrition","prevent","wellness",
#         "fitness","habits","lifestyle"
#     ]

#     query_lower = query.lower()
#     if any(symptom in query_lower for symptom in symptoms):
#         return (
#             "🩺 **It sounds like you may be experiencing some health symptoms. Here's general advice:**\n\n"
#             "- Rest as much as possible and avoid overexertion.\n"
#             "- Stay hydrated (water, ORS, herbal teas, light soups).\n"
#             "- Eat light, nutritious food (fruits, vegetables, whole grains).\n"
#             "- Track your temperature and other symptoms regularly.\n"
#             "- Use home remedies for comfort (steam inhalation, warm water, salt gargle).\n"
#             "- Avoid junk, oily, or very spicy food.\n"
#             "- If symptoms persist, worsen, or include high fever, chest pain, or severe breathlessness, seek medical attention promptly.\n\n"
#             "❤️ Take care — I can also provide information from the documents if you'd like."
#         )
#     if any(keyword in query_lower for keyword in general_health):
#         return (
#             "🌱 **Here are some general wellness tips:**\n\n"
#             "- Eat a balanced diet with fresh fruits, vegetables, and proteins.\n"
#             "- Exercise at least 30 minutes daily (walking, yoga, or light workouts).\n"
#             "- Maintain regular sleep (7–8 hours) and avoid excessive screen time.\n"
#             "- Stay hydrated — 8–10 glasses of water a day.\n"
#             "- Avoid smoking, alcohol, and excessive caffeine.\n"
#             "- Practice stress management (deep breathing, meditation, journaling).\n"
#             "- Wash hands regularly and maintain hygiene to prevent infections.\n\n"
#             "💡 Small lifestyle changes build long-term health benefits!"
#         )
#     return f"{answer}\n\n✨ Remember: Health information here is for awareness. Always consult a doctor for diagnosis and treatment."

# def build_vectordb(persist_dir="./chroma_db"):
#     """Load or build the Chroma vector database with Gemini embeddings."""
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/gemini-embedding-001",
#         google_api_key=os.environ["GEMINI_API_KEY"]
#     )
#     vectordb = Chroma(
#         persist_directory=persist_dir,
#         embedding_function=embeddings
#     )
#     return vectordb

# def query_db(question: str, lang: Optional[str] = "English", persist_dir="./chroma_db") -> str:
#     """
#     Main pipeline for multilingual WhatsApp replies:
#     1. Translate input → English (for LLM understanding)
#     2. LLM + Retrieval
#     3. Humanize response
#     4. Translate back → user's language (fallback)
#     """

#     # Step 1: Translate question → English
#     translated_question = question
#     try:
#         translated_question = GoogleTranslator(source="auto", target="en").translate(question)
#         print(f"DEBUG: Translated question to English: {translated_question}")
#     except Exception as e:
#         print("⚠️ Question translation failed, using original:", e)

#     # Step 2: LLM + Retrieval
#     vectordb = build_vectordb(persist_dir=persist_dir)
#     retriever = vectordb.as_retriever(search_kwargs={"k": 4})

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-1.5-flash",
#         google_api_key=os.environ["GEMINI_API_KEY"]
#     )

#     qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

#     personality = (
#         "You are a helpful, caring, and professional health assistant. "
#         "Always generate answers in the language of the user. "
#         "When users mention symptoms like headache, fever, or weakness, respond empathetically "
#         "and provide point-wise, clear health advice. "
#         "Keep the tone kind, supportive, and respectful."
#     )

#     final_question = f"{personality}\n\nQuestion: {translated_question}"

#     # Step 2a: Invoke LLM safely
#     try:
#         out = qa.invoke({"query": final_question})
#         # Handle both dict or string outputs
#         if isinstance(out, dict):
#             english_answer = out.get("result") or out.get("output_text") or str(out)
#         else:
#             english_answer = str(out)
#     except Exception as e:
#         print("⚠️ LLM invocation failed:", e)
#         english_answer = "⚠️ Sorry, I couldn’t generate an answer. Please try again."

#     print("DEBUG: LLM output (English):", english_answer)

#     # Step 3: Humanize response (still in English)
#     english_answer = humanize_response(translated_question, english_answer)

#     # Step 4: Translate back → user's language if not English
#     if lang not in ["English", None, ""]:
#         target_code = LANG_CODE_MAP.get(lang, "en")
#         try:
#             translated_answer = GoogleTranslator(source="en", target=target_code).translate(english_answer)
#             print(f"DEBUG: Translated answer ({lang}):", translated_answer)
#             return translated_answer
#         except Exception as e:
#             print("⚠️ Answer translation failed:", e)
#             return english_answer

#     return english_answer




# import os
# from typing import Optional

# try:
#     from langchain_chroma import Chroma
# except Exception:
#     from langchain_community.vectorstores import Chroma

# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA
# from deep_translator import GoogleTranslator


# # ✅ Language name → ISO code mapping
# LANG_CODE_MAP = {
#     "English": "en",
#     "Hindi": "hi",
#     "Tamil": "ta",
#     "Telugu": "te",
#     "Bengali": "bn",
#     "Punjabi": "pa",
#     "Marathi": "mr",
#     "Gujarati": "gu",
#     "Kannada": "kn",
#     "Malayalam": "ml",
#     "Urdu": "ur",
#     "Odia": "or",
#     "Assamese": "as",
#     "Maithili": "mai",
#     "Konkani": "kok",
#     "Sindhi": "sd",
#     "Santali": "sat",
#     "Kashmiri": "ks",
#     "Dogri": "doi",
#     "Manipuri": "mni",
#     "Bodo": "brx",
#     "Nepali": "ne"
# }


# def humanize_response(query: str, answer: str) -> str:
#     """Improve answers with empathy, symptom advice, and wellness guidance."""

#     symptoms = [
#         "fever", "headache", "cold", "cough", "dizzy", "nausea", "vomit",
#         "diarrhea", "body pain", "chills", "fatigue", "tired", "weak",
#         "sore throat", "breathless", "pain", "loss of appetite", "stomach ache",
#         "indigestion", "rash", "allergy", "swelling", "joint pain", "back pain",
#         "anxiety", "stress", "insomnia", "dehydration", "skin irritation"
#     ]
#     general_health = [
#         "diet", "exercise", "sleep", "healthy", "nutrition",
#         "prevent", "wellness", "fitness", "habits", "lifestyle"
#     ]

#     query_lower = query.lower()

#     if any(symptom in query_lower for symptom in symptoms):
#         return (
#             "🩺 **It sounds like you may be experiencing some health symptoms. Here's general advice:**\n\n"
#             "- Rest as much as possible and avoid overexertion.\n"
#             "- Stay hydrated (water, ORS, herbal teas, light soups).\n"
#             "- Eat light, nutritious food (fruits, vegetables, whole grains).\n"
#             "- Track your temperature and other symptoms regularly.\n"
#             "- Use home remedies for comfort (steam inhalation, warm water, salt gargle).\n"
#             "- Avoid junk, oily, or very spicy food.\n"
#             "- If symptoms persist, worsen, or include high fever, chest pain, or severe breathlessness, seek medical attention promptly.\n\n"
#             "❤️ Take care — I can also provide information from the documents if you'd like."
#         )

#     if any(keyword in query_lower for keyword in general_health):
#         return (
#             "🌱 **Here are some general wellness tips:**\n\n"
#             "- Eat a balanced diet with fresh fruits, vegetables, and proteins.\n"
#             "- Exercise at least 30 minutes daily (walking, yoga, or light workouts).\n"
#             "- Maintain regular sleep (7–8 hours) and avoid excessive screen time.\n"
#             "- Stay hydrated — 8–10 glasses of water a day.\n"
#             "- Avoid smoking, alcohol, and excessive caffeine.\n"
#             "- Practice stress management (deep breathing, meditation, journaling).\n"
#             "- Wash hands regularly and maintain hygiene to prevent infections.\n\n"
#             "💡 Small lifestyle changes build long-term health benefits!"
#         )

#     return (
#         f"{answer}\n\n"
#         "✨ Remember: Health information here is for awareness. Always consult a doctor for diagnosis and treatment."
#     )


# def build_vectordb(persist_dir="./chroma_db"):
#     """Load or build the Chroma vector database with Gemini embeddings."""
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/gemini-embedding-001",
#         google_api_key=os.environ["GEMINI_API_KEY"]
#     )
#     vectordb = Chroma(
#         persist_directory=persist_dir,
#         embedding_function=embeddings
#     )
#     return vectordb


# def query_db(question: str, lang: Optional[str] = "English", persist_dir="./chroma_db") -> str:
#     """Main query pipeline: detect input → translate to English → LLM → answer → translate back."""

#     # Step 1: Translate question → English (so model understands)
#     translated_question = question
#     try:
#         translated_question = GoogleTranslator(source="auto", target="en").translate(question)
#     except Exception as e:
#         print("⚠️ Question translation failed, using original:", e)

#     # Step 2: Run retrieval + LLM in English
#     vectordb = build_vectordb(persist_dir=persist_dir)
#     retriever = vectordb.as_retriever(search_kwargs={"k": 4})

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-1.5-flash",
#         google_api_key=os.environ["GEMINI_API_KEY"]
#     )

#     qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

#     personality = (
#         "You are a helpful, caring, and professional health assistant. "
#         "Always generate answers in English. "
#         "When users mention symptoms like headache, fever, or weakness, respond empathetically "
#         "and provide point-wise, clear health advice. "
#         "Keep the tone kind, supportive, and respectful."
#     )

#     final_question = f"{personality}\n\nQuestion: {translated_question}"

#     out = qa.invoke({"query": final_question})
#     english_answer = out.get("result", out)

#     # Step 3: Humanize response (still in English)
#     english_answer = humanize_response(translated_question, english_answer)

#     # Step 4: Translate English answer → chosen output language
#     if lang not in ["English", None, ""]:
#         target_code = LANG_CODE_MAP.get(lang, "en")
#         try:
#             translated_answer = GoogleTranslator(source="en", target=target_code).translate(english_answer)
#             return translated_answer
#         except Exception as e:
#             print("⚠️ Answer translation failed:", e)
#             return english_answer

#     return english_answer


# if __name__ == "__main__":
#     print("Loaded PDF RAG chatbot — type 'exit' to quit.")
#     lang = input("Select answer language (English/Hindi/Tamil/Telugu/Bengali/etc.): ").strip()
#     if not lang:
#         lang = "English"

#     while True:
#         q = input("\nAsk a question (or type 'exit'): ").strip()
#         if q.lower() in ("exit", "quit"):
#             break
#         ans = query_db(q, lang=lang)
#         print(f"\n🤖 Answer ({lang}):\n{ans}\n")

# # def rag_answer(question: str) -> str:
# #     # Example — adjust to match your code
# #     docs = retriever.get_relevant_documents(question)
# #     result = qa_chain.run(input_documents=docs, question=question)
# #     return result

# import os
# from typing import Optional

# # Fallback for Chroma import
# try:
#     from langchain_chroma import Chroma
# except Exception:
#     from langchain_community.vectorstores import Chroma

# try:
#     from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# except Exception:
#     # Mock classes if Gemini API is unavailable
#     class GoogleGenerativeAIEmbeddings:
#         def __init__(self, **kwargs):
#             print("⚠️ Gemini embeddings not found, using mock embeddings.")

#         def embed_documents(self, docs):
#             # simple dummy embeddings (list of zeros)
#             return [[0.0] * 768 for _ in docs]

#     class ChatGoogleGenerativeAI:
#         def __init__(self, **kwargs):
#             print("⚠️ Gemini LLM not found, using mock LLM.")

#         def __call__(self, prompt, **kwargs):
#             return "⚠️ Mock answer: Gemini API not available."

# from langchain.chains import RetrievalQA
# from deep_translator import GoogleTranslator

# # ✅ Language name → ISO code mapping
# LANG_CODE_MAP = {
#     "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te", "Bengali": "bn",
#     "Punjabi": "pa", "Marathi": "mr", "Gujarati": "gu", "Kannada": "kn",
#     "Malayalam": "ml", "Urdu": "ur", "Odia": "or", "Assamese": "as",
#     "Maithili": "mai", "Konkani": "kok", "Sindhi": "sd", "Santali": "sat",
#     "Kashmiri": "ks", "Dogri": "doi", "Manipuri": "mni", "Bodo": "brx", "Nepali": "ne"
# }

# def humanize_response(query: str, answer: str) -> str:
#     symptoms = [
#         "fever","headache","cold","cough","dizzy","nausea","vomit","diarrhea",
#         "body pain","chills","fatigue","tired","weak","sore throat","breathless",
#         "pain","loss of appetite","stomach ache","indigestion","rash","allergy",
#         "swelling","joint pain","back pain","anxiety","stress","insomnia",
#         "dehydration","skin irritation"
#     ]
#     general_health = [
#         "diet","exercise","sleep","healthy","nutrition","prevent","wellness",
#         "fitness","habits","lifestyle"
#     ]

#     query_lower = query.lower()
#     if any(symptom in query_lower for symptom in symptoms):
#         return (
#             "🩺 **It sounds like you may be experiencing some health symptoms. Here's general advice:**\n\n"
#             "- Rest as much as possible and avoid overexertion.\n"
#             "- Stay hydrated (water, ORS, herbal teas, light soups).\n"
#             "- Eat light, nutritious food (fruits, vegetables, whole grains).\n"
#             "- Track your temperature and other symptoms regularly.\n"
#             "- Use home remedies for comfort (steam inhalation, warm water, salt gargle).\n"
#             "- Avoid junk, oily, or very spicy food.\n"
#             "- If symptoms persist, worsen, or include high fever, chest pain, or severe breathlessness, seek medical attention promptly.\n\n"
#             "❤️ Take care — I can also provide information from the documents if you'd like."
#         )
#     if any(keyword in query_lower for keyword in general_health):
#         return (
#             "🌱 **Here are some general wellness tips:**\n\n"
#             "- Eat a balanced diet with fresh fruits, vegetables, and proteins.\n"
#             "- Exercise at least 30 minutes daily (walking, yoga, or light workouts).\n"
#             "- Maintain regular sleep (7–8 hours) and avoid excessive screen time.\n"
#             "- Stay hydrated — 8–10 glasses of water a day.\n"
#             "- Avoid smoking, alcohol, and excessive caffeine.\n"
#             "- Practice stress management (deep breathing, meditation, journaling).\n"
#             "- Wash hands regularly and maintain hygiene to prevent infections.\n\n"
#             "💡 Small lifestyle changes build long-term health benefits!"
#         )
#     return f"{answer}\n\n✨ Remember: Health information here is for awareness. Always consult a doctor for diagnosis and treatment."

# def build_vectordb(persist_dir="./chroma_db"):
#     """Load or build the Chroma vector database with Gemini embeddings."""
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/gemini-embedding-001",
#         google_api_key=os.environ.get("GEMINI_API_KEY", "")
#     )
#     vectordb = Chroma(
#         persist_directory=persist_dir,
#         embedding_function=embeddings
#     )
#     return vectordb

# def query_db(question: str, lang: Optional[str] = "English", persist_dir="./chroma_db") -> str:
#     # Step 1: Translate question → English
#     translated_question = question
#     try:
#         translated_question = GoogleTranslator(source="auto", target="en").translate(question)
#         print(f"DEBUG: Translated question to English: {translated_question}")
#     except Exception as e:
#         print("⚠️ Question translation failed, using original:", e)

#     # Step 2: LLM + Retrieval
#     vectordb = build_vectordb(persist_dir=persist_dir)
#     retriever = vectordb.as_retriever(search_kwargs={"k": 4})

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-1.5-flash",
#         google_api_key=os.environ.get("GEMINI_API_KEY", "")
#     )

#     qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

#     personality = (
#         "You are a helpful, caring, and professional health assistant. "
#         "Always generate answers in the language of the user. "
#         "When users mention symptoms like headache, fever, or weakness, respond empathetically "
#         "and provide point-wise, clear health advice. "
#         "Keep the tone kind, supportive, and respectful."
#     )

#     final_question = f"{personality}\n\nQuestion: {translated_question}"

#     # Step 2a: Invoke LLM safely
#     try:
#         out = qa.invoke({"query": final_question})
#         if isinstance(out, dict):
#             english_answer = out.get("result") or out.get("output_text") or str(out)
#         else:
#             english_answer = str(out)
#     except Exception as e:
#         print("⚠️ LLM invocation failed:", e)
#         english_answer = "⚠️ Sorry, I couldn’t generate an answer. Please try again."

#     print("DEBUG: LLM output (English):", english_answer)

#     # Step 3: Humanize response (still in English)
#     english_answer = humanize_response(translated_question, english_answer)

#     # Step 4: Translate back → user's language if not English
#     if lang not in ["English", None, ""]:
#         target_code = LANG_CODE_MAP.get(lang, "en")
#         try:
#             translated_answer = GoogleTranslator(source="en", target=target_code).translate(english_answer)
#             print(f"DEBUG: Translated answer ({lang}):", translated_answer)
#             return translated_answer
#         except Exception as e:
#             print("⚠️ Answer translation failed:", e)
#             return english_answer

#     return english_answer

# # ✅ Main interactive terminal loop
# if __name__ == "__main__":
#     print("💡 Health Assistant Chat (type 'exit' to quit)")
#     lang = input("Enter your language (e.g., English, Hindi): ").strip() or "English"

#     while True:
#         question = input("\nEnter your question: ").strip()
#         if question.lower() in ["exit", "quit"]:
#             print("Goodbye! Stay healthy. ❤️")
#             break

#         answer = query_db(question, lang=lang)
#         print("\n📝 Answer:\n", answer)
