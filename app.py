import os
import time
import pickle
import streamlit as st
from dotenv import load_dotenv

# LangChain Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Loaders + Vector DB
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ----------------------------------------------------
# Load ENV
# ----------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY missing in environment variables.")
    st.stop()

# ----------------------------------------------------
# Streamlit UI
# ----------------------------------------------------
st.title("📈 News Research Tool — Gemini Powered")
st.sidebar.header("Enter News Article URLs")

urls = []
for i in range(3):
    u = st.sidebar.text_input(f"URL {i+1}")
    if u.strip():
        urls.append(u.strip())

process_url = st.sidebar.button("Process URLs")
status_box = st.empty()

file_path = "faiss_store_gemini.pkl"

# ----------------------------------------------------
# Gemini LLM + Embeddings
# ----------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY
)

# ----------------------------------------------------
# PROCESS URLS
# ----------------------------------------------------
if process_url:
    if len(urls) == 0:
        st.warning("⚠️ Enter at least 1 URL.")
        st.stop()

    status_box.text("🔄 Loading articles...")

    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    status_box.text("✂️ Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = splitter.split_documents(data)

    text_list = [d.page_content for d in docs]

    status_box.text("🧠 Creating embeddings...")
    vectorstore = FAISS.from_texts(
        texts=text_list,
        embedding=embeddings,
        metadatas=[d.metadata for d in docs]
    )

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore.serialize_to_bytes(), f)

    status_box.text("✅ Processing Complete! You can now ask questions.")

# ----------------------------------------------------
# QUERY SECTION
# ----------------------------------------------------
query = st.text_input("Ask a question:")

if query:
    if not os.path.exists(file_path):
        st.error("⚠️ Process URLs first!")
        st.stop()

    with open(file_path, "rb") as f:
        serialized = pickle.load(f)

    vectorstore = FAISS.deserialize_from_bytes(
        serialized=serialized,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Use ONLY the context below to answer the question.

Context:
{context}

Question: {query}

Answer clearly:
"""

    answer = llm.predict(prompt)

    st.header("🧾 Answer")
    st.write(answer)

    st.subheader("🔗 Sources")
    for d in docs:
        st.write(d.metadata.get("source", "Unknown"))
