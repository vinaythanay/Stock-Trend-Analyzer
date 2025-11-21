import os
import time
import pickle
import streamlit as st
from dotenv import load_dotenv

# LangChain Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Document Loading + Vector DB
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# -------------------------------------------------------------------
# Load ENV Variables
# -------------------------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    st.error("⚠️ GOOGLE_API_KEY missing! Add it in Streamlit Secrets.")
    st.stop()

# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.title("📈 News Research Tool — Gemini Powered")
st.sidebar.title("Enter News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():
        urls.append(url.strip())

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_gemini.pkl"
status_box = st.empty()

# -------------------------------------------------------------------
# Gemini LLM + Embeddings
# -------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY
)

# -------------------------------------------------------------------
# PROCESS URLs → BUILD VECTOR DB
# -------------------------------------------------------------------
if process_url_clicked:
    if len(urls) == 0:
        st.warning("⚠️ Please enter at least 1 URL.")
        st.stop()

    status_box.text("🔄 Fetching article data...")
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    status_box.text("✂️ Splitting text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = splitter.split_documents(data)

    status_box.text("🧠 Generating Gemini Embeddings...")
    text_list = [d.page_content for d in docs]

    # Generate all embeddings
    vectors = embeddings.embed_documents(text_list)

    status_box.text("📦 Building FAISS Vector DB...")
    vectorstore = FAISS.from_embeddings(
        embeddings=vectors,
        metadatas=[d.metadata for d in docs],
        embedding=embeddings
    )

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore.serialize_to_bytes(), f)

    status_box.text("✅ URLs processed successfully! Ask a question below.")

# -------------------------------------------------------------------
# QUESTION BOX
# -------------------------------------------------------------------
query = st.text_input("Ask a question about the articles:")

if query:
    if not os.path.exists(file_path):
        st.error("⚠️ No FAISS store found. Please process URLs first.")
        st.stop()

    with open(file_path, "rb") as f:
        serialized = pickle.load(f)

    vectorstore = FAISS.deserialize_from_bytes(
        serialized=serialized,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a news analyst. Use ONLY the context below.

Context:
{context}

Question: {query}

Answer clearly and concisely:
"""

    answer = llm.predict(prompt)

    st.header("🧾 Answer")
    st.write(answer)

    st.subheader("🔗 Sources")
    for d in docs:
        st.write(d.metadata.get("source", "Unknown source"))
