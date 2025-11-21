import os
import time
import pickle
import streamlit as st
from dotenv import load_dotenv

# LangChain Imports (2025 safe versions)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from openai import OpenAIError

# ----------------------------------------------------
# Load Environment
# ----------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    st.error("⚠️ OPENAI_API_KEY is missing. Please set it in Streamlit Secrets.")
    st.stop()

# ----------------------------------------------------
# Page UI
# ----------------------------------------------------
st.title("📈 News Research Tool (2025 Working Version)")
st.sidebar.title("Enter News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():
        urls.append(url.strip())

process_url_clicked = st.sidebar.button("Process URLs")

file_path = "faiss_store_openai.pkl"
status_box = st.empty()

# ----------------------------------------------------
# Initialize LLM
# ----------------------------------------------------
llm = ChatOpenAI(
    temperature=0.0,
    max_tokens=400,
    api_key=OPENAI_API_KEY
)

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# ----------------------------------------------------
# Safe Embedding Function (Avoids Rate Limits)
# ----------------------------------------------------
def embed_with_retry(docs, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            return embeddings.embed_documents(docs)
        except OpenAIError:
            sleep_time = 2 ** attempt
            status_box.text(f"⚠️ Rate limit hit. Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
    raise Exception("Embedding failed after multiple retries.")

# ----------------------------------------------------
# PROCESS URLS → BUILD VECTOR STORE
# ----------------------------------------------------
if process_url_clicked:
    if len(urls) == 0:
        st.warning("⚠️ Please enter at least 1 valid URL.")
        st.stop()

    status_box.text("🔄 Loading data from URLs...")

    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    status_box.text("✂️ Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = splitter.split_documents(data)

    status_box.text("🧠 Creating embeddings (this may take 10–20 sec)...")

    text_list = [d.page_content for d in docs]
    vecs = embed_with_retry(text_list)

    status_box.text("📦 Building FAISS vector store...")

    vectorstore = FAISS.from_embeddings(
        embeddings=vecs,
        metadatas=[d.metadata for d in docs],
        embedding=embeddings
    )

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore.serialize_to_bytes(), f)

    status_box.text("✅ Processing Complete! You can now ask questions.")

# ----------------------------------------------------
# QUESTION INPUT
# ----------------------------------------------------
query = st.text_input("Ask a question about the processed articles:")

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

    # Retrieve relevant sections
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
