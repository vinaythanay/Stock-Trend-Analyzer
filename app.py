import os
import time
import pickle
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# FREE local embedding model
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Chat model ONLY — no embeddings here
llm = ChatOpenAI(
    temperature=0.7,
    max_tokens=500,
    api_key=OPENAI_API_KEY
)

# FREE embeddings (local model)
embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-base"
)

urls = []
for i in range(3):
    urls.append(st.sidebar.text_input(f"URL {i+1}"))

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"

main_placeholder = st.empty()

# -------------------- PROCESSING --------------------
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Loading data...")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50
    )

    main_placeholder.text("Splitting text...")
    docs = text_splitter.split_documents(data)

    # ❗ LOCAL EMBEDDINGS – No rate limit, no API cost
    main_placeholder.text("Building FAISS index (local embeddings)...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore.serialize_to_bytes(), f)

    main_placeholder.text("FAISS index saved successfully! ✅")

# -------------------- QUERY SECTION --------------------
query = main_placeholder.text_input("Question:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            pkl = pickle.load(f)

        vectorstore = FAISS.deserialize_from_bytes(
            serialized=pkl,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(query)

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
You are an AI news analyst. Use ONLY this context:

{context}

Question: {query}

Answer:
"""

        answer = llm.predict(prompt)

        st.header("Answer")
        st.write(answer)

        st.subheader("Sources")
        for d in docs:
            st.write(d.metadata.get("source", "Unknown source"))
