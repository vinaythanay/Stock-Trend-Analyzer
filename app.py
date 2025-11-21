import os
import time
import pickle
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from tenacity import retry, wait_random_exponential, stop_after_attempt

# ------------------- CONFIG --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# LLM
llm = ChatOpenAI(
    temperature=0.7,
    max_tokens=500,
    api_key=OPENAI_API_KEY
)

# ------------------- INPUT --------------------
urls = []
for i in range(3):
    urls.append(st.sidebar.text_input(f"URL {i+1}"))

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()

# ------------- RETRY-WRAPPED EMBEDDINGS (IMPORTANT) ----------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # faster, cheaper, higher limits
    api_key=OPENAI_API_KEY
)

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def embed_with_retry(text_list):
    return embeddings.embed_documents(text_list)

# ------------------- PROCESSING --------------------
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50
    )

    main_placeholder.text("Text Splitter...Started...✅")
    docs = text_splitter.split_documents(data)

    # ----------- STABLE EMBEDDING WITH BATCHES ----------
    main_placeholder.text("Embedding Vector Building... (No Rate Limits) ...")
    vectors = []
    batch_size = 20

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        texts = [d.page_content for d in batch]
        vecs = embed_with_retry(texts)
        vectors.extend(vecs)
        time.sleep(0.2)  # smooth out API load

    vectorstore_openai = FAISS.from_embeddings(vectors, docs)

    # serialize
    pkl = vectorstore_openai.serialize_to_bytes()
    with open(file_path, "wb") as f:
        pickle.dump(pkl, f)

    main_placeholder.text("Embedding Vector Building...Done ✅")

# ----------------- QUERY SECTION ---------------------
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
You are an AI news analyst. Use ONLY the following context to answer the question.

Context:
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
