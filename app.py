import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import time

load_dotenv()

st.title("📈 Stock Trend Analyzer (Stable Version)")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Missing API key: Add OPENAI_API_KEY to your environment variables.")
    st.stop()

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ----- Safe Embed Wrapper -----

def embed_with_retry(text_list, max_retries=3):
    for attempt in range(max_retries):
        try:
            st.write(f"Embedding... (Attempt {attempt + 1})")
            vectors = embeddings.embed_documents(text_list)
            return vectors
        except Exception as e:
            st.warning(f"Embedding failed: {e}")
            time.sleep(2)

    raise Exception("Embedding failed after multiple retries.")

# ----- UI -----

text_input = st.text_area("Enter company info or stock-related text:")

if st.button("Generate Embeddings"):
    if not text_input.strip():
        st.warning("Please enter some text.")
        st.stop()

    # split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(text_input)

    st.write(f"🔹 Total Chunks: {len(chunks)}")

    try:
        vecs = embed_with_retry(chunks)
        st.success("Embeddings generated successfully!")
        st.json({"chunks": chunks, "vectors": vecs[:2]})  # show first 2 vectors
    except Exception as e:
        st.error(f"Final Error: {e}")
