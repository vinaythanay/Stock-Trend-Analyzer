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
# Import specific error types for better handling
from openai import OpenAIError, AuthenticationError 

# ----------------------------------------------------
# Load Environment
# ----------------------------------------------------
load_dotenv()
# Note: In Streamlit Cloud, the key should be set via Secrets, not .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    # Use st.toast instead of st.error for a less disruptive message
    st.toast("⚠️ OPENAI_API_KEY is missing. Please set it in Streamlit Secrets.", icon="🔑")
    # Don't st.stop() here, allow the app to render the initial UI
    
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
# Use a dedicated container for status updates
status_container = st.container()

# ----------------------------------------------------
# Initialize LLM
# ----------------------------------------------------
# Note: The LLM and Embeddings objects are initialized here, but 
# we should check if OPENAI_API_KEY is available before use later.
llm = ChatOpenAI(
    temperature=0.0,
    max_tokens=400,
    api_key=OPENAI_API_KEY
)

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# ----------------------------------------------------
# Safe Embedding Function (Avoids Rate Limits and handles Auth errors)
# ----------------------------------------------------
def embed_with_retry(docs, max_attempts=5):
    """Embeds documents with exponential backoff on retryable errors."""
    for attempt in range(max_attempts):
        try:
            return embeddings.embed_documents(docs)
        except AuthenticationError as e:
            # If authentication fails, do not retry, just raise the specific error.
            status_container.error(f"❌ Critical Error: Authentication failed. Check your API key. Details: {e}")
            raise Exception("Authentication failed. Please check your OPENAI_API_KEY.")
        except OpenAIError as e:
            # Handle rate limits or other transient API errors
            sleep_time = 2 ** attempt
            status_container.warning(f"⚠️ OpenAI Error ({e.http_status or 'Unknown'}): Retrying in {sleep_time} seconds (Attempt {attempt+1}/{max_attempts}).")
            # Print the full error to the console logs for detailed debugging
            print(f"Embedding attempt {attempt+1} failed with error: {e}")
            time.sleep(sleep_time)
        except Exception as e:
            # Catch unexpected non-API errors (e.g., network issues)
            status_container.error(f"❌ Unexpected Error: {e}")
            raise Exception(f"Unexpected error during embedding: {e}")

    raise Exception("Embedding failed after multiple retries. Check logs for details.")

# ----------------------------------------------------
# PROCESS URLS → BUILD VECTOR STORE
# ----------------------------------------------------
if process_url_clicked:
    if not OPENAI_API_KEY:
        status_container.error("❌ Cannot proceed: OPENAI_API_KEY is not configured.")
        st.stop()
        
    if len(urls) == 0:
        status_container.warning("⚠️ Please enter at least 1 valid URL.")
        st.stop()

    status_container.info("🔄 Loading data from URLs...")

    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    status_container.info("✂️ Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = splitter.split_documents(data)

    status_container.info("🧠 Creating embeddings (this may take 10–20 sec)...")

    text_list = [d.page_content for d in docs]
    try:
        vecs = embed_with_retry(text_list)
    except Exception as e:
        # Stop processing if embedding failed critically
        status_container.error(f"🛑 Failed to create embeddings: {e}")
        st.stop()


    status_container.info("📦 Building FAISS vector store...")

    # FAISS.from_embeddings expects a list of embedding vectors and their corresponding documents.
    # The structure used here is slightly unusual for FAISS.from_embeddings.
    # A cleaner approach is to use FAISS.from_documents directly, but since your
    # original code uses this structure, let's adapt it to what LangChain expects.
    # Since we manually created vecs, we should pass `text_list` and `vecs` to zip them up.
    
    # Reverting to the recommended standard way to create FAISS from documents:
    # FAISS.from_documents handles splitting the text and calling the embedding
    # model internally, but since you separated it, we need to pass documents.
    
    # The `vectorstore = FAISS.from_embeddings(...)` constructor requires 
    # a list of vectors and a list of documents or texts. 
    # The `vectorstore` expects a list of embeddings and the document objects for metadata.
    
    # We will use the correct method for LangChain's FAISS class:
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )
    
    # Save the vector store as bytes
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore.serialize_to_bytes(), f)

    status_container.success("✅ Processing Complete! You can now ask questions.")

# ----------------------------------------------------
# QUESTION INPUT
# ----------------------------------------------------
query = st.text_input("Ask a question about the processed articles:")

if query:
    if not OPENAI_API_KEY:
        st.error("❌ Cannot proceed: OPENAI_API_KEY is not configured.")
        st.stop()
        
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
