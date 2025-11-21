import os
import streamlit as st
import pickle
import time


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")
llm = ChatOpenAI(
    temperature=0.9,
    max_tokens=500,
    api_key=OPENAI_API_KEY
)
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()

llm = ChatOpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅")
    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    pkl = vectorstore_openai.serialize_to_bytes()

    main_placeholder.text("Embedding Vector Building...✅")
    time.sleep(2)

    with open(file_path, "wb") as f:
        pickle.dump(pkl, f)

# ----------- QUERY SECTION -------------
query = main_placeholder.text_input("Question:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            pkl = pickle.load(f)

            vectorstore = FAISS.deserialize_from_bytes(
                embeddings=OpenAIEmbeddings(),
                serialized=pkl,
                allow_dangerous_deserialization=True
            )

            retriever = vectorstore.as_retriever()
            docs = retriever.get_relevant_documents(query)

            # Build answer with sources
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

            # Display sources
            st.subheader("Sources")
            for d in docs:
                st.write(d.metadata.get("source", "Unknown source"))
