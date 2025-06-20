import os
import requests
import streamlit as st
import pandas as pd

from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
)
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
MODEL_NAME = "deepseek/deepseek-chat"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Sidebar interface
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_files = st.file_uploader(
        "Choose files to upload (PDF, Word, Excel, Text):",
        type=["pdf", "docx", "xlsx", "xls", "txt"],
        accept_multiple_files=True,
        key="user_files"
    )
    upload_trigger = st.button("📥 Upload & Process")

    st.header("⚙️ Model Settings")
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7)
    use_document_context = st.checkbox("Use document context", value=True)
    show_preview = st.checkbox("Show document preview", value=False)
    show_chunks = st.checkbox("Show retrieved chunks", value=False)

    st.markdown(
        """
        <div style="margin-top: 2em; font-size: 0.8em; color: gray; text-align: center;">
            Built by <strong>Pratyush Ranjan Mishra</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

# Initialize session retriever
st.session_state.setdefault("retriever", None)

# Document ingestion triggered manually
if upload_trigger and uploaded_files:
    document_chunks = []

    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        try:
            # Select the appropriate loader
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file.name.endswith((".xlsx", ".xls")):
                loader = UnstructuredExcelLoader(file_path)
            else:
                loader = UnstructuredLoader(file_path)

            docs = loader.load()
            if not docs:
                st.sidebar.warning(f"⚠️ No content loaded from {file.name}")
                continue

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
            chunks = splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata["source"] = file.name
            document_chunks.extend(chunks)

            if show_preview and chunks:
                with st.sidebar.expander(f"Preview: {file.name}"):
                    if file.name.endswith((".xlsx", ".xls")):
                        df = pd.read_excel(file_path)
                        st.dataframe(df.head(3))
                    else:
                        st.text(chunks[0].page_content[:500] + "...")

        except Exception as e:
            st.sidebar.error(f"❗ Error processing {file.name}: {repr(e)}")

    if document_chunks:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(document_chunks, embeddings)
        st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        st.sidebar.info(f"🔍 Total chunks created: {len(document_chunks)}")
        st.sidebar.success(f"✅ Processed {len(uploaded_files)} document(s)")
    else:
        st.sidebar.warning("⚠️ No valid content was processed.")

# Main UI
st.title("🤖 Ask Questions Based on Your Documents")
st.markdown(
    "<div style='text-align: right; font-size: 0.85em; color: gray;'>Built by <strong>Pratyush Ranjan Mishra</strong></div>",
    unsafe_allow_html=True
)

with st.form(key="query_form"):
    query = st.text_area("Enter your question or instruction:", height=150)
    submitted = st.form_submit_button("🚀 Submit")

if submitted and query.strip():
    docs = []

    if use_document_context and st.session_state.retriever:
        docs = st.session_state.retriever.invoke(query)

        if show_chunks and docs:
            with st.expander("📄 Retrieved Chunk Details"):
                for i, doc in enumerate(docs):
                    source = doc.metadata.get("source", "Unknown")
                    st.markdown(f"**Chunk {i+1}:** from `{source}`")
                    st.text(doc.page_content[:200] + "...\n")

        context = "\n\n".join(doc.page_content for doc in docs)
        system_message = (
            "You are a helpful assistant that answers questions based on the following context. "
            "If the answer cannot be found, respond as best you can."
        )
        full_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    else:
        system_message = "You are a helpful assistant."
        full_prompt = query

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": full_prompt}
        ],
        "temperature": temperature
    }

    response = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload)

    if response.status_code == 200:
        reply = response.json()["choices"][0]["message"]["content"]
        st.markdown("### 🤖 Response")
        st.write(reply)

        if use_document_context and docs:
            st.markdown("### 📄 Source Chunks")
            for i, doc in enumerate(docs):
                st.markdown(f"**Document {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                st.text(doc.page_content[:300] + "...\n")
    else:
        st.error(f"OpenRouter API Error: {response.text}")
