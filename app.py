import os
import warnings
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
MAX_FILE_SIZE_MB=1
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
MODEL_NAME = "deepseek/deepseek-chat"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MAX_CONTEXT_CHUNKS = 5  # Cap the number of chunks used in a prompt
MAX_TOKENS_PER_CHUNK = 1000  # Limit characters per chunk for API safety

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
        file_size_mb = len(file.getbuffer()) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.sidebar.warning(f"⚠️ {file.name} is too large ({file_size_mb:.2f} MB). Max allowed is {MAX_FILE_SIZE_MB} MB.")
            continue        
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

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata["source"] = file.name
            document_chunks.extend(chunks)

            if show_preview and chunks:
                with st.sidebar.expander(f"Preview: {file.name}"):
                    if file.name.endswith((".xlsx", ".xls")):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            df = pd.read_excel(file_path, engine="openpyxl")
                        st.dataframe(df.head(3))
                    else:
                        st.text(chunks[0].page_content[:500] + "...")

        except Exception as e:
            st.sidebar.error(f"❗ Error processing {file.name}: {e}")

    if document_chunks:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(document_chunks, embeddings)
        st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        st.sidebar.info(f"🔍 Total chunks created: {len(document_chunks)}")
        st.sidebar.success(f"✅ Processed {len(uploaded_files)} document(s)")
    else:
        st.sidebar.warning("⚠️ No valid content was processed.")

# Main UI
st.markdown(
    """
    <div style="background-color:#2F4F4F;padding:25px;border-radius:10px">
        <h1 style="color:white;text-align:center;">SmartDocAI</h1>
        <h3 style="color:white;text-align:center;">
            Upload. Ask. Unlock insights.
        </h3>
        <p style="color:white;text-align:center;">
            Built by <a href="https://www.linkedin.com/in/leadtherightway/" target="_blank" style="color:#1E90FF;">Pratyush Ranjan Mishra</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.title("🤖 Ask Questions Based on Your Documents")

with st.expander("ℹ️ Help / About this App"):
    st.markdown("""
🚀 **Prototype developed by Pratyush Ranjan Mishra**

This intelligent assistant is designed to help **business analysts, developers, and managers** drastically speed up development, planning, and documentation tasks.

### 💡 What It Can Do:
- 📄 Ingests documents (Excel, Word, PDF, Text, Java class files) and answers context-based questions  
- 🧠 Understands Java class structures and generates integration-ready code  
- 🏗️ Suggests backend services, APIs, and frameworks tailored to your existing stack  
- 📊 Assists with Excel-based data insights, conversion tasks, and mappings  
- ✍️ Generates technical documentation, user stories, architecture narratives  
- 🎯 Creates PowerPoint bullet points and storylines for executive presentations  
- 🧩 Can even generate Python code to create diagrams or flowcharts  
- 👨‍💼 Helps managers with roadmap planning, effort estimates, and backlog organization  

### ✅ How to Use:
1. Upload documents using the sidebar — supported types: `.txt`, `.pdf`, `.docx`, `.xlsx`, etc.  
2. (Optional) Preview content or view processed chunk details via sidebar toggles  
3. Enter your natural-language query in the main text box (e.g., _"Generate Java service class based on this POJO"_)  
4. Hit **Submit** — your custom AI engine will reply with context-aware answers  
5. View response and reference source chunks if needed  
6. You can ask for help in the prompt for better use of this.

> This is a **working prototype** and a glimpse into how AI is transforming modern development practices.

Developed with ❤️ by **Pratyush Ranjan Mishra**
""")
 

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

        # Dynamically trim context to fit token budget
        total_chars = 0
        max_chars = MAX_CONTEXT_CHUNKS * MAX_TOKENS_PER_CHUNK
        trimmed_docs = []

        for doc in docs:
            content = doc.page_content
            if total_chars + len(content) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 0:
                    doc.page_content = content[:remaining]
                    trimmed_docs.append(doc)
                break
            trimmed_docs.append(doc)
            total_chars += len(content)

        docs = trimmed_docs

        if show_chunks and docs:
            with st.expander("📄 Retrieved Chunk Details"):
                for i, doc in enumerate(docs):
                    source = doc.metadata.get("source", "Unknown")
                    st.markdown(f"**Chunk {i+1}:** from `{source}`")
                    st.text(doc.page_content[:200] + "...\n")

        context = "\n\n".join(doc.page_content[:MAX_TOKENS_PER_CHUNK] for doc in docs)
        system_message = (
            "You are a helpful assistant that understands the purpose and capabilities of this prototype..."
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
        "temperature": temperature,
        "max_tokens": 2048
    }

    try:
        response = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
        st.markdown("### 🤖 Response")
        st.write(reply)

        if use_document_context and docs:
            st.markdown("### 📄 Source Chunks")
            for i, doc in enumerate(docs):
                st.markdown(f"**Document {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                st.text(doc.page_content[:300] + "...\n")
    except requests.exceptions.HTTPError:
        st.error("⚠️ API request failed: You may have exceeded your token or credit limit.")
    except Exception:
        st.error("⚠️ An unexpected error occurred. Please try again or check your logs.")
