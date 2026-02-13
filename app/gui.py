import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path
import subprocess
import shlex
from langchain_community.llms import Ollama


def run_llm_safe(prompt: str, timeout: int = 15) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", "mistral:7b-instruct"],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout
        )
        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        return "LLM took too long to respond. Please refine your question."

    except Exception as e:
        return f"LLM execution failed: {str(e)}"


def detect_intent(question: str) -> str:
    q = question.lower()

    lookup_triggers = [
        "what is",
        "what does",
        "define",
        "meaning of",
        "stand for",
        "code for",
        "scac",
        "gl",
    ]

    explain_triggers = [
        "why",
        "when",
        "how",
        "explain",
        "under what",
        "reason",
    ]

    for t in lookup_triggers:
        if t in q:
            return "lookup"

    for t in explain_triggers:
        if t in q:
            return "explain"

    # default (safe)
    return "lookup"


llm = Ollama(
    model="mistral:7b-instruct",
    temperature=0
)


# -------------------------------
# Client discovery
# -------------------------------
RAW_BASE_DIR = Path("data_raw")
RAW_BASE_DIR.mkdir(exist_ok=True)

AVAILABLE_CLIENTS = sorted(
    [p.name for p in RAW_BASE_DIR.iterdir() if p.is_dir()]
)

if not AVAILABLE_CLIENTS:
    st.warning(
        "No clients found yet.\n"
        "Please create a client folder under data_raw/ to begin."
    )
    st.stop()


# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Private AI Knowledge System", layout="centered")
st.title("Private AI Knowledge System")

if not AVAILABLE_CLIENTS:
    st.error("No clients found in data_vectors/")
    st.stop()

# -------------------------------
# Client selector (MUST COME FIRST)
# -------------------------------
client_id = st.selectbox("Select Client", AVAILABLE_CLIENTS)

# -------------------------------
# Admin upload section (NOW SAFE)
# -------------------------------
st.markdown("### Admin: Upload Documents")

uploaded_files = st.file_uploader(
    "Upload documents for selected client",
    type=["pdf", "txt", "csv", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    client_data_dir = Path(f"data_raw/{client_id}")
    client_data_dir.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        file_path = client_data_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success("Files uploaded successfully. Please re-index to use them.")

if st.button("Re-index now"):
    with st.spinner("Re-indexing documents..."):
        result = subprocess.run(
            ["python", "app/ingest.py"],
            text=True,
            capture_output=True
        )

    if result.returncode == 0:
        st.success("Re-index completed successfully.")
        st.info("You can now ask questions based on the newly uploaded documents.")
    else:
        st.error("Re-index failed.")
        st.code(result.stderr)

st.markdown("### ⚠️ Admin: Reset Embeddings")

confirm_reset = st.checkbox(
    "I understand this will delete embeddings and require re-indexing"
)

if st.button("Reset embeddings (current client)"):
    if not confirm_reset:
        st.warning("Please confirm before resetting embeddings.")
    else:
        vector_dir = Path(f"data_vectors/{client_id}")

        if vector_dir.exists():
            with st.spinner("Deleting embeddings..."):
                for item in vector_dir.iterdir():
                    item.unlink()
                vector_dir.rmdir()

            st.success(
                f"Embeddings for client '{client_id}' have been reset.\n"
                "Please re-index to restore knowledge."
            )
        else:
            st.info("No embeddings found for this client.")


# -------------------------------
# Load models & vectors
# -------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("./embeddings/bge-base")

embed_model = load_embedder()

vector_dir = Path(f"data_vectors/{client_id}")
index_path = vector_dir / "index.faiss"
chunks_path = vector_dir / "chunks.txt"
sources_path = vector_dir / "sources.txt"

has_embeddings = (
    index_path.exists()
    and chunks_path.exists()
    and sources_path.exists()
)

if has_embeddings:
    index = faiss.read_index(str(index_path))

    with open(chunks_path) as f:
        chunks = [line.strip() for line in f if line.strip()]

    with open(sources_path) as f:
        sources = [line.strip() for line in f if line.strip()]

# -------------------------------
# Question answering (SINGLE SOURCE OF TRUTH)
# -------------------------------
st.subheader("Ask a question from client documents")

question = st.text_input(
    "Question",
    placeholder=(
        "Type your question here"
        if has_embeddings
        else "Upload documents and re-index to enable questions"
    ),
    disabled=not has_embeddings
)

ask_clicked = st.button("Ask", disabled=not has_embeddings)

if not has_embeddings:
    st.warning(
        "No embeddings found for this client. "
        "Please upload documents and click 'Re-index' before asking questions."
    )

if ask_clicked and has_embeddings and question:
    with st.spinner("Searching documents and generating answer..."):

        intent = detect_intent(question)

        q_embedding = embed_model.encode([question])

        if intent == "lookup":
            st.info("Using lookup path (no deep reasoning).")
            D, I = index.search(np.array(q_embedding), k=3)
        else:
            st.info("Using explanation path (RAG + LLM).")
            D, I = index.search(np.array(q_embedding), k=8)

        st.info(f"Retrieved {len(I[0])} matching chunks from knowledge base.")

        selected_chunks = list(dict.fromkeys([chunks[i] for i in I[0]]))
        context = "\n".join(selected_chunks)

        MAX_CONTEXT_CHARS = 2000
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS]
            st.warning("Context truncated to prevent system overload.")

        st.subheader("🔍 Retrieved Context")
        if not context.strip():
            st.warning("No relevant content found in documents.")
        else:
            st.text(context)

        prompt = f"""
You are an internal audit assistant.

RULES:
- Answer ONLY from the context
- If not found, say "Not found in documents"
- Keep the answer under 120 words
- Be concise and factual

Context:
{context}

Question:
{question}
"""

        answer = run_llm_safe(prompt, timeout=15)

        st.subheader("🧠 Answer")
        st.write(answer)


