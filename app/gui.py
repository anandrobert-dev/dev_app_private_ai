import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path
import subprocess
import pandas as pd
import logging
from datetime import datetime
from lookup import (
    ingest_table, list_tables, get_table_info,
    search_table, delete_table
)

# ---- Logging setup ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "private_ai.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("private_ai")
log.info("=" * 60)
log.info("APP STARTED")


# ---- Helper: detect if file is structured data ----
STRUCTURED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
RAG_EXTENSIONS = {".pdf", ".txt"}


def is_structured_file(filename: str) -> bool:
    """CSV and Excel files go to lookup tables. Everything else goes to RAG."""
    return Path(filename).suffix.lower() in STRUCTURED_EXTENSIONS


# ---- LLM call with timeout and logging ----
def run_llm_safe(prompt: str, timeout: int = 30) -> str:
    log.info("LLM CALL START (timeout=%ds)", timeout)
    start = datetime.now()
    try:
        result = subprocess.run(
            ["ollama", "run", "mistral:7b-instruct"],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout
        )
        elapsed = (datetime.now() - start).total_seconds()
        log.info("LLM CALL DONE in %.1fs, output length=%d chars", elapsed, len(result.stdout))
        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        elapsed = (datetime.now() - start).total_seconds()
        log.warning("LLM TIMEOUT after %.1fs", elapsed)
        return "LLM took too long to respond. Please refine your question."

    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        log.error("LLM ERROR after %.1fs: %s", elapsed, str(e))
        return f"LLM execution failed: {str(e)}"


# ---- Smart search: check tables first, then RAG ----
def search_all_tables(client_id: str, search_term: str) -> pd.DataFrame:
    """Search ALL tables for a client. Returns combined results."""
    tables = list_tables(client_id)
    all_results = []
    for tbl in tables:
        results = search_table(client_id, tbl, search_term)
        if not results.empty:
            results.insert(0, "_table", tbl)
            all_results.append(results)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


# ---- Extract keywords for table search ----
def extract_search_terms(question: str) -> list:
    """Extract meaningful search terms from a question."""
    # Remove common question words
    stop_words = {
        "what", "is", "the", "a", "an", "does", "do", "how", "why",
        "when", "where", "which", "who", "for", "of", "in", "it",
        "this", "that", "stand", "mean", "meaning", "define",
        "explain", "tell", "me", "about", "can", "you", "please",
        "consider", "considered", "as", "are", "was", "were", "be",
        "to", "and", "or", "not", "from", "with", "by"
    }
    words = question.replace("?", "").replace(".", "").replace(",", "").split()
    terms = [w for w in words if w.lower() not in stop_words and len(w) > 1]
    return terms


# -------------------------------
# Client discovery
# -------------------------------
RAW_BASE_DIR = PROJECT_ROOT / "data_raw"
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
st.set_page_config(page_title="Private AI Knowledge System", layout="wide")
st.title("🔒 Private AI Knowledge System")

if not AVAILABLE_CLIENTS:
    st.error("No clients found in data_raw/")
    st.stop()

# -------------------------------
# Client selector
# -------------------------------
client_id = st.selectbox("Select Client", AVAILABLE_CLIENTS)

# ===============================
# SECTION 1: UPLOAD (SINGLE)
# ===============================
st.markdown("---")
st.subheader("� Upload Documents")
st.caption(
    "Drop any file here. The system auto-detects the type:\n"
    "• **CSV / Excel** → stored as lookup tables (exact searches)\n"
    "• **PDF / TXT / Other** → embedded for AI Q&A (RAG)"
)

uploaded_files = st.file_uploader(
    "Drag and drop files here",
    accept_multiple_files=True,
    key="unified_uploader"
)

if uploaded_files:
    rag_files = []
    table_files = []

    for uploaded_file in uploaded_files:
        if is_structured_file(uploaded_file.name):
            # --- Structured file → Lookup table ---
            temp_path = Path(f"/tmp/{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            result = ingest_table(client_id, temp_path)
            temp_path.unlink(missing_ok=True)

            if result["success"]:
                table_files.append(
                    f"✅ **{uploaded_file.name}** → lookup table "
                    f"`{result['table_name']}` ({result['rows']} rows)"
                )
                log.info("TABLE INGESTED: %s → %s (%d rows)",
                         uploaded_file.name, result['table_name'], result['rows'])
            else:
                table_files.append(
                    f"❌ **{uploaded_file.name}** failed: {result['error']}"
                )
                log.error("TABLE FAILED: %s — %s", uploaded_file.name, result['error'])
        else:
            # --- Everything else → RAG ---
            client_data_dir = PROJECT_ROOT / f"data_raw/{client_id}"
            client_data_dir.mkdir(parents=True, exist_ok=True)
            file_path = client_data_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            rag_files.append(uploaded_file.name)
            log.info("RAG FILE SAVED: %s", uploaded_file.name)

    # Show results
    if table_files:
        st.success("📊 Structured files processed:")
        for msg in table_files:
            st.markdown(msg)

    if rag_files:
        st.success(
            f"📄 {len(rag_files)} document(s) saved for RAG: "
            f"{', '.join(rag_files)}"
        )
        st.info("👉 Click **Re-index** below to make them searchable.")

# Re-index button
if st.button("� Re-index documents for RAG"):
    with st.spinner("Re-indexing documents..."):
        log.info("RE-INDEX STARTED for client: %s", client_id)
        result = subprocess.run(
            ["python", str(PROJECT_ROOT / "app/ingest.py"), client_id],
            text=True,
            capture_output=True
        )

    if result.returncode == 0:
        st.success("Re-index completed successfully.")
        log.info("RE-INDEX DONE: %s", result.stdout.strip())
    else:
        st.error("Re-index failed.")
        st.code(result.stderr)
        log.error("RE-INDEX FAILED: %s", result.stderr)


# ===============================
# SECTION 2: ASK A QUESTION
# ===============================
st.markdown("---")
st.subheader("💬 Ask a Question")
st.caption(
    "Ask anything. The system will:\n"
    "1. Check lookup tables for exact matches\n"
    "2. Search embedded documents for context\n"
    "3. Use AI to explain the answer"
)

# Load embedder
@st.cache_resource
def load_embedder():
    return SentenceTransformer(str(PROJECT_ROOT / "embeddings/bge-base"))

embed_model = load_embedder()

# Check if embeddings exist
vector_dir = PROJECT_ROOT / f"data_vectors/{client_id}"
index_path = vector_dir / "index.faiss"
chunks_path = vector_dir / "chunks.txt"
sources_path = vector_dir / "sources.txt"

has_embeddings = (
    index_path.exists()
    and chunks_path.exists()
    and sources_path.exists()
)

has_tables = len(list_tables(client_id)) > 0

if has_embeddings:
    index = faiss.read_index(str(index_path))

    with open(chunks_path) as f:
        chunks = [line.strip() for line in f if line.strip()]

    with open(sources_path) as f:
        sources = [line.strip() for line in f if line.strip()]

can_ask = has_embeddings or has_tables

with st.form("ask_form"):
    question = st.text_input(
        "Question",
        placeholder=(
            "Type your question here..."
            if can_ask
            else "Upload documents first to enable questions"
        ),
        disabled=not can_ask
    )

    ask_clicked = st.form_submit_button("Ask", disabled=not can_ask)

if not can_ask:
    st.warning(
        "No data found for this client. "
        "Upload documents above to get started."
    )

if ask_clicked and can_ask and question:
    log.info("--- NEW QUESTION ---")
    log.info("Client: %s", client_id)
    log.info("Question: %s", question)

    with st.spinner("Searching..."):

        # ---- STEP 1: Check lookup tables ----
        table_results = pd.DataFrame()
        if has_tables:
            search_terms = extract_search_terms(question)
            log.info("Step 1 - Searching tables for: %s", search_terms)

            for term in search_terms:
                results = search_all_tables(client_id, term)
                if not results.empty:
                    table_results = pd.concat(
                        [table_results, results], ignore_index=True
                    ).drop_duplicates()

            if not table_results.empty:
                log.info("Step 1 - Found %d table matches", len(table_results))

        # ---- STEP 2: Search RAG (if embeddings exist) ----
        rag_chunks_with_sources = []
        rag_context = ""
        if has_embeddings:
            q_embedding = embed_model.encode([question])
            D, I = index.search(np.array(q_embedding), k=5)

            # Build chunks with source info and citation IDs
            seen = set()
            for i, idx in enumerate(I[0]):
                chunk_text = chunks[idx]
                source_name = sources[idx] if idx < len(sources) else "unknown"
                if chunk_text not in seen:
                    seen.add(chunk_text)
                    rag_chunks_with_sources.append({
                        "id": len(rag_chunks_with_sources) + 1,
                        "text": chunk_text,
                        "source": source_name
                    })

            # Create context string with citation markers like [1], [2]
            rag_context = ""
            for item in rag_chunks_with_sources:
                rag_context += f"[{item['id']}] (Source: {item['source']}) {item['text']}\n\n"
            log.info("Step 2 - RAG search done, %d chunks, %d chars",
                     len(rag_chunks_with_sources), len(rag_context))

            MAX_CONTEXT_CHARS = 2000
            if len(rag_context) > MAX_CONTEXT_CHARS:
                rag_context = rag_context[:MAX_CONTEXT_CHARS]
                log.warning("Step 2b - Context TRUNCATED to %d chars", MAX_CONTEXT_CHARS)

    # ---- DISPLAY RESULTS ----

    # Show table results if found
    if not table_results.empty:
        st.subheader("📊 From Lookup Tables")
        st.dataframe(table_results, use_container_width=True)
        log.info("Step 3 - Displayed %d table results", len(table_results))

    # Show RAG context WITH SOURCES
    if rag_chunks_with_sources:
        st.subheader("📄 Verified Document Context")
        st.caption("Information retrieved directly from your uploaded files:")
        for chunk_info in rag_chunks_with_sources:
            with st.expander(f"Reference [{chunk_info['id']}] — {chunk_info['source']}", expanded=True):
                st.text(chunk_info["text"])
    else:
        log.info("Step 3b - No RAG context found")

    # ---- STEP 3: LLM explanation ----
    # Build combined context from both sources
    combined_context = ""

    if not table_results.empty:
        combined_context += "LOOKUP TABLE DATA:\n"
        combined_context += table_results.to_string(index=False)
        combined_context += "\n\n"

    if rag_context.strip():
        combined_context += "DOCUMENT CONTEXT:\n"
        combined_context += rag_context

    if combined_context.strip():
        prompt = f"""You are an internal audit assistant.

STRICT RULES:
- Answer ONLY using the exact information from the context below.
- Each sentence in your answer MUST cite its reference using the bracketed number, e.g., "Consolidators group shipments [1]."
- If the context says "FCL", you must say "FCL" — never change abbreviations or terminology.
- If the answer is not in the context, say "Not found in the provided documents."
- Be extremely factual and concise.

Context:
{combined_context}

Question:
{question}
"""

        log.info("Step 4 - Sending prompt to LLM (%d chars)", len(prompt))
        answer = run_llm_safe(prompt, timeout=30)
        log.info("Step 5 - Answer received (%d chars)", len(answer))
        log.info("Answer: %s", answer[:200])

        st.subheader("🧠 AI Explanation")
        st.caption("⚠️ Generated by AI based on the documents above. Always verify against source.")
        st.write(answer)
    else:
        st.warning("No relevant data found. Please upload documents and re-index.")



# ===============================
# SECTION 3: ADMIN (collapsed)
# ===============================
st.markdown("---")
with st.expander("⚙️ Admin Tools"):

    # Show existing tables
    tables = list_tables(client_id)
    if tables:
        st.markdown("**📋 Existing Lookup Tables:**")
        for tbl in tables:
            info = get_table_info(client_id, tbl)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f"`{tbl}` — {info['rows']} rows, "
                    f"columns: {', '.join(info['columns'])}"
                )
            with col2:
                if st.button(f"🗑️ Delete", key=f"del_{tbl}"):
                    delete_table(client_id, tbl)
                    st.rerun()

    # Reset embeddings
    st.markdown("---")
    st.markdown("**⚠️ Reset Embeddings**")
    confirm_reset = st.checkbox(
        "I understand this will delete embeddings and require re-indexing"
    )

    if st.button("Reset embeddings (current client)"):
        if not confirm_reset:
            st.warning("Please confirm before resetting embeddings.")
        else:
            vector_dir = PROJECT_ROOT / f"data_vectors/{client_id}"

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
