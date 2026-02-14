from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path
import subprocess

CLIENT_ID = "client_demo"

VECTOR_DIR = Path(f"data_vectors/{CLIENT_ID}")

# Load embedding model
embed_model = SentenceTransformer("./embeddings/bge-base")

# Load FAISS index
index = faiss.read_index(str(VECTOR_DIR / "index.faiss"))

# Load source texts
with open(VECTOR_DIR / "sources.txt") as f:
    sources = [line.strip() for line in f.readlines()]

# Load original chunks again (simple approach for now)
with open(VECTOR_DIR / "chunks.txt") as f:
    chunks = [line.strip() for line in f.readlines()]


# User question
question = "What is the penalty for late invoices?"

# Embed question
q_embedding = embed_model.encode([question])

# Search memory
D, I = index.search(np.array(q_embedding), k=2)

selected_chunks = list(dict.fromkeys([chunks[i] for i in I[0]]))

print("\n--- RETRIEVED CONTEXT ---")
for c in selected_chunks:
    print("-", c)

context = "\n".join(selected_chunks)


prompt = f"""
You are an internal audit assistant.

RULES:
- Answer ONLY from the context below
- Include ALL relevant conditions, time limits, and percentages
- Do NOT shorten the answer
- If information is missing, say "Not found in documents"

Context:
{context}

Question:
{question}

Answer in complete sentences.
"""

# Call Ollama
result = subprocess.run(
    ["ollama", "run", "mistral:7b-instruct"],
    input=prompt,
    text=True,
    capture_output=True
)

print("\n--- AI ANSWER ---\n")
print(result.stdout)
