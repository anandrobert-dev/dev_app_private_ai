from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path
import pandas as pd
from pypdf import PdfReader

# Paths
CLIENT_ID = "client_demo"

DATA_DIR = Path(f"data_raw/{CLIENT_ID}")
VECTOR_DIR = Path(f"data_vectors/{CLIENT_ID}")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

# Load embedding model
model = SentenceTransformer("./embeddings/bge-base")

documents = []
sources = []

# Read files
for file in DATA_DIR.iterdir():

    text = ""

    if file.suffix == ".txt":
        text = file.read_text()

    elif file.suffix == ".pdf":
        try:
            reader = PdfReader(str(file))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
        except Exception:
            print(f"Skipping invalid PDF: {file.name}")
            continue

    elif file.suffix in [".csv", ".xlsx"]:
        df = pd.read_csv(file) if file.suffix == ".csv" else pd.read_excel(file)
        text = " ".join(df.astype(str).values.flatten())

    if not text.strip():
        continue

    sentences = [s.strip() for s in text.split(".") if s.strip()]

    buffer = ""
    for sentence in sentences:
        buffer += sentence + ". "
        if len(buffer) > 120:
            documents.append(buffer.strip())
            sources.append(file.name)
            buffer = ""

    if buffer:
        documents.append(buffer.strip())
        sources.append(file.name)

# Create embeddings
embeddings = model.encode(documents)

# Store in FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

faiss.write_index(index, str(VECTOR_DIR / "index.faiss"))

with open(VECTOR_DIR / "sources.txt", "w") as f:
    for src in sources:
        f.write(src + "\n")

with open(VECTOR_DIR / "chunks.txt", "w") as f:
    for chunk in documents:
        f.write(chunk.replace("\n", " ") + "\n")

print("Ingestion complete. Chunks:", len(documents))
