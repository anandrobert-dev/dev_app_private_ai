from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path
import pandas as pd
from pypdf import PdfReader
import sys

# Paths
CLIENT_ID = sys.argv[1] if len(sys.argv) > 1 else "client_demo"

DATA_DIR = Path(f"data_raw/{CLIENT_ID}")
VECTOR_DIR = Path(f"data_vectors/{CLIENT_ID}")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

# Load embedding model
model = SentenceTransformer("./embeddings/bge-base")

documents = []
sources = []

# Read and chunk files
for file in DATA_DIR.iterdir():
    file_text = ""
    print(f"Processing: {file.name} ({file.stat().st_size / 1024:.0f} KB)...")

    if file.suffix == ".txt":
        try:
            file_text = file.read_text(errors="ignore")
        except Exception as e:
            print(f"  ERROR reading text file: {e}")
            continue

    elif file.suffix == ".pdf":
        try:
            reader = PdfReader(str(file))
            total_pages = len(reader.pages)
            print(f"  PDF has {total_pages} pages")

            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        file_text += page_text + " "
                    # Periodically report progress for very large PDFs
                    if (i + 1) % 50 == 0:
                        print(f"    ... extracted {i+1}/{total_pages} pages")
                except Exception as e:
                    print(f"  WARNING: page {i+1} failed: {e}")
                    continue

        except Exception as e:
            print(f"  ERROR reading PDF: {e}")
            continue

    elif file.suffix in [".csv", ".xlsx"]:
        try:
            df = pd.read_csv(file) if file.suffix == ".csv" else pd.read_excel(file)
            file_text = " ".join(df.astype(str).values.flatten())
        except Exception as e:
            print(f"  ERROR reading spreadsheet: {e}")
            continue

    if not file_text.strip():
        continue

    # Split into sentences and create chunks
    sentences = [s.strip() for s in file_text.split(".") if s.strip()]
    buffer = ""
    file_chunks = 0
    for sentence in sentences:
        buffer += sentence + ". "
        if len(buffer) > 150: # Slightly larger chunks for better context
            documents.append(buffer.strip())
            sources.append(file.name)
            buffer = ""
            file_chunks += 1

    if buffer:
        documents.append(buffer.strip())
        sources.append(file.name)
        file_chunks += 1
    
    print(f"  Generated {file_chunks} chunks")

if not documents:
    print("WARNING: No documents to index!")
    sys.exit(0)

# Create embeddings in batches to save RAM
BATCH_SIZE = 32
total_chunks = len(documents)
print(f"\nCreating embeddings for {total_chunks} chunks in batches of {BATCH_SIZE}...")

all_embeddings = []
for i in range(0, total_chunks, BATCH_SIZE):
    batch = documents[i:i + BATCH_SIZE]
    batch_embeddings = model.encode(batch)
    all_embeddings.append(batch_embeddings)
    
    # Progress report
    progress = min(100, (i + len(batch)) / total_chunks * 100)
    print(f"  Progress: {progress:.1f}% ({i + len(batch)}/{total_chunks})")

embeddings = np.vstack(all_embeddings)

# Store in FAISS
print("\nSaving index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

faiss.write_index(index, str(VECTOR_DIR / "index.faiss"))

with open(VECTOR_DIR / "sources.txt", "w") as f:
    for src in sources:
        f.write(src + "\n")

with open(VECTOR_DIR / "chunks.txt", "w") as f:
    for chunk in documents:
        # Save chunks as single lines
        clean_chunk = chunk.replace("\n", " ").strip()
        f.write(clean_chunk + "\n")

print(f"Ingestion complete. Total Chunks: {len(documents)}")
