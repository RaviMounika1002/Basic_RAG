
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer


DATA_PATH = Path(__file__).parent / "data" / "ai_notes.txt"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

TOP_K = 3


# STEP 1: Load documents
def load_document():
    return DATA_PATH.read_text(encoding="utf-8")


# STEP 2: Split text into chunks
def chunk_text(text, chunk_size=400, overlap=80):
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


# STEP 3: Generate embeddings
def create_embeddings(chunks, model):
    return model.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


# STEP 4: Retrieve relevant chunks
def retrieve(query, chunks, embeddings, model):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    # Dot product of normalized vectors = cosine similarity
    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:TOP_K]

    return [
        {
            "text": chunks[int(i)],
            "score": float(scores[i]),
        }
        for i in top_indices
    ]


# STEP 5: Generate an answer using retrieved context
def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(
        f"[Source {i}]\n{item['text']}"
        for i, item in enumerate(retrieved_chunks, start=1)
    )

    prompt = f"""
You are an AI learning assistant.

Answer the question using ONLY the provided context.
If the context does not contain the answer, say:
"I don't have enough information in my notes."

Do not invent facts.
Cite source numbers when answering.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()

    return response.json()["response"]


def main():
    print("Loading knowledge base...")

    text = load_document()
    chunks = chunk_text(text)

    print(f"Created {len(chunks)} chunks")

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = create_embeddings(chunks, model)

    print("\nAskMyNotes is ready!")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Your question: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        results = retrieve(query, chunks, embeddings, model)

        print("\nRetrieved context:")
        for i, item in enumerate(results, start=1):
            print(f"[{i}] Similarity: {item['score']:.3f}")
            print(item["text"][:150], "\n")

        try:
            answer = generate_answer(query, results)
            print("\nAI Answer:\n", answer, "\n")
        except requests.RequestException as error:
            print(f"\nOllama request failed: {error}")
            print("Ensure Ollama is running and the model is installed.")


if __name__ == "__main__":
    main()
