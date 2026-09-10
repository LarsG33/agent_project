import numpy as np
from sentence_transformers import SentenceTransformer


embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def split_text(text, chunk_size=100, overlap=20):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

    return chunks


with open("knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()


chunks = split_text(text)

chunk_embeddings = embedding_model.encode(chunks)


def cosine_similarity(a, b):
    return (a @ b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def retrieve(question, top_k=2, threshold=0.25):
    question_embedding = embedding_model.encode(question)

    scores = []

    for i in range(len(chunks)):
        score = cosine_similarity(
            question_embedding,
            chunk_embeddings[i]
        )

        scores.append((score, i))

    scores.sort(reverse=True)

    top_chunks = []

    for score, i in scores[:top_k]:

        if score >= threshold:
            top_chunks.append(chunks[i])

    return top_chunks