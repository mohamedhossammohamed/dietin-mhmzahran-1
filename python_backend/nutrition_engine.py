from typing import Optional

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
import re

class NutritionRecord(BaseModel):
    id: str
    name: str
    preparation: str
    density_g_cm3: float
    yield_factor: float

MOCK_DATA = [
    {"id": "1", "name": "Fried Chicken", "preparation": "fried", "density_g_cm3": 0.5, "yield_factor": 1.0},
    {"id": "2", "name": "Raw Chicken", "preparation": "raw", "density_g_cm3": 1.05, "yield_factor": 0.75},
    {"id": "3", "name": "White Rice", "preparation": "boiled", "density_g_cm3": 0.85, "yield_factor": 2.5},
    {"id": "4", "name": "Brown Rice", "preparation": "boiled", "density_g_cm3": 0.8, "yield_factor": 2.5},
    {"id": "5", "name": "Scrambled Eggs", "preparation": "cooked", "density_g_cm3": 0.6, "yield_factor": 1.0},
    {"id": "6", "name": "Raw Egg", "preparation": "raw", "density_g_cm3": 1.03, "yield_factor": 1.0},
    {"id": "7", "name": "Mashed Potatoes", "preparation": "mashed", "density_g_cm3": 1.0, "yield_factor": 1.0},
    {"id": "8", "name": "Raw Potato", "preparation": "raw", "density_g_cm3": 1.08, "yield_factor": 0.8},
    {"id": "9", "name": "Apple", "preparation": "raw", "density_g_cm3": 0.82, "yield_factor": 1.0},
    {"id": "10", "name": "Banana", "preparation": "raw", "density_g_cm3": 0.95, "yield_factor": 1.0},
    {"id": "11", "name": "French Fries", "preparation": "fried", "density_g_cm3": 0.45, "yield_factor": 1.0},
    {"id": "12", "name": "Grilled Salmon", "preparation": "grilled", "density_g_cm3": 0.98, "yield_factor": 0.8},
    {"id": "13", "name": "Oatmeal", "preparation": "boiled", "density_g_cm3": 1.0, "yield_factor": 6.0},
    {"id": "14", "name": "Broccoli", "preparation": "steamed", "density_g_cm3": 0.45, "yield_factor": 1.0},
    {"id": "15", "name": "Ground Beef", "preparation": "cooked", "density_g_cm3": 0.85, "yield_factor": 0.75}
]

# ── Module-level singleton to prevent multiple PersistentClient instances ──
_chroma_client: Optional[chromadb.PersistentClient] = None
_sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def seed_database():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path="./chroma_db")

    collection = _chroma_client.get_or_create_collection(
        name="nutrition",
        embedding_function=_sentence_transformer_ef,
    )

    if collection.count() == 0:
        ids = [str(item["id"]) for item in MOCK_DATA]
        documents = [f"{item['name']} {item['preparation']}" for item in MOCK_DATA]
        metadatas = [item for item in MOCK_DATA]
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
    return collection

class HybridSearchEngine:
    def __init__(self):
        self.collection = seed_database()
        
    def search(self, query: str, prep_filter: str = "") -> Optional[NutritionRecord]:
        # ── Guard: empty / None inputs ──
        if not query or not query.strip():
            return None

        query = query.strip()
        prep_filter = (prep_filter or "").strip().lower()

        # 1. Dense Search
        results = self.collection.query(
            query_texts=[query],
            n_results=10,
            include=["metadatas", "documents", "distances"],
        )

        if not results["metadatas"] or not results["metadatas"][0]:
            return None

        candidates = results["metadatas"][0]
        documents = results["documents"][0]
        distances = results["distances"][0]

        # 2. Sparse (BM25) initialization
        tokenized_corpus = [doc.lower().split(" ") for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = (query + " " + prep_filter).lower().split(" ")
        bm25_scores = bm25.get_scores(query_tokens)

        best_candidate = None
        best_score = float("-inf")

        for i, candidate in enumerate(candidates):
            # Dense similarity  (inverse distance → higher is better)
            dense_score = 1.0 / (1.0 + distances[i])
            # Sparse similarity
            sparse_score = bm25_scores[i]

            # Penalize candidate whose preparation doesn't match the filter
            penalty = 1.0
            if prep_filter and candidate.get("preparation", "").lower() != prep_filter:
                penalty = 0.2

            final_score = (dense_score + 0.1 * sparse_score) * penalty

            if final_score > best_score:
                best_score = final_score
                best_candidate = candidate

        if best_candidate:
            # ChromaDB may return numeric metadata as strings — coerce
            return NutritionRecord(
                id=str(best_candidate["id"]),
                name=str(best_candidate["name"]),
                preparation=str(best_candidate["preparation"]),
                density_g_cm3=float(best_candidate["density_g_cm3"]),
                yield_factor=float(best_candidate["yield_factor"]),
            )
        return None

if __name__ == "__main__":
    engine = HybridSearchEngine()
    result = engine.search("Spicy crispy chicken", "fried")
    if result:
        print(f"Match found: {result.name} (Density: {result.density_g_cm3})")
    else:
        print("No match found.")
