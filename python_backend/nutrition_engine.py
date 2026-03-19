import json
import os
from typing import List, Optional
from pydantic import BaseModel, Field
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class NutritionRecord(BaseModel):
    id: str
    name: str
    preparation: str
    density_g_cm3: float
    yield_factor: float
    calories_per_100g: int
    protein_per_100g: float
    carbs_per_100g: float
    fat_per_100g: float
    is_liquid: bool = False

# Mock dataset of 15 complex food items with deterministic density constants
MOCK_FOOD_DATA = [
    {"id": "food_1", "name": "Fried Chicken", "preparation": "fried", "density_g_cm3": 0.85, "yield_factor": 0.75, "calories_per_100g": 246, "protein_per_100g": 13.5, "carbs_per_100g": 10.5, "fat_per_100g": 16.5, "is_liquid": False},
    {"id": "food_2", "name": "Raw Chicken Breast", "preparation": "raw", "density_g_cm3": 1.05, "yield_factor": 1.0, "calories_per_100g": 120, "protein_per_100g": 22.5, "carbs_per_100g": 0.0, "fat_per_100g": 2.6, "is_liquid": False},
    {"id": "food_3", "name": "White Rice", "preparation": "boiled", "density_g_cm3": 0.72, "yield_factor": 2.5, "calories_per_100g": 130, "protein_per_100g": 2.7, "carbs_per_100g": 28.0, "fat_per_100g": 0.3, "is_liquid": False},
    {"id": "food_4", "name": "Apple", "preparation": "raw", "density_g_cm3": 0.82, "yield_factor": 0.9, "calories_per_100g": 52, "protein_per_100g": 0.3, "carbs_per_100g": 14.0, "fat_per_100g": 0.2, "is_liquid": False},
    {"id": "food_5", "name": "Banana", "preparation": "raw", "density_g_cm3": 0.95, "yield_factor": 0.65, "calories_per_100g": 89, "protein_per_100g": 1.1, "carbs_per_100g": 23.0, "fat_per_100g": 0.3, "is_liquid": False},
    {"id": "food_6", "name": "Orange Juice", "preparation": "liquid", "density_g_cm3": 1.04, "yield_factor": 1.0, "calories_per_100g": 45, "protein_per_100g": 0.7, "carbs_per_100g": 10.0, "fat_per_100g": 0.2, "is_liquid": True},
    {"id": "food_7", "name": "Whole Milk", "preparation": "liquid", "density_g_cm3": 1.03, "yield_factor": 1.0, "calories_per_100g": 61, "protein_per_100g": 3.2, "carbs_per_100g": 4.8, "fat_per_100g": 3.3, "is_liquid": True},
    {"id": "food_8", "name": "Scrambled Eggs", "preparation": "cooked", "density_g_cm3": 0.96, "yield_factor": 0.9, "calories_per_100g": 148, "protein_per_100g": 10.0, "carbs_per_100g": 1.0, "fat_per_100g": 11.0, "is_liquid": False},
    {"id": "food_9", "name": "Boiled Potatoes", "preparation": "boiled", "density_g_cm3": 1.08, "yield_factor": 1.0, "calories_per_100g": 87, "protein_per_100g": 1.9, "carbs_per_100g": 20.0, "fat_per_100g": 0.1, "is_liquid": False},
    {"id": "food_10", "name": "French Fries", "preparation": "fried", "density_g_cm3": 0.45, "yield_factor": 0.8, "calories_per_100g": 312, "protein_per_100g": 3.4, "carbs_per_100g": 41.0, "fat_per_100g": 15.0, "is_liquid": False},
    {"id": "food_11", "name": "Steak", "preparation": "grilled", "density_g_cm3": 1.05, "yield_factor": 0.75, "calories_per_100g": 271, "protein_per_100g": 25.0, "carbs_per_100g": 0.0, "fat_per_100g": 19.0, "is_liquid": False},
    {"id": "food_12", "name": "Broccoli", "preparation": "steamed", "density_g_cm3": 0.37, "yield_factor": 0.95, "calories_per_100g": 35, "protein_per_100g": 2.4, "carbs_per_100g": 7.0, "fat_per_100g": 0.4, "is_liquid": False},
    {"id": "food_13", "name": "Salmon", "preparation": "baked", "density_g_cm3": 1.01, "yield_factor": 0.8, "calories_per_100g": 206, "protein_per_100g": 22.0, "carbs_per_100g": 0.0, "fat_per_100g": 13.0, "is_liquid": False},
    {"id": "food_14", "name": "Almonds", "preparation": "raw", "density_g_cm3": 0.60, "yield_factor": 1.0, "calories_per_100g": 579, "protein_per_100g": 21.0, "carbs_per_100g": 22.0, "fat_per_100g": 50.0, "is_liquid": False},
    {"id": "food_15", "name": "Pizza", "preparation": "baked", "density_g_cm3": 0.75, "yield_factor": 1.0, "calories_per_100g": 266, "protein_per_100g": 11.0, "carbs_per_100g": 33.0, "fat_per_100g": 10.0, "is_liquid": False},
]

class HybridSearchEngine:
    def __init__(self, db_path: str = "./chroma_db"):
        self.db_path = db_path
        self._initialized = False

    def _initialize(self):
        if self._initialized:
            return
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name="nutrition_data")
        # Load embedding model on CPU to ensure it works reliably across environments for DB seed/search
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.records = MOCK_FOOD_DATA
        self.corpus = [f"{r['name']} {r['preparation']}".lower() for r in self.records]
        # Precompute O(1) reverse-lookup map for BM25 scoring
        self.corpus_index_map = {text: idx for idx, text in enumerate(self.corpus)}
        tokenized_corpus = [doc.split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Check if we need to seed
        if self.collection.count() == 0:
            self.seed_database()
        self._initialized = True

    def seed_database(self):
        print("Seeding database with mock nutrition data...")
        ids = [r["id"] for r in self.records]
        documents = [f"{r['name']} {r['preparation']}" for r in self.records]
        metadatas = [r for r in self.records]

        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        print(f"Successfully seeded {len(self.records)} records.")

    def search(self, query: str, prep_filter: Optional[str] = None) -> NutritionRecord:
        """
        Perform a hybrid search using dense vectors (Chroma) and sparse (BM25) penalty.
        """
        self._initialize()
        # 1. Dense Vector Search (Top 10 candidates)
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=10
        )

        if not results['metadatas'] or not results['metadatas'][0]:
            return None

        candidates = results['metadatas'][0]

        # 2. Sparse BM25 Keyword Penalty Calculation
        best_candidate = None
        best_score = -float('inf')

        # If prep_filter exists, we calculate BM25 scores for the corpus to see how well it matches
        bm25_scores = None
        if prep_filter:
            tokenized_query = prep_filter.lower().split(" ")
            bm25_scores = self.bm25.get_scores(tokenized_query)

        for idx, candidate in enumerate(candidates):
            # Dense similarity score (approximate, since chromadb returns distance, lower distance is better)
            # We convert distance to a similarity score for combining
            distance = results['distances'][0][idx]
            dense_score = 1.0 / (1.0 + distance)

            # Sparse Penalty: if prep_filter is provided, use BM25 score to penalize/reward
            sparse_modifier = 0.0
            if prep_filter and bm25_scores is not None:
                # Find the index of this candidate in the original corpus
                # (assuming ids map exactly or we search by name/prep)
                # For simplicity, we search the corpus list to find the match index
                cand_str = f"{candidate['name']} {candidate['preparation']}".lower()
                corpus_idx = self.corpus_index_map.get(cand_str)
                if corpus_idx is not None:
                    bm25_score = bm25_scores[corpus_idx]

                    # If BM25 score is very low, penalize heavily.
                    # If it's high, it mitigates the penalty.
                    if bm25_score < 0.1:
                        sparse_modifier = -0.5  # Heavy penalty
                    else:
                        sparse_modifier = 0.1 * bm25_score # Small reward
                else:
                    sparse_modifier = -0.5 # Penalty if not found somehow

            final_score = dense_score + sparse_modifier

            if final_score > best_score:
                best_score = final_score
                best_candidate = candidate

        if best_candidate:
            return NutritionRecord(**best_candidate)

        return None

# Singleton instance
engine = HybridSearchEngine()
