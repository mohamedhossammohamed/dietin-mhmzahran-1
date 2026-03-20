import json
import os
import time
from typing import List, Optional
from pydantic import BaseModel, Field
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from logger import logger

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
        
        start_time = time.time()
        logger.info(f"Nutrition Engine: Initializing Hybrid Search Engine at {self.db_path}")
        
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_or_create_collection(name="nutrition_data")
            
            logger.info("Nutrition Engine: Loading SentenceTransformer model (all-MiniLM-L6-v2) on CPU...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            
            self.records = MOCK_FOOD_DATA
            self.corpus = [f"{r['name']} {r['preparation']}".lower() for r in self.records]
            self.corpus_index_map = {text: idx for idx, text in enumerate(self.corpus)}
            tokenized_corpus = [doc.split(" ") for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)

            if self.collection.count() == 0:
                self.seed_database()
            
            self._initialized = True
            duration = (time.time() - start_time) * 1000
            logger.info(f"Nutrition Engine: Initialized successfully in {duration:.2f}ms")
        except Exception as e:
            logger.error(f"Nutrition Engine INIT FAILURE: {str(e)}")
            raise

    def seed_database(self):
        logger.info(f"Nutrition Engine: Seeding database with {len(self.records)} mock records...")
        try:
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
            logger.info(f"Nutrition Engine: Successfully seeded {len(self.records)} records.")
        except Exception as e:
            logger.error(f"Nutrition Engine SEED FAILURE: {str(e)}")
            raise

    def search(self, query: str, prep_filter: Optional[str] = None) -> NutritionRecord:
        """
        Perform a hybrid search using dense vectors (Chroma) and sparse (BM25) penalty.
        """
        start_time = time.time()
        logger.debug(f"Nutrition Engine: Starting search for '{query}' (Prep Filter: {prep_filter})")
        
        try:
            self._initialize()
        except Exception as e:
            logger.error(f"Nutrition Engine FAILURE: Search aborted due to initialization failure: {str(e)}")
            return None

        try:
            # 1. Dense Vector Search (Top 10 candidates)
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=10
            )

            if not results['metadatas'] or not results['metadatas'][0]:
                logger.warning(f"Nutrition Engine: No results found for query '{query}'")
                return None

            candidates = results['metadatas'][0]
            logger.debug(f"Nutrition Engine: Found {len(candidates)} dense candidates")

            # 2. Sparse BM25 Keyword Penalty Calculation
            best_candidate = None
            best_score = -float('inf')

            bm25_scores = None
            if prep_filter:
                tokenized_query = prep_filter.lower().split(" ")
                bm25_scores = self.bm25.get_scores(tokenized_query)

            for idx, candidate in enumerate(candidates):
                distance = results['distances'][0][idx]
                dense_score = 1.0 / (1.0 + distance)

                sparse_modifier = 0.0
                if prep_filter and bm25_scores is not None:
                    cand_str = f"{candidate['name']} {candidate['preparation']}".lower()
                    corpus_idx = self.corpus_index_map.get(cand_str)
                    if corpus_idx is not None:
                        bm25_score = bm25_scores[corpus_idx]
                        if bm25_score < 0.1:
                            sparse_modifier = -0.5
                        else:
                            sparse_modifier = 0.1 * bm25_score
                    else:
                        sparse_modifier = -0.5

                final_score = dense_score + sparse_modifier

                if final_score > best_score:
                    best_score = final_score
                    best_candidate = candidate

            if best_candidate:
                duration = (time.time() - start_time) * 1000
                logger.info(f"Nutrition Engine: Match found for '{query}': {best_candidate['name']} (Score: {best_score:.4f}, Duration: {duration:.2f}ms)")
                return NutritionRecord(**best_candidate)

            logger.warning(f"Nutrition Engine: No suitable candidate found for query '{query}' after hybrid scoring")
            return None

        except Exception as e:
            logger.error(f"Nutrition Engine FAILURE: Search process failed: {str(e)}")
            return None

# Singleton instance
engine = HybridSearchEngine()

