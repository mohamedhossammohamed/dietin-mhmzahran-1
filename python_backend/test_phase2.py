"""
Phase 2 Nutrition Engine — Comprehensive Stress Test Suite
==========================================================
Tests the HybridSearchEngine against the master plan's acceptance criteria
and a battery of edge-case, concurrency, and data integrity scenarios.
"""

import concurrent.futures
import pytest
from nutrition_engine import (
    HybridSearchEngine,
    NutritionRecord,
    MOCK_DATA,
    seed_database,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def engine():
    """One shared engine for all tests — mirrors production singleton."""
    return HybridSearchEngine()


@pytest.fixture(scope="module")
def collection():
    """Direct access to the ChromaDB collection for data-integrity checks."""
    return seed_database()


# ─────────────────────────────────────────────────────────────
# 1. ACCEPTANCE CRITERIA (Master Plan §Phase 2 line 100)
# ─────────────────────────────────────────────────────────────
class TestAcceptanceCriteria:
    """
    Master plan states:
    'Querying "Spicy crispy chicken" with filter "fried"
     must reliably return the "Fried Chicken" record
     and its exact density_g_cm3 without hallucination.'
    """

    def test_spicy_crispy_chicken_returns_fried_chicken(self, engine):
        result = engine.search("Spicy crispy chicken", "fried")
        assert result is not None, "Expected a match, got None"
        assert result.name == "Fried Chicken"

    def test_density_exact_no_hallucination(self, engine):
        result = engine.search("Spicy crispy chicken", "fried")
        assert result.density_g_cm3 == 0.5, (
            f"Density should be exactly 0.5, got {result.density_g_cm3}"
        )

    def test_return_type_is_pydantic_model(self, engine):
        result = engine.search("Spicy crispy chicken", "fried")
        assert isinstance(result, NutritionRecord)


# ─────────────────────────────────────────────────────────────
# 2. PREPARATION DISCRIMINATION
# ─────────────────────────────────────────────────────────────
class TestPrepDiscrimination:
    """The prep_filter must reliably steer search toward the right variant."""

    def test_chicken_raw_returns_raw_chicken(self, engine):
        result = engine.search("chicken", "raw")
        assert result is not None
        assert result.name == "Raw Chicken"
        assert result.density_g_cm3 == 1.05

    def test_chicken_fried_returns_fried_chicken(self, engine):
        result = engine.search("chicken", "fried")
        assert result is not None
        assert result.name == "Fried Chicken"

    def test_egg_raw_vs_cooked(self, engine):
        raw = engine.search("egg", "raw")
        cooked = engine.search("egg", "cooked")
        assert raw is not None and cooked is not None
        assert raw.name == "Raw Egg"
        assert cooked.name == "Scrambled Eggs"

    def test_potato_raw_vs_mashed(self, engine):
        raw = engine.search("potato", "raw")
        mashed = engine.search("potato", "mashed")
        assert raw is not None and mashed is not None
        assert raw.name == "Raw Potato"
        assert mashed.name == "Mashed Potatoes"


# ─────────────────────────────────────────────────────────────
# 3. DATA SEEDING INTEGRITY
# ─────────────────────────────────────────────────────────────
class TestSeedingIntegrity:
    """Verify all 15 items are seeded exactly once."""

    def test_collection_has_15_items(self, collection):
        assert collection.count() == 15

    def test_idempotent_seeding(self, collection):
        """Calling seed_database() again must NOT duplicate entries."""
        second_collection = seed_database()
        assert second_collection.count() == 15


# ─────────────────────────────────────────────────────────────
# 4. EXACT DENSITY INTEGRITY (no drift / hallucination)
# ─────────────────────────────────────────────────────────────
class TestDensityIntegrity:
    """For every mock item, querying by exact name+prep must
    return the correct density from MOCK_DATA."""

    @pytest.mark.parametrize(
        "name,prep,expected_density",
        [(item["name"], item["preparation"], item["density_g_cm3"]) for item in MOCK_DATA],
        ids=[item["name"] for item in MOCK_DATA],
    )
    def test_exact_density_per_item(self, engine, name, prep, expected_density):
        result = engine.search(name, prep)
        assert result is not None, f"No match for '{name}' + '{prep}'"
        assert result.density_g_cm3 == expected_density, (
            f"{name}: expected density {expected_density}, got {result.density_g_cm3}"
        )


# ─────────────────────────────────────────────────────────────
# 5. PYDANTIC MODEL VALIDATION
# ─────────────────────────────────────────────────────────────
class TestPydanticModel:
    def test_all_fields_present(self, engine):
        result = engine.search("rice", "boiled")
        assert result is not None
        assert result.id is not None
        assert result.name is not None
        assert result.preparation is not None
        assert isinstance(result.density_g_cm3, float)
        assert isinstance(result.yield_factor, float)

    def test_model_serializes_to_dict(self, engine):
        result = engine.search("rice", "boiled")
        d = result.model_dump()
        assert set(d.keys()) == {"id", "name", "preparation", "density_g_cm3", "yield_factor"}


# ─────────────────────────────────────────────────────────────
# 6. EDGE CASES
# ─────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_empty_query_returns_none(self, engine):
        assert engine.search("", "fried") is None

    def test_whitespace_only_query_returns_none(self, engine):
        assert engine.search("   ", "fried") is None

    def test_none_prep_filter_returns_result(self, engine):
        """When no prep filter is given, should still return something."""
        result = engine.search("chicken", None)
        assert result is not None

    def test_empty_prep_filter_returns_result(self, engine):
        result = engine.search("banana", "")
        assert result is not None

    def test_gibberish_query_does_not_crash(self, engine):
        # Should return something (closest match) or None — must not raise
        _ = engine.search("xyzzy123foobarbaz", "fried")

    def test_very_long_query_does_not_crash(self, engine):
        long_q = "fried chicken " * 200
        _ = engine.search(long_q, "fried")

    def test_case_insensitive_prep_filter(self, engine):
        """Prep filters like 'FRIED' and 'Fried' should work identically."""
        upper = engine.search("chicken", "FRIED")
        lower = engine.search("chicken", "fried")
        assert upper is not None and lower is not None
        assert upper.name == lower.name


# ─────────────────────────────────────────────────────────────
# 7. PREP PENALTY EFFECTIVENESS
# ─────────────────────────────────────────────────────────────
class TestPrepPenalty:
    """The prep penalty must measurably influence the ranking."""

    def test_fried_filter_prefers_fried_over_raw(self, engine):
        fried = engine.search("chicken", "fried")
        raw = engine.search("chicken", "raw")
        assert fried.name == "Fried Chicken"
        assert raw.name == "Raw Chicken"
        # They MUST be different
        assert fried.name != raw.name


# ─────────────────────────────────────────────────────────────
# 8. IDEMPOTENT SEEDING (extra coverage)
# ─────────────────────────────────────────────────────────────
class TestIdempotency:
    def test_triple_seed_no_duplicates(self):
        seed_database()
        seed_database()
        c = seed_database()
        assert c.count() == 15


# ─────────────────────────────────────────────────────────────
# 9. CONCURRENT SEARCH
# ─────────────────────────────────────────────────────────────
class TestConcurrency:
    """50 parallel searches must not crash or return inconsistent results."""

    def test_50_parallel_searches(self, engine):
        def do_search(_):
            return engine.search("Spicy crispy chicken", "fried")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(do_search, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for r in results:
            assert r is not None
            assert r.name == "Fried Chicken"
            assert r.density_g_cm3 == 0.5


# ─────────────────────────────────────────────────────────────
# 10. BM25 SCORING SANITY
# ─────────────────────────────────────────────────────────────
class TestBM25Sanity:
    """BM25 scores should be non-negative and influence ranking."""

    def test_bm25_scores_non_negative(self, engine):
        from rank_bm25 import BM25Okapi

        corpus = [f"{item['name']} {item['preparation']}".lower().split() for item in MOCK_DATA]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores("fried chicken".split())
        assert all(s >= 0 for s in scores), f"Negative BM25 score found: {scores}"

    def test_bm25_gives_highest_score_to_exact_match(self, engine):
        from rank_bm25 import BM25Okapi

        corpus = [f"{item['name']} {item['preparation']}".lower().split() for item in MOCK_DATA]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores("fried chicken".split())
        # "Fried Chicken fried" is at index 0 in MOCK_DATA
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        assert MOCK_DATA[best_idx]["name"] == "Fried Chicken"
