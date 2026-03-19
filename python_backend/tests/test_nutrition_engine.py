from nutrition_engine import engine

def test_hybrid_search():
    # Test retrieving a simple item
    record = engine.search(query="Spicy crispy chicken", prep_filter="fried")
    assert record is not None
    assert record.name == "Fried Chicken"
    assert record.preparation == "fried"
    assert record.density_g_cm3 == 0.85

    # Test retrieving with penalty (e.g., trying to find boiled chicken but it returns raw or fried)
    # If we specifically filter for 'raw', it shouldn't return 'Fried Chicken' even if "crispy chicken" is similar.
    record = engine.search(query="chicken", prep_filter="raw")
    assert record is not None
    assert record.name == "Raw Chicken Breast"
    assert record.preparation == "raw"
    assert record.density_g_cm3 == 1.05

    # Test retrieval for pizza
    record = engine.search(query="Pizza Slice", prep_filter="baked")
    assert record is not None
    assert record.name == "Pizza"
    assert record.calories_per_100g == 266

    # Test boolean flag logic
    record = engine.search(query="A glass of cold whole milk", prep_filter="liquid")
    assert record is not None
    assert record.name == "Whole Milk"
    assert record.is_liquid == True
