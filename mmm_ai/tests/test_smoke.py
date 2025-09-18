def test_imports():
    import mmm_ai.app.pipeline as p
    assert callable(p.analyze_model)
