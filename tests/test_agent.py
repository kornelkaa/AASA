from aasa.agent import AutonomousSceneAgent


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    try:
        AutonomousSceneAgent(api_key=None)
    except ValueError as err:
        assert "GROQ_API_KEY" in str(err)
    else:
        raise AssertionError("expected ValueError when no API key is set")


def test_prompt_lists_detected_objects():
    agent = AutonomousSceneAgent(api_key="test-key")
    fusion_results = [
        {"label": "truck", "confidence": 0.9, "distance_m": 12.3},
        {"label": "person", "confidence": 0.4, "distance_m": None},
    ]

    prompt = agent.build_prompt(fusion_results)

    assert "- TRUCK at 12.3 meters (Detection Confidence: 90%)" in prompt
    assert "- PERSON at Unknown distance (Detection Confidence: 40%)" in prompt
