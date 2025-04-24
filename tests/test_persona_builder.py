import pytest
from agents.persona_builder import build_personas

def test_build_personas_valid():
    stakeholders = ["CEO", "CFO", "HR"]
    personas = build_personas(stakeholders)
    assert len(personas) == 3
    assert all("name" in p and "goals" in p and "biases" in p and "tone" in p for p in personas)

def test_build_personas_invalid_count():
    with pytest.raises(ValueError, match="3–7 stakeholders required"):
        build_personas(["CEO"])
