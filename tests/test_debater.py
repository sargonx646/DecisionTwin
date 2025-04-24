import pytest
from agents.debater import simulate_debate
from unittest.mock import patch

def test_simulate_debate_success():
    personas = [
        {"name": "CEO", "goals": ["Lead"], "biases": ["Pro-growth"], "tone": "Strategic"},
        {"name": "CFO", "goals": ["Save"], "biases": ["Cost-conscious"], "tone": "Analytical"},
        {"name": "HR", "goals": ["Support"], "biases": ["Compliance-focused"], "tone": "Emotional"}
    ]
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "I propose investing in growth."}}]
        }
        transcript = simulate_debate(personas, rounds=1)
        assert len(transcript) == 3
        assert all("agent" in t and "message" in t for t in transcript)
