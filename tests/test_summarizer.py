import pytest
from agents.summarizer import summarize_and_analyze
from unittest.mock import patch

def test_summarize_and_analyze_success():
    transcript = [{"agent": "CEO", "message": "Invest in growth."}]
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": '{"summary": "Debate on growth.", "keywords": ["growth"], "suggestion": "Increase input."}}'}]
        }
        summary, keywords, suggestion = summarize_and_analyze(transcript)
        assert summary == "Debate on growth."
        assert keywords == ["growth"]
        assert suggestion == "Increase input."
