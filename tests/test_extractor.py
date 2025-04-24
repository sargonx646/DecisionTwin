import pytest
from agents.extractor import extract_info
from unittest.mock import patch

def test_extract_info_success():
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": '{"stakeholders": ["CEO", "CFO", "HR"], "issues": [], "process": []}'}}]
        }
        result = extract_info("Test dilemma", "Test process")
        assert result["stakeholders"] == ["CEO", "CFO", "HR"]

def test_extract_info_invalid_stakeholders():
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": '{"stakeholders": ["CEO"], "issues": [], "process": []}'}}]
        }
        with pytest.raises(ValueError, match="Invalid response"):
            extract_info("Test dilemma", "Test process")
