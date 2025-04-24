import pytest
from utils.visualizer import generate_visuals
import os

def test_generate_visuals():
    keywords = ["budget", "growth", "resources"]
    transcript = [{"agent": "CEO", "message": "Invest in growth."}]
    generate_visuals(keywords, transcript)
    assert os.path.exists("visualization.png")
    assert os.path.exists("heatmap.png")
    os.remove("visualization.png")
    os.remove("heatmap.png")
