import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud
from typing import List, Dict
import pandas as pd

def generate_visuals(keywords: List[str], transcript: List[Dict]):
    """Generate visualizations for the debate transcript."""
    # Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(keywords))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("word_cloud.png", format="png")
    plt.close()

    # Stakeholder Interaction Network
    G = nx.Graph()
    for entry in transcript:
        G.add_node(entry["agent"])
    for i, entry1 in enumerate(transcript):
        for entry2 in transcript[i+1:]:
            if entry1["round"] == entry2["round"] and entry1["step"] == entry2["step"]:
                G.add_edge(entry1["agent"], entry2["agent"])
    plt.figure(figsize=(10, 5))
    nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray")
    plt.savefig("network_graph.png", format="png")
    plt.close()

    # Conflict Heatmap
    conflict_counts = {}
    for entry in transcript:
        step = entry["step"]
        if any(word in entry["message"].lower() for word in ["conflict", "disagree", "oppose", "challenge"]):
            conflict_counts[(entry["agent"], step)] = conflict_counts.get((entry["agent"], step), 0) + 1
    agents = list(set(entry["agent"] for entry in transcript))
    steps = list(set(entry["step"] for entry in transcript))
    heatmap_data = pd.DataFrame(0, index=agents, columns=steps)
    for (agent, step), count in conflict_counts.items():
        heatmap_data.loc[agent, step] = count
    plt.figure(figsize=(10, 5))
    sns.heatmap(heatmap_data, annot=True, cmap="Reds")
    plt.title("Negotiation Conflicts by Stakeholder and Step")
    plt.savefig("conflict_heatmap.png", format="png")
    plt.close()
