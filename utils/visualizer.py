import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud
import os
import numpy as np
from typing import List, Dict

def generate_visuals(keywords: List[str], transcript: List[Dict], personas: List[Dict]):
    """
    Generate visualizations for the debate transcript.

    Args:
        keywords (List[str]): List of keywords extracted from the transcript.
        transcript (List[Dict]): Debate transcript with agent, round, step, and message.
        personas (List[Dict]): List of personas with name, goals, biases, tone, etc.
    """
    try:
        # Ensure output directory exists
        os.makedirs("visuals", exist_ok=True)

        # Word Cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(keywords))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig("visuals/word_cloud.png")
        plt.close()

        # Stakeholder Interaction Network
        G = nx.DiGraph()
        agents = list(set(entry['agent'] for entry in transcript))
        for agent in agents:
            G.add_node(agent)
        for i, entry in enumerate(transcript[:-1]):
            G.add_edge(entry['agent'], transcript[i+1]['agent'])
        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', arrows=True)
        plt.title("Stakeholder Interaction Network")
        plt.savefig("visuals/network_graph.png")
        plt.close()

        # Conflict Heatmap
        conflict_matrix = np.random.rand(len(agents), len(agents))  # Placeholder for actual conflict scores
        plt.figure(figsize=(8, 6))
        sns.heatmap(conflict_matrix, xticklabels=agents, yticklabels=agents, cmap='Reds', annot=True)
        plt.title("Negotiation Conflict Heatmap")
        plt.savefig("visuals/conflict_heatmap.png")
        plt.close()

        # Tone Heatmap
        tone_scores = {agent: random.uniform(-1, 1) for agent in agents}  # Placeholder for sentiment analysis
        tone_matrix = np.array([[tone_scores[agent] for _ in agents] for agent in agents])
        plt.figure(figsize=(8, 6))
        sns.heatmap(tone_matrix, xticklabels=agents, yticklabels=agents, cmap='coolwarm', annot=True)
        plt.title("Tone Analysis Heatmap")
        plt.savefig("visuals/tone_heatmap.png")
        plt.close()

        # Contention Network
        C = nx.Graph()
        for agent in agents:
            C.add_node(agent)
        for i, entry in enumerate(transcript):
            if "disagree" in entry['message'].lower() or "conflict" in entry['message'].lower():
                next_agent = transcript[(i+1)%len(transcript)]['agent']
                C.add_edge(entry['agent'], next_agent)
        pos = nx.spring_layout(C)
        plt.figure(figsize=(10, 6))
        nx.draw(C, pos, with_labels=True, node_color='salmon', node_size=2000, font_size=10, font_weight='bold')
        plt.title("Contention Network")
        plt.savefig("visuals/contention_network.png")
        plt.close()

    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
        # Create placeholder images to prevent MediaFileStorageError
        plt.figure(figsize=(1, 1))
        plt.text(0.5, 0.5, "Visualization Unavailable", ha='center', va='center')
        plt.axis('off')
        for fname in ["word_cloud.png", "network_graph.png", "conflict_heatmap.png", "tone_heatmap.png", "contention_network.png"]:
            plt.savefig(f"visuals/{fname}")
        plt.close()
