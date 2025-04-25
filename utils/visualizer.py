from typing import List, Dict
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def generate_visualizations(keywords: List[str], transcript: List[Dict], personas: List[Dict]):
    """
    Generate visualizations for the debate transcript and store in session state.

    Args:
        keywords (List[str]): List of keywords extracted from the transcript.
        transcript (List[Dict]): Debate transcript with agent, round, step, and message.
        personas (List[Dict]): List of personas with name, goals, biases, tone, etc.
    """
    try:
        # Word Cloud
        fig_wordcloud = plt.figure(figsize=(10, 5))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(keywords))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.session_state['wordcloud_fig'] = fig_wordcloud

        # Stakeholder Interaction Network
        G = nx.DiGraph()
        agents = list(set(entry['agent'] for entry in transcript))
        for agent in agents:
            G.add_node(agent)
        for i, entry in enumerate(transcript[:-1]):
            G.add_edge(entry['agent'], transcript[i+1]['agent'])
        pos = nx.spring_layout(G)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        node_x, node_y = [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()), textposition='top center', marker=dict(size=10, color='lightblue'))
        fig_network = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0), xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False)))
        fig_network.update_layout(title="Stakeholder Interaction Network")
        st.session_state['network_fig'] = fig_network

    except Exception as e:
        st.session_state['visualization_error'] = str(e)
