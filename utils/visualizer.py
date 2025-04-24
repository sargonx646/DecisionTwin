import wordcloud
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None
try:
    from gensim import corpora
    from gensim.models import LdaModel
except ImportError:
    corpora = None
    LdaModel = None
from typing import List, Dict
import numpy as np
import pandas as pd
import os

def generate_visuals(keywords: List[str], transcript: List[Dict]):
    """
    Generate visualizations for the debate transcript.

    Args:
        keywords (List[str]): List of extracted keywords.
        transcript (List[Dict]): Debate transcript with agent, round, step, and message.
    """
    # Create placeholder files to prevent FileNotFoundError
    placeholder_html = "<p>Visualization unavailable.</p>"
    for filename in ["network_graph.html", "timeline_chart.html", "sentiment_chart.html", "topic_modeling_chart.html"]:
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                f.write(placeholder_html)

    # 1. Word Cloud
    try:
        wordcloud_obj = wordcloud.WordCloud(width=800, height=400, background_color='white', max_words=15).generate(' '.join(keywords))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_obj, interpolation='bilinear')
        plt.axis('off')
        plt.savefig('word_cloud.png', bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Word Cloud Generation Error: {str(e)}")
        with open('word_cloud.png', 'w') as f:
            f.write('')  # Create empty placeholder

    # 2. Network Graph
    try:
        G = nx.Graph()
        stakeholders = list(set(entry["agent"] for entry in transcript))
        stakeholder_roles = {}
        for entry in transcript:
            agent = entry["agent"]
            if agent not in stakeholder_roles:
                stakeholder_roles[agent] = "Unknown"
            G.add_node(agent, role=stakeholder_roles[agent])

        for entry in transcript:
            speaker = entry["agent"]
            message = entry["message"].lower()
            for other_stakeholder in stakeholders:
                if other_stakeholder != speaker and other_stakeholder.lower() in message:
                    if G.has_edge(speaker, other_stakeholder):
                        G[speaker][other_stakeholder]["weight"] += 1
                    else:
                        G.add_edge(speaker, other_stakeholder, weight=1)

        pos = nx.spring_layout(G)
        edge_x = []
        edge_y = []
        edge_weights = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.extend([edge[2]["weight"] * 2] * 2 + [None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=edge_weights, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_roles = [G.nodes[node]["role"] for node in G.nodes()]
        role_colors = px.colors.qualitative.Plotly[:len(set(node_roles))]
        node_colors = [role_colors[i % len(role_colors)] for i in range(len(node_roles))]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=list(G.nodes()),
            textposition="top center",
            hoverinfo='text',
            marker=dict(size=15, color=node_colors, line=dict(width=2))
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Stakeholder Interaction Network',
                            showlegend=False,
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False))
                        )
        fig.write_html("network_graph.html", include_plotlyjs='cdn')
    except Exception as e:
        print(f"Network Graph Generation Error: {str(e)}")

    # 3. Timeline Chart
    try:
        priorities = []
        for entry in transcript:
            message = entry["message"].lower()
            priority = "Unknown"
            if "humanitarian" in message or "relief" in message:
                priority = "Humanitarian"
            elif "security" in message or "military" in message:
                priority = "Security"
            elif "economic" in message or "infrastructure" in message:
                priority = "Economic"
            priorities.append({
                "Agent": entry["agent"],
                "Round": entry["round"],
                "Step": entry["step"],
                "Priority": priority
            })

        df = pd.DataFrame(priorities)
        fig = px.scatter(
            df,
            x="Round",
            y="Agent",
            color="Priority",
            symbol="Priority",
            title="Stakeholder Priorities Over Rounds",
            labels={"Round": "Debate Round", "Agent": "Stakeholder"},
            hover_data=["Step"]
        )
        fig.update_traces(marker=dict(size=12))
        fig.write_html("timeline_chart.html", include_plotlyjs='cdn')
    except Exception as e:
        print(f"Timeline Chart Generation Error: {str(e)}")

    # 4. Sentiment Analysis
    if TextBlob:
        try:
            sentiments = []
            for entry in transcript:
                blob = TextBlob(entry["message"])
                sentiment = blob.sentiment.polarity
                sentiments.append({
                    "Agent": entry["agent"],
                    "Round": entry["round"],
                    "Sentiment": "Positive" if sentiment > 0.1 else "Negative" if sentiment < -0.1 else "Neutral"
                })

            df = pd.DataFrame(sentiments)
            fig = px.bar(
                df,
                x="Round",
                y="Agent",
                color="Sentiment",
                title="Sentiment Analysis",
                color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"}
            )
            fig.write_html("sentiment_chart.html", include_plotlyjs='cdn')
        except Exception as e:
            print(f"Sentiment Analysis Chart Generation Error: {str(e)}")
    else:
        print("TextBlob not available.")

    # 5. Topic Modeling
    if corpora and LdaModel:
        try:
            texts = [entry["message"].lower().split() for entry in transcript]
            dictionary = corpora.Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]
            lda_model = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)
            topics = lda_model.print_topics(num_words=5)
            topic_data = []
            for topic_id, topic in topics:
                terms = topic.split("+")
                for term in terms:
                    weight, word = term.split("*")
                    topic_data.append({
                        "Topic": f"Topic {topic_id + 1}",
                        "Term": word.strip('"'),
                        "Weight": float(weight)
                    })

            df = pd.DataFrame(topic_data)
            fig = px.bar(
                df,
                x="Weight",
                y="Term",
                color="Topic",
                title="Top Terms in Topics",
                orientation='h'
            )
            fig.write_html("topic_modeling_chart.html", include_plotlyjs='cdn')
        except Exception as e:
            print(f"Topic Modeling Chart Generation Error: {str(e)}")
    else:
        print("Gensim not available.")
