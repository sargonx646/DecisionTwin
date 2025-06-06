import json
from typing import Dict, List
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from collections import Counter

# Download NLTK data at module initialization
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Warning: Failed to download NLTK data: {e}")

def transcript_analyzer(input_data: str) -> str:
    """
    Analyze the debate transcript for keywords, sentiment, arguments, and insights.

    Args:
        input_data (str): JSON string containing transcript and dilemma.

    Returns:
        str: JSON string with analysis results.
    """
    try:
        data = json.loads(input_data)
        transcript = data.get("transcript", [])
        dilemma = data.get("dilemma", "")

        # Initialize sentiment analyzer
        sid = SentimentIntensityAnalyzer()
        stop_words = set(stopwords.words('english'))

        # Keyword frequency analysis (replacing topic modeling)
        words = [word.lower() for entry in transcript for word in word_tokenize(entry['message']) if word.lower() not in stop_words and word.isalnum()]
        word_freq = Counter(words)
        top_keywords = [{"label": f"Keyword {i+1}", "keywords": [word], "weight": count / len(words)} for i, (word, count) in enumerate(word_freq.most_common(5))]

        # Sentiment analysis
        sentiment_analysis = []
        for entry in transcript:
            scores = sid.polarity_scores(entry['message'])
            tone = "positive" if scores['compound'] > 0.1 else "negative" if scores['compound'] < -0.1 else "neutral"
            sentiment_analysis.append({
                "agent": entry['agent'],
                "round": entry['round'],
                "tone": tone,
                "score": scores['compound']
            })

        # Argument mining
        key_arguments = []
        conflicts = []
        for i, entry in enumerate(transcript):
            message = entry['message'].lower()
            if any(word in message for word in ["propose", "suggest", "recommend"]):
                key_arguments.append({
                    "agent": entry['agent'],
                    "type": "Proposal",
                    "content": entry['message'][:100] + "..."
                })
            elif any(word in message for word in ["agree", "support"]):
                key_arguments.append({
                    "agent": entry['agent'],
                    "type": "Agreement",
                    "content": entry['message'][:100] + "..."
                })
            elif any(word in message for word in ["disagree", "conflict", "oppose"]):
                key_arguments.append({
                    "agent": entry['agent'],
                    "type": "Disagreement",
                    "content": entry['message'][:100] + "..."
                })
                next_entry = transcript[(i+1)%len(transcript)]
                conflicts.append({
                    "issue": f"Disagreement in {entry['step']}",
                    "stakeholders": [entry['agent'], next_entry['agent']]
                })

        # Insights
        insights = (
            f"The debate focused on {len(top_keywords)} key terms, with {len(conflicts)} conflicts identified. "
            f"Sentiment varied, with {sum(1 for s in sentiment_analysis if s['tone'] == 'positive')} positive, "
            f"{sum(1 for s in sentiment_analysis if s['tone'] == 'negative')} negative, and "
            f"{sum(1 for s in sentiment_analysis if s['tone'] == 'neutral')} neutral statements."
        )

        # Areas for improvement
        improvement_areas = []
        if len(conflicts) > len(transcript) / 4:
            improvement_areas.append("High conflict rate suggests need for pre-negotiation alignment.")
        if len(top_keywords) < 2:
            improvement_areas.append("Limited keyword diversity; encourage broader discussion.")

        # Process suggestions
        process_suggestions = [
            "Conduct pre-negotiation workshops to align stakeholder goals.",
            "Use a neutral facilitator to manage conflicts.",
            "Implement structured decision-making frameworks to reduce ambiguity.",
            "Encourage evidence-based arguments to minimize bias."
        ]

        analysis = {
            "topics": top_keywords,  # Renamed from "keywords" for compatibility with app.py
            "sentiment_analysis": sentiment_analysis,
            "key_arguments": key_arguments,
            "conflicts": conflicts,
            "insights": insights,
            "improvement_areas": improvement_areas,
            "process_suggestions": process_suggestions
        }
        return json.dumps(analysis)
    except Exception as e:
        return json.dumps({"error": f"Analysis failed: {str(e)}"})
