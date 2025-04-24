from pydantic import Field
import json
from typing import List, Dict
from textblob import TextBlob

class TranscriptAnalyzerConfig:
    name = "transcript_analyzer"
    description = "Analyze the debate transcript to identify key themes, conflicts, and negotiation insights."

async def transcript_analyzer(query: str) -> str:
    try:
        data = json.loads(query)
        transcript = data.get("transcript", [])
        dilemma = data.get("dilemma", "")

        # Keyword and sentiment analysis
        keywords = []
        conflicts = []
        sentiments = []
        for entry in transcript:
            message = entry.get("message", "").lower()
            # Detect conflicts
            if any(word in message for word in ["conflict", "disagree", "oppose", "challenge"]):
                conflicts.append(f"{entry['agent']} in Round {entry['round']} (Step: {entry['step']}) showed disagreement.")
            # Extract keywords
            for word in message.split():
                if len(word) > 5 and word not in keywords and word.isalpha():
                    keywords.append(word)
            # Sentiment analysis
            blob = TextBlob(entry["message"])
            sentiments.append({
                "agent": entry["agent"],
                "round": entry["round"],
                "step": entry["step"],
                "sentiment": blob.sentiment.polarity
            })

        # Negotiation insights
        negotiation_issues = []
        if len(conflicts) > 3:
            negotiation_issues.append("High conflict count suggests potential deadlocks or power imbalances.")
        if any(s["sentiment"] < -0.2 for s in sentiments):
            negotiation_issues.append("Negative sentiment detected; possible emotional or adversarial negotiation barriers.")
        sentiment_variance = max(s["sentiment"] for s in sentiments) - min(s["sentiment"] for s in sentiments)
        if sentiment_variance > 0.5:
            negotiation_issues.append("High sentiment variance indicates polarized stakeholder positions.")

        result = {
            "themes": keywords[:10],
            "conflicts": conflicts[:5],
            "sentiments": sentiments,
            "negotiation_issues": negotiation_issues,
            "insights": f"Discussion on '{dilemma}' revealed {len(keywords)} unique themes, {len(conflicts)} conflicts, and {len(negotiation_issues)} negotiation issues."
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Analysis failed: {str(e)}"})
