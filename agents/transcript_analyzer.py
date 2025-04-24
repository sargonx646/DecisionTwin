import json
from typing import Dict, List
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import re

nltk.download('vader_lexicon')

def transcript_analyzer(input_data: str) -> str:
    """
    Analyze the debate transcript for themes, conflicts, and negotiation insights.

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

        # Extract themes
        words = [word.lower() for entry in transcript for word in entry['message'].split() if len(word) > 5]
        common_words = [word for word, count in Counter(words).most_common(10) if word not in dilemma.lower().split()]
        themes = common_words

        # Identify conflicts and contentions
        conflicts = []
        contentions = []
        for i, entry in enumerate(transcript):
            if any(word in entry['message'].lower() for word in ["disagree", "conflict", "oppose"]):
                conflicts.append(f"{entry['agent']} disagrees in {entry['step']} (Round {entry['round']})")
                next_entry = transcript[(i+1)%len(transcript)]
                contentions.append({
                    "issue": f"Disagreement in {entry['step']}",
                    "stakeholders": [entry['agent'], next_entry['agent']]
                })

        # Negotiation issues
        negotiation_issues = [f"Unresolved conflict in {conflict}" for conflict in conflicts]

        # Insights
        insights = f"Debate focused on {', '.join(themes)}. Conflicts arose in {len(conflicts)} instances, indicating potential misalignment."

        # Main points
        main_points = []
        for entry in transcript:
            if any(word in entry['message'].lower() for word in ["propose", "suggest", "recommend"]):
                main_points.append(f"{entry['agent']} proposed: {entry['message'][:100]}...")

        # Areas for improvement
        improvement_areas = []
        if len(conflicts) > len(transcript) / 4:
            improvement_areas.append("Reduce stakeholder conflicts through pre-negotiation alignment.")
        if len(themes) < 3:
            improvement_areas.append("Encourage broader discussion to cover more aspects of the dilemma.")

        # Problematic stakeholders
        problematic_stakeholders = []
        agent_conflicts = Counter([entry['agent'] for entry in transcript if "disagree" in entry['message'].lower()])
        for agent, count in agent_conflicts.items():
            if count > 1:
                problematic_stakeholders.append({"name": agent, "issue": f"Involved in {count} conflicts"})

        # Tone analysis
        tone_analysis = []
        for entry in transcript:
            scores = sid.polarity_scores(entry['message'])
            tone = "positive" if scores['compound'] > 0.1 else "negative" if scores['compound'] < -0.1 else "neutral"
            tone_analysis.append({"agent": entry['agent'], "tone": tone, "score": scores['compound']})

        # Process suggestions
        process_suggestions = [
            "Implement pre-negotiation workshops to align stakeholder goals.",
            "Use a neutral mediator to facilitate contentious discussions.",
            "Incorporate structured voting to resolve deadlocks.",
            "Encourage data-driven arguments to reduce bias-driven conflicts."
        ]

        analysis = {
            "themes": themes,
            "conflicts": conflicts,
            "negotiation_issues": negotiation_issues,
            "insights": insights,
            "main_points": main_points,
            "improvement_areas": improvement_areas,
            "problematic_stakeholders": problematic_stakeholders,
            "tone_analysis": tone_analysis,
            "contentions": contentions,
            "process_suggestions": process_suggestions
        }
        return json.dumps(analysis)
    except Exception as e:
        return json.dumps({"error": f"Analysis failed: {str(e)}"})
