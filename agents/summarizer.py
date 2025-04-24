import json
import os
from openai import OpenAI
from typing import List, Dict, Tuple
from tenacity import retry, stop_after_attempt, wait_fixed

def generate_summary_and_suggestion(transcript: List[Dict]) -> Tuple[str, str]:
    """
    Summarize the debate and provide optimization suggestions.

    Args:
        transcript (List[Dict]): Debate transcript with agent, round, step, and message.

    Returns:
        Tuple[str, str]: Summary and optimization suggestion.
    """
    client = OpenAI(
        base_url="https://api.x.ai/v1",
        api_key=os.getenv("XAI_API_KEY")
    )

    transcript_json = json.dumps(transcript, indent=2)

    prompt = (
        "Analyze a decision-making debate transcript and return in JSON format:\n"
        "1. 'summary': 150–200 word summary of key arguments and outcomes.\n"
        "2. 'faultlines': Major conflicts between stakeholders.\n"
        "3. 'chokepoints': Process bottlenecks or constraints.\n"
        "4. 'suggestion': 150–200 word actionable recommendations.\n"
        f"Transcript:\n{transcript_json[:2000]}...\n"
    )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def make_api_call():
        return client.chat.completions.create(
            model="grok-3-beta",
            messages=[
                {"role": "system", "content": "You are analyzing debate transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

    try:
        completion = make_api_call()
        result = json.loads(completion.choices[0].message.content)
        summary = result.get("summary", "No summary generated.")
        faultlines = result.get("faultlines", "No faultlines identified.")
        chokepoints = result.get("chokepoints", "No chokepoints identified.")
        suggestion = result.get("suggestion", "No suggestions provided.")
        enhanced_suggestion = f"Faultlines: {faultlines}\nChokepoints: {chokepoints}\nRecommendations: {suggestion}"
        return summary, enhanced_suggestion
    except Exception as e:
        print(f"Summarization Error: {str(e)}")
        return (
            "The debate focused on key decision points, but a summary could not be generated.",
            "Consider reviewing stakeholder roles and process steps."
        )
