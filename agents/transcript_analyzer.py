from agentiq import register_function, FunctionBaseConfig, Builder, FunctionInfo
from pydantic import Field
import json
from typing import List, Dict

class TranscriptAnalyzerConfig(FunctionBaseConfig, name="transcript_analyzer"):
    description: str = Field(default="Analyze the debate transcript to identify key themes, conflicts, and insights.")

@register_function(config_type=TranscriptAnalyzerConfig)
async def transcript_analyzer(config: TranscriptAnalyzerConfig, builder: Builder):
    async def _inner(query: str) -> str:
        try:
            data = json.loads(query)
            transcript = data.get("transcript", [])
            dilemma = data.get("dilemma", "")

            # Simple keyword-based analysis
            keywords = []
            conflicts = []
            for entry in transcript:
                message = entry.get("message", "").lower()
                if "conflict" in message or "disagree" in message:
                    conflicts.append(f"{entry['agent']} in Round {entry['round']} disagreed.")
                for word in message.split():
                    if len(word) > 5 and word not in keywords:
                        keywords.append(word)

            result = {
                "themes": keywords[:10],
                "conflicts": conflicts[:5],
                "insights": f"Discussion on '{dilemma}' revealed {len(keywords)} unique themes and {len(conflicts)} conflicts."
            }
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": f"Analysis failed: {str(e)}"})

    yield FunctionInfo.from_fn(_inner, description=config.description)