
import json
import os
from openai import OpenAI
from typing import Dict, List
from config import STAKEHOLDER_ANALYSIS
from tenacity import retry, stop_after_attempt, wait_fixed

def extract_decision_structure(dilemma: str, process_hint: str, scenarios: str = "") -> Dict:
    """
    Extract a decision structure from user inputs using xAI's Grok-3-Beta.

    Args:
        dilemma (str): The decision context provided by the user.
        process_hint (str): Details about the process and/or stakeholders.
        scenarios (str): Optional alternative scenarios or external factors.

    Returns:
        Dict: Extracted decision structure.
    """
    client = OpenAI(
        base_url="https://api.x.ai/v1",
        api_key=os.getenv("XAI_API_KEY")
    )

    prompt = (
        "Extract a decision structure in JSON format with:\n"
        "1. 'decision_type': Strategic, Tactical, Operational, or Other.\n"
        "2. 'stakeholders': At least 4, each with name, role, psychological_traits, influences, biases, historical_behavior, bio.\n"
        "3. 'issues': 2–3 key issues.\n"
        "4. 'process': 3–5 process steps.\n"
        "5. 'external_factors': 1–2 factors.\n"
        f"Inputs:\nDilemma: {dilemma}\nProcess Hint: {process_hint}\nScenarios: {scenarios}\n"
    )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def make_api_call():
        return client.chat.completions.create(
            model="grok-3-beta",
            messages=[
                {"role": "system", "content": "You are extracting decision structures."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

    try:
        completion = make_api_call()
        result = json.loads(completion.choices[0].message.content)

        decision_type = result.get("decision_type", "Strategic (Assumed)")
        stakeholders = result.get("stakeholders", [])
        
        seen_names = set()
        seen_roles = set()
        unique_stakeholders = []
        for i, s in enumerate(stakeholders):
            name = s.get("name", f"Stakeholder {i+1} (Inferred by AI)")
            role = s.get("role", f"Team Member {i+1} (Inferred by AI)")
            
            base_name = name.split(" (Inferred by AI)")[0]
            counter = 1
            new_name = base_name
            while new_name in seen_names:
                new_name = f"{base_name} {counter}"
                counter += 1
            if "(Inferred by AI)" in name:
                name = f"{new_name} (Inferred by AI)"
            else:
                name = new_name
            seen_names.add(name)

            base_role = role.split(" (Inferred by AI)")[0]
            counter = 1
            new_role = base_role
            while new_role in seen_roles:
                new_role = f"{base_role} {counter}"
                counter += 1
            if "(Inferred by AI)" in role:
                role = f"{new_role} (Inferred by AI)"
            else:
                role = new_role
            seen_roles.add(new_role)

            unique_stakeholders.append({
                "name": name,
                "role": role,
                "psychological_traits": s.get("psychological_traits", "Analytical (Inferred by AI)"),
                "influences": s.get("influences", "Public Opinion (Inferred by AI)"),
                "biases": s.get("biases", "Confirmation Bias (Inferred by AI)"),
                "historical_behavior": s.get("historical_behavior", "Consensus-Driven (Inferred by AI)"),
                "bio": s.get("bio", f"{name} has experience as a {role.split(' (Inferred by AI)')[0]}. (Inferred by AI)")
            })

        if len(unique_stakeholders) < 4:
            for i in range(4 - len(unique_stakeholders)):
                name = f"Stakeholder {len(unique_stakeholders)+i+1} (Inferred by AI)"
                role = f"Team Member {len(unique_stakeholders)+i+1} (Inferred by AI)"
                unique_stakeholders.append({
                    "name": name,
                    "role": role,
                    "psychological_traits": "Analytical (Inferred by AI)",
                    "influences": "Public Opinion (Inferred by AI)",
                    "biases": "Confirmation Bias (Inferred by AI)",
                    "historical_behavior": "Consensus-Driven (Inferred by AI)",
                    "bio": f"{name} has experience as a {role.split(' (Inferred by AI)')[0]}. (Inferred by AI)"
                })

        issues = result.get("issues", ["Cost (Assumed)", "Time (Assumed)"])
        process = result.get("process", ["Plan (Assumed)", "Discuss (Assumed)", "Decide (Assumed)"])
        external_factors = result.get("external_factors", ["Resource Availability (Assumed)"])

        ascii_process = generate_ascii_process(process)
        ascii_stakeholders = generate_ascii_stakeholders(unique_stakeholders)

        return {
            "decision_type": decision_type,
            "stakeholders": unique_stakeholders,
            "issues": issues,
            "process": process,
            "external_factors": external_factors,
            "ascii_process": ascii_process,
            "ascii_stakeholders": ascii_stakeholders
        }
    except Exception as e:
        print(f"Extraction Error: {str(e)}")
        return {
            "decision_type": "Strategic (Assumed)",
            "stakeholders": [
                {"name": "Alex Carter (Inferred by AI)", "role": "Manager (Inferred by AI)", "psychological_traits": "Analytical", "influences": "Public Opinion", "biases": "Confirmation Bias", "historical_behavior": "Consensus-Driven", "bio": "Alex Carter has experience as a Manager. (Inferred by AI)"},
                {"name": "Maria Lopez (Inferred by AI)", "role": "Expert (Inferred by AI)", "psychological_traits": "Risk-Averse", "influences": "Government Policies", "biases": "Status Quo Bias", "historical_behavior": "Data-Driven", "bio": "Maria Lopez has experience as an Expert. (Inferred by AI)"},
                {"name": "James Kim (Inferred by AI)", "role": "Team Lead (Inferred by AI)", "psychological_traits": "Collaborative", "influences": "Shareholders", "biases": "Optimism Bias", "historical_behavior": "Long-Term Strategy", "bio": "James Kim has experience as a Team Lead. (Inferred by AI)"},
                {"name": "Sarah Patel (Inferred by AI)", "role": "Analyst (Inferred by AI)", "psychological_traits": "Decisive", "influences": "Industry Trends", "biases": "Groupthink", "historical_behavior": "Unilateral Decision-Maker", "bio": "Sarah Patel has experience as an Analyst. (Inferred by AI)"}
            ],
            "issues": ["Cost (Assumed)", "Time (Assumed)"],
            "process": ["Plan (Assumed)", "Discuss (Assumed)", "Decide (Assumed)"],
            "external_factors": ["Resource Availability (Assumed)"],
            "ascii_process": generate_ascii_process(["Plan (Assumed)", "Discuss (Assumed)", "Decide (Assumed)"]),
            "ascii_stakeholders": generate_ascii_stakeholders([
                {"name": "Alex Carter (Inferred by AI)", "role": "Manager (Inferred by AI)"},
                {"name": "Maria Lopez (Inferred by AI)", "role": "Expert (Inferred by AI)"},
                {"name": "James Kim (Inferred by AI)", "role": "Team Lead (Inferred by AI)"},
                {"name": "Sarah Patel (Inferred by AI)", "role": "Analyst (Inferred by AI)"}
            ])
        }

def generate_ascii_process(process: List[str]) -> str:
    if not process:
        return "No process steps available."
    timeline = "=== Process Timeline ===\n"
    for i, step in enumerate(process, 1):
        timeline += f"{i}. {step}\n"
    timeline += "======================="
    return timeline

def generate_ascii_stakeholders(stakeholders: List[Dict]) -> str:
    if not stakeholders:
        return "No stakeholders available."
    hierarchy = "=== Stakeholders ===\n"
    for s in stakeholders:
        name = s.get("name", "Unknown")
        role = s.get("role", "Unknown")
        hierarchy += f"- {name} ({role})\n"
    hierarchy += "==================="
    return hierarchy
