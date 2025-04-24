from typing import Dict, List
import random
import os
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

def generate_personas(extracted: Dict) -> List[dict]:
    """
    Build detailed personas for stakeholders.

    Args:
        extracted (Dict): Extracted decision structure.

    Returns:
        List[dict]: List of personas with name, goals, biases, tone, bio, and expected behavior.
    """
    client = OpenAI(
        base_url="https://api.x.ai/v1",
        api_key=os.getenv("XAI_API_KEY")
    )

    stakeholders = extracted.get("stakeholders", [])
    if not stakeholders:
        return []

    names = [stakeholder["name"] for stakeholder in stakeholders]
    goals_options = [
        "maximize impact",
        "ensure stability",
        "promote growth",
        "maintain oversight",
        "enhance influence"
    ]
    biases_options = [
        "confirmation bias",
        "optimism bias",
        "groupthink",
        "status quo bias"
    ]
    tones = ["diplomatic", "assertive", "analytical", "cautious"]

    stakeholder_dict = {s["name"]: s for s in stakeholders}
    personas = []

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def make_api_call(prompt):
        return client.chat.completions.create(
            model="grok-3-beta",
            messages=[
                {"role": "system", "content": "You are generating stakeholder profiles."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )

    for name in names:
        extracted_data = stakeholder_dict.get(name, {})
        goals = random.sample(goals_options, k=2)
        biases = random.sample(biases_options, k=2)
        tone = extracted_data.get("tone", random.choice(tones))
        initial_bio = extracted_data.get("bio", f"{name} has experience in their field.")

        context = (
            f"Decision Type: {extracted.get('decision_type', 'Unknown')}\n"
            f"Issues: {', '.join(extracted.get('issues', ['Unknown']))}\n"
            f"Stakeholder: Name: {name}, Role: {extracted_data.get('role', 'Unknown')}"
        )

        prompt = (
            f"Generate a bio (100–150 words) and negotiation behavior (50–100 words) for {name}.\n"
            f"Context:\n{context}\nInitial Bio: {initial_bio}\n"
            "Return as plain text with bio and behavior separated by '\n\n'."
        )

        try:
            completion = make_api_call(prompt)
            response = completion.choices[0].message.content
            bio, behavior = response.split("\n\n", 1) if "\n\n" in response else (response, f"{name} negotiates with a {tone} tone.")
        except Exception as e:
            print(f"Error generating profile for {name}: {str(e)}")
            bio = initial_bio
            behavior = f"{name} negotiates with a {tone} tone."

        personas.append({
            "name": name,
            "goals": goals,
            "biases": biases,
            "tone": tone,
            "bio": bio.strip(),
            "expected_behavior": behavior.strip()
        })

    return personas
