from typing import Dict, List
import openai
import os
from tenacity import retry, stop_after_attempt, wait_fixed

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def generate_personas(extracted: Dict) -> List[Dict]:
    """
    Generate personas for stakeholders based on extracted decision structure.

    Args:
        extracted (Dict): Extracted decision structure with stakeholders, dilemma, and process.

    Returns:
        List[Dict]: List of generated personas.
    """
    try:
        # Ensure OpenAI API key is set
        if not os.getenv("XAI_API_KEY"):
            raise ValueError("XAI_API_KEY environment variable is not set")

        openai.api_key = os.getenv("XAI_API_KEY")

        # Extract stakeholder names from dictionaries
        stakeholders = extracted.get("stakeholders", [])
        stakeholder_names = []
        for stakeholder in stakeholders:
            if isinstance(stakeholder, dict) and "name" in stakeholder:
                stakeholder_names.append(stakeholder["name"])
            elif isinstance(stakeholder, str):
                stakeholder_names.append(stakeholder)
            else:
                raise ValueError(f"Invalid stakeholder format: {stakeholder}")

        if not stakeholder_names:
            raise ValueError("No valid stakeholders found in extracted data")

        # Prepare prompt for persona generation
        dilemma = extracted.get("dilemma", "Unknown dilemma")
        process = extracted.get("process", [])
        prompt = (
            f"Generate detailed personas for the following stakeholders involved in a decision: {', '.join(stakeholder_names)}. "
            f"Decision context: {dilemma}. "
            f"Process: {', '.join(process)}. "
            "Each persona should include: name, role, bio, psychological_traits (list), influences (list), biases (list), "
            "historical_behavior, tone, goals (list), and expected_behavior. "
            "Return the result as a JSON list of dictionaries."
        )

        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Adjust model as needed
            messages=[
                {"role": "system", "content": "You are an expert in creating detailed stakeholder personas."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )

        # Parse response
        personas = json.loads(response.choices[0].message.content)
        if not isinstance(personas, list):
            raise ValueError("Expected a list of personas from API response")

        return personas

    except Exception as e:
        raise Exception(f"Failed to generate personas: {str(e)}")
