from typing import Dict, List
import openai
import os
import json
import logging
from tenacity import retry, stop_after_attempt, wait_fixed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
def generate_personas(extracted: Dict) -> List[Dict]:
    """
    Generate personas for stakeholders based on extracted decision structure using OpenAI API.

    Args:
        extracted (Dict): Extracted decision structure with stakeholders, dilemma, and process.

    Returns:
        List[Dict]: List of generated personas.
    """
    try:
        # Log input data
        logger.info(f"Input extracted: {json.dumps(extracted, indent=2, default=str)}")

        # Validate input
        if not isinstance(extracted, dict):
            raise ValueError(f"Expected dict for extracted, got {type(extracted)}")

        # Ensure OpenAI API key is set
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable is not set")
        logger.info("API key configured")

        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized")

        # Extract stakeholder names
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
        logger.info(f"Stakeholder names: {stakeholder_names}")

        # Prepare prompt
        dilemma = extracted.get("dilemma", "Unknown dilemma")
        process = extracted.get("process", [])
        prompt = (
            f"Generate detailed personas for the following stakeholders involved in a decision: {', '.join(stakeholder_names)}. "
            f"Decision context: {dilemma}. "
            f"Process: {', '.join(process)}. "
            "Each persona should include: name, role, bio, psychological_traits (list), influences (list), biases (list), "
            "historical_behavior, tone, goals (list), expected_behavior. "
            "Return the result as a JSON list of dictionaries."
        )
        logger.info(f"Prompt: {prompt[:500]}...")

        # Call OpenAI API
        logger.info("Making OpenAI API call")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in creating detailed stakeholder personas."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        logger.info("API call successful")

        # Parse response
        content = response.choices[0].message.content
        logger.info(f"Raw OpenAI response: {content[:500]}...")
        try:
            personas = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            raise ValueError(f"Failed to parse OpenAI response as JSON: {str(e)}")

        if not isinstance(personas, list):
            logger.error(f"Expected list, got {type(personas)}")
            raise ValueError(f"Expected a list of personas, got: {type(personas)}")

        # Validate personas
        required_keys = ["name", "role", "bio", "psychological_traits", "influences", "biases", "historical_behavior", "tone", "goals", "expected_behavior"]
        for persona in personas:
            missing_keys = [key for key in required_keys if key not in persona]
            if missing_keys:
                logger.error(f"Persona missing required keys: {missing_keys}")
                raise ValueError(f"Persona missing required keys: {missing_keys}")
            if not isinstance(persona["psychological_traits"], list):
                raise ValueError(f"psychological_traits must be a list, got {type(persona['psychological_traits'])}")
            if not isinstance(persona["influences"], list):
                raise ValueError(f"influences must be a list, got {type(persona['influences'])}")
            if not isinstance(persona["biases"], list):
                raise ValueError(f"biases must be a list, got {type(persona['biases'])}")
            if not isinstance(persona["goals"], list):
                raise ValueError(f"goals must be a list, got {type(persona['goals'])}")

        logger.info(f"Generated {len(personas)} personas successfully")
        return personas

    except openai.AuthenticationError as e:
        logger.error(f"OpenAI authentication error: {str(e)}")
        raise Exception(f"OpenAI authentication error: {str(e)}")
    except openai.RateLimitError as e:
        logger.error(f"OpenAI rate limit error: {str(e)}")
        raise Exception(f"OpenAI rate limit error: {str(e)}")
    except openai.OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise Exception(f"OpenAI API error: {str(e)}")
    except Exception as e:
        logger.error(f"Persona generation failed: {str(e)}")
        raise Exception(f"Failed to generate personas: {str(e)}")
