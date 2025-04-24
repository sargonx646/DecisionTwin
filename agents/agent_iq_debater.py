import json
import os
import time
import logging
from typing import List, Dict
from config import DEBATE_ROUNDS
from tenacity import retry, stop_after_attempt, wait_fixed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from agentiq import AIQRunner
except ImportError:
    AIQRunner = None
    logger.error("Failed to import 'agentiq'. Ensure 'agentiq==1.0.0' is installed from NVIDIA's repository (build.nvidia.com).")

def simulate_debate_agent_iq(personas: List[Dict], dilemma: str, process_hint: str, extracted: Dict, scenarios: str = "", rounds: int = DEBATE_ROUNDS, max_simulation_time: int = 180) -> List[Dict]:
    """
    Simulate a debate among stakeholder personas using NVIDIA AgentIQ.

    Args:
        personas (List[Dict]): List of personas with name, goals, biases, tone, bio, and expected behavior.
        dilemma (str): The user-provided decision dilemma.
        process_hint (str): The user-provided process and stakeholder details.
        extracted (Dict): Extracted decision structure with process steps and stakeholder roles.
        scenarios (str): Optional alternative scenarios or external factors.
        rounds (int): Number of debate rounds.
        max_simulation_time (int): Maximum allowed time in seconds.

    Returns:
        List[Dict]: Debate transcript with agent, round, step, and message.
    """
    if AIQRunner is None:
        error_msg = (
            "AgentIQ simulation failed: 'agentiq' package is not installed. "
            "Please install 'agentiq==1.0.0' from NVIDIA's repository (visit build.nvidia.com for access) "
            "or contact NVIDIA support. Ensure 'rich==13.9.0' is installed to avoid conflicts."
        )
        logger.error(error_msg)
        return [{
            "agent": "System",
            "round": 1,
            "step": "Error",
            "message": error_msg
        }]

    transcript = []
    process_steps = extracted.get("process", [])
    if len(process_steps) < rounds:
        process_steps.extend([process_steps[-1]] * (rounds - len(process_steps)))
    process_steps = process_steps[:rounds]

    # Filter personas to exclude USAID-related stakeholders
    stakeholder_roles = {}
    for line in process_hint.split("\n"):
        if ":" in line and any(s["name"] in line for s in extracted.get("stakeholders", [])):
            name, role = line.split(":", 1)
            name = name.strip().split(".")[-1].strip()
            role = role.strip()
            if "USAID" not in role:
                stakeholder_roles[name] = role
    filtered_personas = [p for p in personas if "USAID" not in stakeholder_roles.get(p["name"], "")]

    # Save personas to a temporary JSON file in /tmp for Streamlit Cloud compatibility
    personas_file = "/tmp/personas.json"
    try:
        with open(personas_file, "w") as f:
            json.dump(filtered_personas, f, indent=2)
    except Exception as e:
        transcript.append({
            "agent": "System",
            "round": 1,
            "step": "Error",
            "message": f"Failed to write personas file: {str(e)}"
        })
        return transcript

    # Initialize AgentIQ runner
    config_file = "agents/agent_iq_config.yml"
    try:
        runner = AIQRunner(config_file=config_file)
    except Exception as e:
        transcript.append({
            "agent": "System",
            "round": 1,
            "step": "Error",
            "message": f"Failed to initialize AgentIQ runner: {str(e)}"
        })
        os.remove(personas_file) if os.path.exists(personas_file) else None
        return transcript

    # Define process objectives
    process_objectives = {
        "Situation Assessment": "Analyze the dilemma and identify key challenges.",
        "Options Development": "Propose 2–3 actionable options with pros and cons.",
        "Interagency Coordination": "Refine options into a cohesive plan with compromises.",
        "Task Force Deliberation": "Focus on implementation details and mitigation strategies.",
        "Recommendation and Approval": "Finalize the recommendation and propose next steps."
    }

    # Initialize cumulative context
    cumulative_context = f"Dilemma: {dilemma}\nProcess: {process_hint}\n"
    if scenarios:
        cumulative_context += f"Scenarios: {scenarios}\n"

    start_time = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def run_workflow(agent_name: str, input_data: str):
        return runner.run(input=input_data)

    # Simulate debate
    for round_num in range(rounds):
        elapsed_time = time.time() - start_time
        if elapsed_time > max_simulation_time:
            transcript.append({
                "agent": "System",
                "round": round_num + 1,
                "step": process_steps[round_num] if round_num < len(process_steps) else "Unknown",
                "message": f"Simulation interrupted: Exceeded maximum time of {max_simulation_time} seconds."
            })
            break

        current_step = process_steps[round_num]
        step_key = current_step.split("(")[0].strip()
        objective = process_objectives.get(step_key, "Continue the discussion.")

        # Process Manager Agent: Orchestrate the round
        manager_input = json.dumps({
            "agent_type": "process_manager",
            "round": round_num + 1,
            "step": current_step,
            "objective": objective,
            "personas_file": personas_file,
            "context": cumulative_context[-500:],
            "dilemma": dilemma
        })
        try:
            manager_result = run_workflow("Process Manager", manager_input)
            manager_response = json.loads(manager_result)
            transcript.append({
                "agent": "Process Manager",
                "round": round_num + 1,
                "step": current_step,
                "message": manager_response.get("message", "Initiated debate round.")
            })
        except Exception as e:
            transcript.append({
                "agent": "Process Manager",
                "round": round_num + 1,
                "step": current_step,
                "message": f"Error initiating round: {str(e)}"
            })

        # Stakeholder Agents: Contribute to the debate
        round_transcript = []
        for persona in filtered_personas:
            elapsed_time = time.time() - start_time
            if elapsed_time > max_simulation_time:
                transcript.append({
                    "agent": "System",
                    "round": round_num + 1,
                    "step": current_step,
                    "message": f"Simulation interrupted: Exceeded maximum time of {max_simulation_time} seconds."
                })
                break

            stakeholder_name = persona["name"]
            role = stakeholder_roles.get(stakeholder_name, "Team Member")
            stakeholder_input = json.dumps({
                "agent_type": "stakeholder",
                "name": stakeholder_name,
                "role": role,
                "goals": persona["goals"],
                "biases": persona["biases"],
                "tone": persona["tone"],
                "bio": persona["bio"],
                "round": round_num + 1,
                "step": current_step,
                "objective": objective,
                "context": cumulative_context[-500:],
                "dilemma": dilemma
            })
            try:
                result = run_workflow(stakeholder_name, stakeholder_input)
                response = json.loads(result)
                round_transcript.append({
                    "agent": stakeholder_name,
                    "round": round_num + 1,
                    "step": current_step,
                    "message": response.get("message", f"{stakeholder_name} contributed to the debate.")
                })
            except Exception as e:
                round_transcript.append({
                    "agent": stakeholder_name,
                    "round": round_num + 1,
                    "step": current_step,
                    "message": f"Error generating response: {str(e)}"
                })

        transcript.extend(round_transcript)
        cumulative_context += f"\nRound {round_num + 1} ({current_step}):\n"
        for entry in round_transcript:
            cumulative_context += f"- {entry['agent']}: {entry['message'][:100]}...\n"

    # Analysis Agent: Analyze the transcript
    analysis_input = json.dumps({
        "agent_type": "analysis",
        "transcript": transcript,
        "dilemma": dilemma,
        "context": cumulative_context[-500:]
    })
    try:
        analysis_result = run_workflow("Analysis Agent", analysis_input)
        analysis_response = json.loads(analysis_result)
        transcript.append({
            "agent": "Analysis Agent",
            "round": rounds + 1,
            "step": "Analysis",
            "message": analysis_response.get("message", "Analysis completed.")
        })
    except Exception as e:
        transcript.append({
            "agent": "Analysis Agent",
            "round": rounds + 1,
            "step": "Analysis",
            "message": f"Error analyzing transcript: {str(e)}"
        })

    # Clean up
    if os.path.exists(personas_file):
        os.remove(personas_file)

    return transcript
