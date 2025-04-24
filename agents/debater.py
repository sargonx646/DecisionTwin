import json
import os
import time
import random
import numpy as np
from openai import OpenAI, APITimeoutError
from typing import List, Dict
from config import DEBATE_ROUNDS, MAX_TOKENS, TIMEOUT_S
from tenacity import retry, stop_after_attempt, wait_fixed

def simulate_debate(personas: List[Dict], dilemma: str, process_hint: str, extracted: Dict, scenarios: str = "", rounds: int = DEBATE_ROUNDS, max_simulation_time: int = 180, simulation_type: str = "Grok 3 Beta Simulation") -> List[Dict]:
    """
    Simulate a debate among stakeholder personas using the specified simulation method.

    Args:
        personas (List[Dict]): List of personas with name, goals, biases, tone, bio, and expected behavior.
        dilemma (str): The user-provided decision dilemma.
        process_hint (str): The user-provided process and stakeholder details.
        extracted (Dict): Extracted decision structure with process steps and stakeholder roles.
        scenarios (str): Optional alternative scenarios or external factors.
        rounds (int): Number of debate rounds.
        max_simulation_time (int): Maximum allowed time for the entire simulation in seconds.
        simulation_type (str): Type of simulation ("Grok 3 Beta Simulation", "Monte Carlo Simulation", "Game Theory Simulation").

    Returns:
        List[Dict]: Debate transcript with agent, round, step, and message.
    """
    transcript = []
    process_steps = extracted.get("process", [])
    if len(process_steps) < rounds:
        process_steps.extend([process_steps[-1]] * (rounds - len(process_steps)))
    process_steps = process_steps[:rounds]

    stakeholder_roles = {}
    role_focus = {}
    if isinstance(process_hint, str):
        for line in process_hint.split("\n"):
            if ":" in line and any(s["name"] in line for s in extracted.get("stakeholders", [])):
                try:
                    name, role = line.split(":", 1)
                    name = name.strip().split(".")[-1].strip()
                    role = role.strip()
                    if "USAID" not in role:
                        stakeholder_roles[name] = role
                        role_focus[role] = f"Focus on priorities relevant to {role.lower()}."
                except Exception as e:
                    print(f"Error parsing process_hint line: {str(e)}")
    else:
        print(f"Warning: process_hint is not a string, got type {type(process_hint)}. Using default roles.")
        for s in extracted.get("stakeholders", []):
            name = s.get("name", "Unknown")
            role = s.get("role", "Team Member")
            stakeholder_roles[name] = role
            role_focus[role] = f"Focus on priorities relevant to {role.lower()}."

    filtered_personas = [persona for persona in personas if "USAID" not in stakeholder_roles.get(persona["name"], "")]

    process_objectives = {
        "Situation Assessment": "Analyze the dilemma and identify key challenges.",
        "Options Development": "Propose 2–3 actionable options with pros and cons.",
        "Interagency Coordination": "Refine options into a cohesive plan with compromises.",
        "Task Force Deliberation": "Focus on implementation details and mitigation strategies.",
        "Recommendation and Approval": "Finalize the recommendation and propose next steps."
    }

    cumulative_context = f"Dilemma: {dilemma}\nProcess: {process_hint if isinstance(process_hint, str) else json.dumps(process_hint)}\n"
    if scenarios:
        cumulative_context += f"Scenarios: {scenarios}\n"

    start_time = time.time()

    if simulation_type == "Grok 3 Beta Simulation":
        client = OpenAI(base_url="https://api.x.ai/v1", api_key=os.getenv("XAI_API_KEY"))

        @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
        def make_api_call(prompt):
            return client.chat.completions.create(
                model="grok-3-beta",
                messages=[
                    {"role": "system", "content": "You are simulating a stakeholder in a debate."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=600,
                timeout=30
            )

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
                focus_area = role_focus.get(role, f"Focus on priorities relevant to {role.lower()}.")

                prompt = (
                    f"You are {stakeholder_name}, role: {role}. Expertise: {focus_area}\n"
                    f"Goals: {', '.join(persona['goals'])}\nBiases: {', '.join(persona['biases'])}\nTone: {persona['tone']}\n"
                    f"Step: {current_step} (Round {round_num + 1})\nObjective: {objective}\n"
                    f"Context (last 500 chars): {cumulative_context[-500:]}\n"
                    "Provide a 150–200 word response in JSON format with keys 'agent', 'round', 'step', 'message'."
                )

                try:
                    completion = make_api_call(prompt)
                    response = json.loads(completion.choices[0].message.content)
                    if all(key in response for key in ["agent", "round", "step", "message"]):
                        round_transcript.append(response)
                    else:
                        raise ValueError("Invalid JSON structure")
                except APITimeoutError:
                    round_transcript.append({
                        "agent": stakeholder_name,
                        "round": round_num + 1,
                        "step": current_step,
                        "message": f"As {stakeholder_name}, I focus on {focus_area.lower()}. Response timed out."
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

    elif simulation_type == "Monte Carlo Simulation":
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

            round_transcript = []
            for persona in filtered_personas:
                stakeholder_name = persona["name"]
                role = stakeholder_roles.get(stakeholder_name, "Team Member")
                focus_area = role_focus.get(role, f"Focus on priorities relevant to {role.lower()}.")

                # Assign probabilities based on biases and goals
                agree_prob = 0.5
                if "confirmation bias" in persona["biases"]:
                    agree_prob += 0.2
                if "status quo bias" in persona["biases"]:
                    agree_prob -= 0.1
                decision = np.random.choice(
                    ["agree", "disagree", "compromise"],
                    p=[agree_prob, 0.3 - agree_prob / 2, 0.7 - agree_prob / 2]
                )

                message = (
                    f"As {stakeholder_name} ({role}), I {decision} on the proposed approach for {current_step}. "
                    f"My focus is {focus_area.lower()}. {objective} "
                    f"Given my goals ({', '.join(persona['goals'])}), I believe this {decision} aligns with our priorities."
                )

                round_transcript.append({
                    "agent": stakeholder_name,
                    "round": round_num + 1,
                    "step": current_step,
                    "message": message
                })

            transcript.extend(round_transcript)
            cumulative_context += f"\nRound {round_num + 1} ({current_step}):\n"
            for entry in round_transcript:
                cumulative_context += f"- {entry['agent']}: {entry['message'][:100]}...\n"

    elif simulation_type == "Game Theory Simulation":
        # Simple Nash equilibrium simulation
        strategies = ["cooperate", "defect"]
        payoff_matrix = {
            ("cooperate", "cooperate"): (3, 3),
            ("cooperate", "defect"): (0, 5),
            ("defect", "cooperate"): (5, 0),
            ("defect", "defect"): (1, 1)
        }

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

            round_transcript = []
            for i, persona in enumerate(filtered_personas):
                stakeholder_name = persona["name"]
                role = stakeholder_roles.get(stakeholder_name, "Team Member")
                focus_area = role_focus.get(role, f"Focus on priorities relevant to {role.lower()}.")

                # Choose strategy based on biases
                strategy = "cooperate" if "collaborative" in persona["psychological_traits"] else random.choice(strategies)
                opponent = filtered_personas[(i + 1) % len(filtered_personas)] if len(filtered_personas) > 1 else persona
                opponent_strategy = "cooperate" if "collaborative" in opponent["psychological_traits"] else random.choice(strategies)

                payoff = payoff_matrix.get((strategy, opponent_strategy), (1, 1))[0]
                message = (
                    f"As {stakeholder_name} ({role}), I choose to {strategy} in {current_step}. "
                    f"My focus is {focus_area.lower()}. {objective} "
                    f"Given my goals ({', '.join(persona['goals'])}), this strategy yields a payoff of {payoff}."
                )

                round_transcript.append({
                    "agent": stakeholder_name,
                    "round": round_num + 1,
                    "step": current_step,
                    "message": message
                })

            transcript.extend(round_transcript)
            cumulative_context += f"\nRound {round_num + 1} ({current_step}):\n"
            for entry in round_transcript:
                cumulative_context += f"- {entry['agent']}: {entry['message'][:100]}...\n"

    return transcript
