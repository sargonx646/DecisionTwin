
import streamlit as st
import json
import os
import random
import PyPDF2
from io import BytesIO
from typing import List, Dict
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import networkx as nx  # Ensure networkx is imported
from agents.extractor import extract_decision_structure
from agents.persona_builder import generate_personas
from agents.debater import simulate_debate
from agents.summarizer import generate_summary_and_suggestion
from agents.transcript_analyzer import transcript_analyzer
from utils.visualizer import generate_visualizations
from utils.db import save_persona, get_all_personas, init_db, update_persona, delete_persona

# Initialize database
init_db()

# Create personas directory
os.makedirs("personas", exist_ok=True)

# Check for API key
if not os.getenv("XAI_API_KEY"):
    st.error("XAI_API_KEY environment variable is not set. Please configure it in .env.")
    st.stop()

# Embedded hardcoded personas
HARDCODED_PERSONAS = [
    {
        "name": "John F. Kennedy",
        "role": "Former U.S. President",
        "bio": "John Fitzgerald Kennedy (1917–1963) was the 35th President of the United States, serving from 1961 until his assassination in 1963. A charismatic leader, he navigated the Cold War, the Cuban Missile Crisis, and the Civil Rights Movement. Known for his eloquent speeches and vision for space exploration, JFK emphasized diplomacy and innovation.",
        "psychological_traits": ["charismatic", "decisive", "optimistic", "pragmatic"],
        "influences": ["public opinion", "international allies", "military advisors", "media"],
        "biases": ["optimism bias", "groupthink", "confirmation bias"],
        "historical_behavior": "Consensus-driven, proactive in crises, long-term strategist",
        "tone": "inspirational",
        "goals": ["promote global peace", "advance technology", "strengthen national unity"],
        "expected_behavior": "JFK negotiates with an inspirational and diplomatic tone, seeking consensus while pushing innovative solutions."
    },
    {
        "name": "Abraham Lincoln",
        "role": "Former U.S. President",
        "bio": "Abraham Lincoln (1809–1865) was the 16th President of the United States, leading the nation through the Civil War (1861–1865). He issued the Emancipation Proclamation, preserving the Union and advancing abolition. A self-educated lawyer, Lincoln was known for his honesty, empathy, and strategic thinking.",
        "psychological_traits": ["empathetic", "analytical", "resilient", "collaborative"],
        "influences": ["abolitionists", "military leaders", "public sentiment", "economic advisors"],
        "biases": ["status quo bias", "anchoring bias"],
        "historical_behavior": "Data-driven, consensus-driven, long-term strategist",
        "tone": "empathetic",
        "goals": ["preserve union", "advance equality", "stabilize economy"],
        "expected_behavior": "Lincoln negotiates with empathy and persuasion, focusing on data-driven solutions and long-term stability."
    },
    {
        "name": "Joe Biden",
        "role": "Former U.S. President",
        "bio": "Joseph Robinette Biden Jr. (born November 20, 1942) served as the 46th U.S. President (2021–2025), defeating Donald Trump in 2020. A career politician, he was Vice President (2009–2017) under Barack Obama and a U.S. Senator from Delaware (1973–2009). Biden’s presidency focused on COVID-19 recovery, infrastructure, climate change, and restoring U.S. alliances, but faced criticism over inflation, Afghanistan withdrawal, and immigration. Known for his empathy and resilience, shaped by personal tragedies, Biden is a moderate Democrat with a pragmatic approach. His verbal gaffes and age-related concerns dominated his 2024 campaign narrative.",
        "psychological_traits": ["empathetic", "resilient", "deliberative", "conciliatory", "prone to overconfidence"],
        "influences": ["Democratic Party establishment", "labor unions", "international allies", "public opinion", "personal advisors"],
        "biases": ["status quo bias", "confirmation bias", "anchoring bias"],
        "historical_behavior": "Consensus-driven, pragmatic, relationship-focused",
        "tone": "empathetic",
        "goals": ["restore democratic norms", "advance social equity", "strengthen alliances", "combat climate change"],
        "expected_behavior": "Biden negotiates with a focus on compromise, leveraging relationships and institutional knowledge, but may struggle with rapid debates."
    },
    {
        "name": "Emmanuel Macron",
        "role": "President of France",
        "bio": "Emmanuel Jean-Michel Frédéric Macron (born December 21, 1977) is a centrist politician leading France since 2017. A former investment banker, he founded La République En Marche! and won the presidency in 2017 and 2022. His policies emphasize labor market flexibility, pension reform, and green energy, but have sparked protests. Globally, he champions multilateralism, climate action, and European sovereignty, often mediating in conflicts. His intellectual style and perceived elitism polarize voters.",
        "psychological_traits": ["intellectual", "ambitious", "adaptive", "perfectionist", "aloof"],
        "influences": ["European leaders", "French technocrats", "global institutions", "public sentiment"],
        "biases": ["elitism bias", "optimism bias", "self-serving bias"],
        "historical_behavior": "Strategic, diplomatic, risk-taking",
        "tone": "articulate",
        "goals": ["strengthen EU autonomy", "drive economic modernization", "lead global climate efforts", "maintain France’s influence"],
        "expected_behavior": "Macron negotiates with intellectual rigor, seeking win-win outcomes but prioritizing French and EU interests."
    },
    {
        "name": "Donald Trump",
        "role": "U.S. President",
        "bio": "Donald John Trump (born June 14, 1946) is the 47th U.S. President (2025–present; 45th, 2017–2021). A businessman and media personality, he reshaped U.S. politics with his populist, America First agenda. His presidencies focus on tax cuts, deregulation, border security, and trade protectionism. His first term included the 2017 Tax Cuts and Jobs Act and Abraham Accords, but was marred by impeachment and COVID-19 criticism. His 2024 campaign leveraged anti-establishment sentiment, securing a second term. Recent actions include declassifying JFK files and imposing tariffs.",
        "psychological_traits": ["assertive", "narcissistic", "opportunistic", "resilient", "polarizing"],
        "influences": ["conservative base", "business allies", "international strongmen", "media"],
        "biases": ["confirmation bias", "zero-sum bias", "recency bias"],
        "historical_behavior": "Disruptive, transactional, media-savvy",
        "tone": "confident",
        "goals": ["restore U.S. economic dominance", "secure borders", "dismantle deep state", "project global strength"],
        "expected_behavior": "Trump negotiates aggressively, using leverage to extract concessions, thriving in high-stakes confrontations."
    }
]

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 0
if "dilemma" not in st.session_state:
    st.session_state.dilemma = ""
if "extracted" not in st.session_state:
    st.session_state.extracted = {}
if "personas" not in st.session_state:
    st.session_state.personas = []
if "transcript" not in st.session_state:
    st.session_state.transcript = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "suggestion" not in st.session_state:
    st.session_state.suggestion = ""
if "analysis" not in st.session_state:
    st.session_state.analysis = {}
if "replace_index" not in st.session_state:
    st.session_state.replace_index = {}

# Sidebar with logo and navigation
st.sidebar.image("https://github.com/sargonx646/DF_22AprilLate/raw/main/assets/decisionforge_logo.png.png", use_column_width=True)
st.sidebar.markdown("<h2 style='text-align: center;'>DecisionTwin</h2>", unsafe_allow_html=True)
progress = st.session_state.step / 5
st.sidebar.progress(progress)
st.sidebar.markdown(f"**Step {st.session_state.step} of 5**")
if st.session_state.step > 0:
    if st.sidebar.button("Back", key="back"):
        st.session_state.step = max(0, st.session_state.step - 1)
        st.rerun()
if st.session_state.step < 5:
    if st.sidebar.button("Forward", key="forward"):
        st.session_state.step = min(5, st.session_state.step + 1)
        st.rerun()

# Custom CSS
st.markdown("""
<style>
    .main-title { font-size: 2.5em; color: #4CAF50; text-align: center; margin-bottom: 20px; }
    .step-box { background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
    .persona-card { background-color: #e8f4f8; padding: 15px; border-radius: 8px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .summary-box { background-color: #d4edda; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    .suggestion-box { background-color: #fff3cd; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    .cta-box { background-color: #e2e3e5; padding: 20px; border-radius: 10px; text-align: center; margin-top: 30px; }
    .cta-box button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
    .cta-box button:hover { background-color: #45a049; }
</style>
""", unsafe_allow_html=True)

def read_pdf(file) -> str:
    """Extract text from uploaded PDF."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def generate_mock_dilemma():
    """Generate a mock decision dilemma."""
    scenarios = [
        {
            "dilemma": "Should the city allocate its annual budget surplus to improve public transportation or to fund affordable housing initiatives?\n\n"
                      "Process: 1. Budget Committee reviews options over 2 months. 2. Public consultation in month 3. 3. City Council votes in month 4.\n"
                      "Stakeholders: Sarah (Budget Committee Chair), James (Public Transit Advocate), Maria (Housing Authority Director), Tom (City Council Member)"
        },
        {
            "dilemma": "Should the company invest in a new AI-driven product line or focus on expanding its existing market share in traditional products?\n\n"
                      "Process: 1. R&D team evaluates feasibility (3 months). 2. Marketing assesses market demand (1 month). 3. Board decides (2 weeks).\n"
                      "Stakeholders: Alex (CEO), Rachel (R&D Director), Sam (Marketing VP), Lisa (CFO)"
        }
    ]
    return random.choice(scenarios)["dilemma"]

def save_persona_to_json(persona: Dict, filename: str):
    """Save persona to a JSON file."""
    try:
        with open(f"personas/{filename}", "w") as f:
            json.dump(persona, f, indent=2)
    except Exception as e:
        st.error(f"Error saving persona to JSON: {str(e)}")

def load_persona_from_json(filename: str) -> Dict:
    """Load persona from a JSON file or return embedded data."""
    try:
        with open(f"personas/{filename}", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        for persona in HARDCODED_PERSONAS:
            if persona["name"].replace(" ", "_").lower() + ".json" == filename:
                return persona
        st.error(f"Persona file {filename} not found and not in embedded data.")
        return {}
    except Exception as e:
        st.error(f"Error loading persona from JSON: {str(e)}")
        return {}

def display_persona_cards(personas: List[Dict]):
    """Display personas as a card deck with editable fields."""
    cols = st.columns(3)
    for i, persona in enumerate(personas):
        with cols[i % 3]:
            st.markdown(f"**{persona['name']} ({persona.get('role', 'Unknown Role')})**", unsafe_allow_html=True)
            with st.form(key=f"edit_persona_card_{i}", clear_on_submit=True):
                name = st.text_input("Name", value=persona["name"], key=f"card_name_{i}")
                role = st.text_input("Role/Title", value=persona.get("role", "Unknown Role"), key=f"card_role_{i}")
                bio = st.text_area("Bio", value=persona["bio"], height=100, key=f"card_bio_{i}")
                psychological_traits = st.text_area("Psychological Traits", value=", ".join(persona.get("psychological_traits", persona["goals"])), key=f"card_traits_{i}")
                influences = st.text_area("Influences", value=", ".join(persona.get("influences", persona["biases"])), key=f"card_influences_{i}")
                biases = st.text_area("Biases", value=", ".join(persona["biases"]), key=f"card_biases_{i}")
                historical_behavior = st.text_area("Historical Behavior", value=persona.get("historical_behavior", persona["expected_behavior"]), key=f"card_behavior_{i}")
                tone = st.text_input("Tone", value=persona["tone"], key=f"card_tone_{i}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("Update Persona"):
                        updated_persona = {
                            "name": name,
                            "role": role,
                            "bio": bio,
                            "psychological_traits": psychological_traits.split(", "),
                            "influences": influences.split(", "),
                            "biases": biases.split(", "),
                            "historical_behavior": historical_behavior,
                            "tone": tone,
                            "goals": persona["goals"],
                            "expected_behavior": persona["expected_behavior"]
                        }
                        personas[i] = updated_persona
                        save_persona(updated_persona)
                        save_persona_to_json(updated_persona, f"{name.replace(' ', '_').lower()}.json")
                        st.success(f"Persona {name} updated and saved!")
                        st.rerun()
                with col2:
                    if st.form_submit_button("Save to Library"):
                        save_persona(persona)
                        save_persona_to_json(persona, f"{persona['name'].replace(' ', '_').lower()}.json")
                        st.success(f"Persona {persona['name']} saved to library!")
                        st.rerun()
            try:
                if st.button("Replace with Library Persona", key=f"replace_persona_{i}"):
                    st.session_state.replace_index[i] = True
                    st.rerun()
                if st.session_state.replace_index.get(i, False):
                    saved_personas = get_all_personas()
                    hardcoded_personas = HARDCODED_PERSONAS
                    library_options = [p["name"] for p in saved_personas + hardcoded_personas if p]
                    if library_options:
                        selected_persona = st.selectbox("Select Persona", library_options, key=f"select_persona_{i}")
                        if st.button("Confirm Replace", key=f"confirm_replace_{i}"):
                            for p in saved_personas + hardcoded_personas:
                                if p and p["name"] == selected_persona:
                                    personas[i] = p
                                    save_persona(p)
                                    save_persona_to_json(p, f"{p['name'].replace(' ', '_').lower()}.json")
                                    st.session_state.replace_index[i] = False
                                    st.rerun()
                    else:
                        st.warning("No personas in library.")
            except Exception as e:
                st.error(f"Error in replace persona: {str(e)}")

def display_process_visualization(process: List[str]):
    """Display the decision-making process as ASCII timeline, graph, and a networkx graph."""
    st.markdown("### Decision-Making Process")
    ascii_timeline = "=== Process Timeline ===\n"
    for i, step in enumerate(process, 1):
        ascii_timeline += f"{i}. {step}\n"
    ascii_timeline += "======================="
    st.code(ascii_timeline)
    ascii_graph = "=== Process Dependency Graph ===\n"
    for i, step in enumerate(process, 1):
        ascii_graph += f"[{step}]"
        if i < len(process):
            ascii_graph += " --> "
        if i % 2 == 0:
            ascii_graph += "\n"
    ascii_graph += "\n==============================="
    st.code(ascii_graph)
    st.markdown("#### Process Flowchart")
    try:
        # Ensure networkx is available
        if not hasattr(nx, 'DiGraph'):
            raise ImportError("networkx module is not properly loaded")
        
        G = nx.DiGraph()
        for i, step in enumerate(process):
            G.add_node(f"S{i+1}", label=step)
            if i < len(process) - 1:
                G.add_edge(f"S{i+1}", f"S{i+2}")
        G.add_node("End", label="End")
        G.add_edge(f"S{len(process)}", "End")
        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', arrows=True)
        st.pyplot(plt)
        plt.close()
    except ImportError as e:
        st.error(f"Failed to generate process graph: {str(e)}. Please ensure networkx is installed.")
    except Exception as e:
        st.error(f"Failed to generate process graph: {str(e)}")

def main():
    st.markdown("<h1 class='main-title'>DecisionTwin for Decision Making</h1>", unsafe_allow_html=True)

    # Step 0: Password Authentication
    if st.session_state.step == 0:
        st.image("https://github.com/sargonx646/DF_22AprilLate/raw/main/assets/decisionforge_logo.png.png", use_column_width=True)
        st.markdown("### Please Enter the Password to Access the App")
        password_input = st.text_input("Password", type="password", key="password")
        if st.button("Submit", key="submit_password"):
            if password_input == "Simulation2025":
                st.session_state.step = 1
                st.success("Access granted! Proceeding to the app.")
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")

    # Step 1: Define Decision
    elif st.session_state.step == 1:
        st.header("Step 1: Define Your Decision")
        st.info("Provide details about your decision dilemma, process, stakeholders, and upload a PDF for context.")
        if st.button("Generate Mock Dilemma", key="mock_dilemma"):
            st.session_state.dilemma = generate_mock_dilemma()
            st.rerun()
        with st.form(key="decision_form"):
            context_input = st.text_area(
                "Describe the decision dilemma, process, and stakeholders:",
                height=200,
                value=st.session_state.dilemma,
                placeholder="E.g., Allocate $10M budget across departments. Involves CEO, CFO, HR, and department heads.",
                key="context_input"
            )
            uploaded_file = st.file_uploader("Upload a PDF with additional context (optional)", type="pdf", key="pdf_upload")
            if st.form_submit_button("Extract Decision Structure"):
                if context_input.strip():
                    if uploaded_file:
                        pdf_text = read_pdf(uploaded_file)
                        context_input += "\n\nPDF Context:\n" + pdf_text
                    try:
                        with st.spinner("Extracting decision structure..."):
                            st.session_state.extracted = extract_decision_structure(context_input, context_input, "")
                            st.session_state.dilemma = context_input
                        st.session_state.step = 2
                        st.success("Decision structure extracted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error extracting decision structure: {str(e)}")
                else:
                    st.error("Please provide a decision context.")

    # Step 2: Review Personas and Process
    elif st.session_state.step == 2:
        st.header("Step 2: Review Personas and Process")
        st.info("Review and modify the AI-generated personas and process. Use the persona library to swap or save personas.")
        st.markdown("### Decision Context")
        st.markdown(f'<div class="step-box">{st.session_state.dilemma}</div>', unsafe_allow_html=True)
        st.markdown("### Personas")
        if not st.session_state.personas:
            if st.button("Generate Personas", key="generate_personas"):
                try:
                    with st.spinner("Generating personas..."):
                        st.session_state.personas = generate_personas(st.session_state.extracted)
                        st.session_state.replace_index = {}
                        for persona in st.session_state.personas:
                            save_persona(persona)
                            save_persona_to_json(persona, f"{persona['name'].replace(' ', '_').lower()}.json")
                    st.success("Personas generated and saved successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to generate personas: {str(e)}")
        else:
            display_persona_cards(st.session_state.personas)
        st.markdown("### Persona Library")
        saved_personas = get_all_personas()
        if saved_personas:
            with st.expander("View/Edit Persona Library", expanded=False):
                for persona in saved_personas:
                    if not persona.get("id"):
                        continue
                    with st.form(key=f"edit_db_persona_{persona['id']}", clear_on_submit=True):
                        name = st.text_input("Name", value=persona["name"], key=f"name_{persona['id']}")
                        goals = st.text_area("Goals", value=", ".join(persona["goals"]), key=f"goals_{persona['id']}")
                        biases = st.text_area("Biases", value=", ".join(persona["biases"]), key=f"biases_{persona['id']}")
                        tone = st.text_input("Tone", value=persona["tone"], key=f"tone_{persona['id']}")
                        bio = st.text_area("Bio", value=persona["bio"], height=150, key=f"bio_{persona['id']}")
                        expected_behavior = st.text_area("Expected Behavior", value=persona["expected_behavior"], height=100, key=f"behavior_{persona['id']}")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.form_submit_button("Update Persona"):
                                updated_persona = {
                                    "id": persona["id"],
                                    "name": name,
                                    "goals": goals.split(", "),
                                    "biases": biases.split(", "),
                                    "tone": tone,
                                    "bio": bio,
                                    "expected_behavior": expected_behavior
                                }
                                update_persona(updated_persona)
                                save_persona_to_json(updated_persona, f"{name.replace(' ', '_').lower()}.json")
                                st.success(f"Persona {name} updated in database!")
                                st.rerun()
                        with col2:
                            if st.form_submit_button("Delete Persona"):
                                delete_persona(persona["id"])
                                st.success(f"Persona {name} deleted from database!")
                                st.rerun()
        else:
            st.write("No personas in library.")
        st.markdown("### Decision Process")
        if st.session_state.extracted.get("process"):
            display_process_visualization(st.session_state.extracted["process"])
        if st.button("Launch Simulation", key="launch_simulation"):
            st.session_state.step = 3
            st.rerun()

    # Step 3: Run Simulation
    elif st.session_state.step == 3:
        st.header("Step 3: Run Simulation")
        st.info("Select a simulation method to model stakeholder debates.")
        st.write("Debug: Dilemma:", st.session_state.dilemma[:100] + "..." if len(st.session_state.dilemma) > 100 else st.session_state.dilemma)
        st.write("Debug: Personas:", [p["name"] for p in st.session_state.personas])
        st.write("Debug: Extracted Process:", st.session_state.extracted.get("process", []))
        simulation_type = st.selectbox(
            "Simulation Method:",
            [
                "Grok 3 Beta Simulation",
                "AgentIQ Simulation (Work in Progress)",
                "Monte Carlo Simulation",
                "Game Theory Simulation"
            ],
            key="simulation_type"
        )
        simulation_time_minutes = st.slider(
            "Set Maximum Simulation Time (minutes):",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            key="simulation_time"
        )
        simulation_time_seconds = simulation_time_minutes * 60
        if st.button("Start Simulation", key="start_simulation"):
            try:
                with st.spinner(f"Running {simulation_type} (timeout: {simulation_time_minutes} minutes)..."):
                    dilemma = str(st.session_state.dilemma) if st.session_state.dilemma else "Unknown dilemma"
                    if simulation_type == "AgentIQ Simulation (Work in Progress)":
                        st.warning("AgentIQ Simulation is under development and not yet available.")
                        st.session_state.transcript = [{
                            "agent": "System",
                            "round": 1,
                            "step": "N/A",
                            "message": "AgentIQ Simulation is not implemented. Please select another method."
                        }]
                    else:
                        st.session_state.transcript = simulate_debate(
                            personas=st.session_state.personas,
                            dilemma=dilemma,
                            process_hint=dilemma,
                            extracted=st.session_state.extracted,
                            scenarios="",
                            max_simulation_time=simulation_time_seconds,
                            simulation_type=simulation_type
                        )
                st.session_state.step = 4
                st.success("Simulation complete!")
                st.rerun()
            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")

    # Step 4: Watch Debate
    elif st.session_state.step == 4:
        st.header("Step 4: Watch the Debate")
        st.info("Follow the simulated debate among stakeholders.")
        for entry in st.session_state.transcript:
            st.markdown(f"**{entry['agent']} (Round {entry['round']}, {entry['step']})**")
            st.write(entry['message'])
            st.markdown("---")
        if st.button("Analyze Results", key="analyze_results"):
            try:
                with st.spinner("Generating summary, suggestions, and visualizations..."):
                    st.session_state.summary, st.session_state.suggestion = generate_summary_and_suggestion(st.session_state.transcript)
                    analysis_input = json.dumps({"transcript": st.session_state.transcript, "dilemma": st.session_state.dilemma})
                    st.session_state.analysis = json.loads(transcript_analyzer(analysis_input))
                    st.session_state.keywords = [word for entry in st.session_state.transcript for word in entry['message'].split() if len(word) > 5]
                    generate_visualizations(st.session_state.keywords, st.session_state.transcript, st.session_state.personas)
                st.session_state.step = 5
                st.success("Analysis complete!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to generate analysis: {str(e)}")
                st.session_state.step = 5
                st.rerun()

    # Step 5: View Results
    elif st.session_state.step == 5:
        st.header("Step 5: Unlock Your Insights")
        st.info("Explore the simulation results, optimization suggestions, and visualizations.")
        st.markdown("### Decision Summary")
        st.markdown(f'<div class="summary-box">{st.session_state.summary}</div>', unsafe_allow_html=True)
        st.markdown("### Optimization Suggestion")
        st.markdown(f'<div class="suggestion-box">{st.session_state.suggestion}</div>', unsafe_allow_html=True)
        st.markdown("### Negotiation Analysis")
        analysis = st.session_state.analysis
        if analysis.get("topics"):
            st.markdown("**Key Topics**")
            for topic in analysis["topics"]:
                st.write(f"- {topic['label']}: {', '.join(topic['keywords'])} (Weight: {topic['weight']:.2f})")
        if analysis.get("sentiment_analysis"):
            st.markdown("**Sentiment Analysis**")
            for sentiment in analysis["sentiment_analysis"]:
                st.write(f"- {sentiment['agent']} (Round {sentiment['round']}): {sentiment['tone']} (Score: {sentiment['score']:.2f})")
        if analysis.get("key_arguments"):
            st.markdown("**Key Arguments**")
            for arg in analysis["key_arguments"]:
                st.write(f"- {arg['agent']}: {arg['type']} - {arg['content']}")
        if analysis.get("conflicts"):
            st.markdown("**Negotiation Conflicts**")
            for conflict in analysis["conflicts"]:
                st.write(f"- {conflict['issue']} (Involving: {', '.join(conflict['stakeholders'])})")
        if analysis.get("insights"):
            st.markdown("**Insights**")
            st.write(analysis["insights"])
        if analysis.get("improvement_areas"):
            st.markdown("**Areas for Improvement**")
            for area in analysis["improvement_areas"]:
                st.write(f"- {area}")
        if analysis.get("process_suggestions"):
            st.markdown("**Suggestions for Smoother Negotiation**")
            for suggestion in analysis["process_suggestions"]:
                st.write(f"- {suggestion}")

        st.markdown("### Visual Insights")
        st.subheader("Word Cloud")
        try:
            fig = plt.figure(figsize=(10, 5))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(st.session_state.keywords))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"Failed to generate word cloud: {str(e)}")

        st.subheader("Stakeholder Interaction Network")
        try:
            import plotly.graph_objects as go
            G = nx.DiGraph()
            agents = list(set(entry['agent'] for entry in st.session_state.transcript))
            for agent in agents:
                G.add_node(agent)
            for i, entry in enumerate(st.session_state.transcript[:-1]):
                G.add_edge(entry['agent'], st.session_state.transcript[i+1]['agent'])
            pos = nx.spring_layout(G)
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            node_x, node_y = [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
            node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()), textposition='top center', marker=dict(size=10, color='lightblue'))
            fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0), xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False)))
            fig.update_layout(title="Stakeholder Interaction Network")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Failed to generate interaction network: {str(e)}")

        st.subheader("Topic Distribution")
        try:
            if analysis.get("topics"):
                df = pd.DataFrame([(t['label'], t['weight']) for t in analysis["topics"]], columns=["Topic", "Weight"])
                fig = px.bar(df, x="Topic", y="Weight", title="Topic Distribution in Debate")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Failed to generate topic distribution: {str(e)}")

        st.subheader("Sentiment Trend")
        try:
            if analysis.get("sentiment_analysis"):
                df = pd.DataFrame([(s['agent'], s['round'], s['score']) for s in analysis["sentiment_analysis"]], columns=["Agent", "Round", "Sentiment"])
                fig = px.line(df, x="Round", y="Sentiment", color="Agent", title="Sentiment Trend Over Rounds")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Failed to generate sentiment trend: {str(e)}")

        st.markdown("### Export Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.download_button(
                label="📄 Transcript (JSON)",
                data=json.dumps(st.session_state.transcript, indent=2),
                file_name="transcript.json",
                mime="application/json",
                key="download_transcript"
            )
        with col2:
            st.download_button(
                label="📝 Summary (TXT)",
                data=st.session_state.summary,
                file_name="summary.txt",
                mime="text/plain",
                key="download_summary"
            )
        with col3:
            try:
                buf = BytesIO()
                fig = plt.figure(figsize=(10, 5))
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(st.session_state.keywords))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.savefig(buf, format="png")
                plt.close()
                buf.seek(0)
                st.download_button(
                    label="🖼️ Word Cloud (PNG)",
                    data=buf,
                    file_name="word_cloud.png",
                    mime="image/png",
                    key="download_word_cloud"
                )
            except Exception as e:
                st.warning(f"Failed to generate word cloud for download: {str(e)}")
        with col4:
            st.download_button(
                label="📊 Analysis (JSON)",
                data=json.dumps(st.session_state.analysis, indent=2),
                file_name="analysis.json",
                mime="application/json",
                key="download_analysis"
            )
        st.markdown('''
        <div class="cta-box">
            <h3>Loved the Experience?</h3>
            <p>Start a new simulation to explore more possibilities!</p>
            <button onclick="restartSimulation()">Start New Simulation</button>
        </div>
        <script>
            function restartSimulation() {
                window.location.reload();
            }
        </script>
        ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
