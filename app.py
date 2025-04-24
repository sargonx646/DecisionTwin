import streamlit as st
import json
import os
import pkg_resources
import random
from agents.extractor import extract_decision_structure
from agents.persona_builder import generate_personas
from agents.debater import simulate_debate
try:
    from agents.agent_iq_debater import simulate_debate_agent_iq
    agentiq_available = True
except ImportError:
    agentiq_available = False
    simulate_debate_agent_iq = None
from agents.summarizer import generate_summary_and_suggestion
from utils.visualizer import generate_visuals
from utils.db import save_persona, get_all_personas, init_db, update_persona, delete_persona

# Initialize database
init_db()

# Log installed packages for debugging
installed_packages = [f"{p.key}=={p.version}" for p in pkg_resources.working_set]
st.write("Debug: Installed packages:", installed_packages)

# Check for API keys
if not os.getenv("XAI_API_KEY"):
    st.error("XAI_API_KEY environment variable is not set. Please configure it to use the Grok-3-Beta simulation.")
    st.stop()
if not os.getenv("NVIDIA_API_KEY") and agentiq_available:
    st.error("NVIDIA_API_KEY environment variable is not set. Please configure it to use the AgentIQ simulation.")
    st.stop()
if not os.getenv("TAVILY_API_KEY") and agentiq_available:
    st.warning("TAVILY_API_KEY is not set. AgentIQ simulation may have limited functionality.")

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 0
if "dilemma" not in st.session_state:
    st.session_state.dilemma = ""
if "process_hint" not in st.session_state:
    st.session_state.process_hint = ""
if "scenarios" not in st.session_state:
    st.session_state.scenarios = ""
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

# Sidebar with logo, progress bar, and navigation
st.sidebar.image("https://github.com/sargonx646/DF_22AprilLate/raw/main/assets/decisionforge_logo.png.png", use_column_width=True)
st.sidebar.markdown("<h2 style='text-align: center;'>DecisionTwin</h2>", unsafe_allow_html=True)
progress = st.session_state.step / 5
st.sidebar.progress(progress)
st.sidebar.markdown(f"**Step {st.session_state.step} of 5**")
if st.session_state.step > 0:
    if st.sidebar.button("Back"):
        st.session_state.step = max(0, st.session_state.step - 1)
        st.rerun()
if st.session_state.step < 5:
    if st.sidebar.button("Forward"):
        st.session_state.step = min(5, st.session_state.step + 1)
        st.rerun()

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5em;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 20px;
    }
    .step-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .persona-card {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .summary-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .suggestion-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .cta-box {
        background-color: #e2e3e5;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 30px;
    }
    .cta-box button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .cta-box button:hover {
        background-color: #45a049;
    }
    .mock-button {
        background-color: #ff0000;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .mock-button:hover {
        background-color: #cc0000;
    }
</style>
""", unsafe_allow_html=True)

# Mock decision dilemma generator
def generate_mock_dilemma():
    scenarios = [
        {
            "type": "Government",
            "dilemma": "Should the city allocate its annual budget surplus to improve public transportation or to fund affordable housing initiatives?",
            "process_hint": "1. Budget Committee reviews options over 2 months. 2. Public consultation in month 3. 3. City Council votes in month 4.\nStakeholders:\n- Sarah: Budget Committee Chair\n- James: Public Transit Advocate\n- Maria: Housing Authority Director\n- Tom: City Council Member",
            "scenarios": "Potential budget cuts from state funding or increased construction costs could impact either initiative."
        },
        {
            "type": "Foreign Policy",
            "dilemma": "Should the country increase diplomatic efforts or impose economic sanctions to address a neighboring nation's aggressive border policies?",
            "process_hint": "1. Foreign Ministry assesses situation (1 month). 2. Interagency meeting with Defense and Trade (2 weeks). 3. Cabinet decision (1 month).\nStakeholders:\n- Emma: Foreign Minister\n- General Lee: Defense Advisor\n- Clara: Trade Secretary\n- Ambassador Patel: Regional Expert",
            "scenarios": "Escalation of border tensions or international pressure from allies could influence the decision."
        },
        {
            "type": "Corporate",
            "dilemma": "Should the company invest in a new AI-driven product line or focus on expanding its existing market share in traditional products?",
            "process_hint": "1. R&D team evaluates feasibility (3 months). 2. Marketing assesses market demand (1 month). 3. Board decides (2 weeks).\nStakeholders:\n- Alex: CEO\n- Rachel: R&D Director\n- Sam: Marketing VP\n- Lisa: CFO",
            "scenarios": "Competitor launches a similar AI product, or economic downturn reduces consumer spending."
        },
        {
            "type": "HR Dispute",
            "dilemma": "Should the company implement a mandatory return-to-office policy or maintain a hybrid work model to resolve employee dissatisfaction?",
            "process_hint": "1. HR conducts employee surveys (1 month). 2. Management reviews feedback (2 weeks). 3. Policy decision by executives (1 week).\nStakeholders:\n- Olivia: HR Director\n- Mark: Operations Manager\n- Sophia: Employee Representative\n- David: CEO",
            "scenarios": "New health regulations or a competitor’s fully remote policy could impact employee retention."
        }
    ]
    return random.choice(scenarios)

# Step 0: Password Authentication
if st.session_state.step == 0:
    st.markdown("<h1 class='main-title'>DecisionTwin for Decision Making</h1>", unsafe_allow_html=True)
    st.image("https://github.com/sargonx646/DF_22AprilLate/raw/main/assets/decisionforge_logo.png.png", use_column_width=True)
    st.markdown("### Please Enter the Password to Access the App")
    
    password_input = st.text_input("Password", type="password")
    
    if st.button("Submit"):
        if password_input == "Simulation2025":
            st.session_state.step = 1
            st.success("Access granted! Proceeding to the app.")
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")

# Step 1: Define Your Decision
elif st.session_state.step == 1:
    st.markdown("<h1 class='main-title'>DecisionTwin for Decision Making</h1>", unsafe_allow_html=True)
    st.header("Step 1: Define Your Decision")
    st.info("Provide details about your decision dilemma, decision-making process, stakeholders, and any alternative scenarios.")
    
    # Mock dilemma button outside the form
    if st.button("Generate a Mock Decision Dilemma", key="mock_dilemma", help="Generate a random decision scenario"):
        mock = generate_mock_dilemma()
        st.session_state.dilemma = mock["dilemma"]
        st.session_state.process_hint = mock["process_hint"]
        st.session_state.scenarios = mock["scenarios"]
        st.rerun()

    with st.form("decision_form"):
        st.markdown("### Decision Dilemma")
        dilemma = st.text_area("What decision are you trying to make? Be specific about the problem or dilemma.", height=100, value=st.session_state.dilemma)
        
        st.markdown("### Decision-Making Process and Stakeholders")
        process_hint = st.text_area("Describe the decision-making process, timeline, and stakeholders involved. Optionally, include their roles or titles.", height=150, value=st.session_state.process_hint)
        
        st.markdown("### Alternative Scenarios or External Factors (Optional)")
        scenarios = st.text_area("Describe any alternative scenarios or external factors that might impact the decision (e.g., budget cuts, political changes).", height=100, value=st.session_state.scenarios)
        
        submitted = st.form_submit_button("Extract Decision Structure")
        
        # Load personas from database
        st.markdown("### Load Stakeholders from Database")
        saved_personas = get_all_personas()
        if saved_personas:
            selected_personas = st.multiselect("Select personas to include in process:", [p["name"] for p in saved_personas])
            if st.button("Add Selected Personas to Process"):
                current_process = st.session_state.process_hint
                new_stakeholders = "\n".join([f"- {p['name']}: {p['bio'][:50]}..." for p in saved_personas if p["name"] in selected_personas])
                st.session_state.process_hint = f"{current_process}\n\nSelected Stakeholders:\n{new_stakeholders}" if current_process else new_stakeholders
                st.rerun()
        else:
            st.write("No personas in database.")

        if submitted:
            if not dilemma or not process_hint:
                st.error("Please provide both the decision dilemma and the decision-making process/stakeholders.")
            else:
                st.session_state.dilemma = dilemma
                st.session_state.process_hint = process_hint
                st.session_state.scenarios = scenarios
                try:
                    with st.spinner("Extracting decision structure..."):
                        st.session_state.extracted = extract_decision_structure(dilemma, process_hint, scenarios)
                    st.session_state.step = 2
                    st.success("Decision structure extracted successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to extract decision structure: {str(e)}")

# Step 2: Extract Decision Structure
elif st.session_state.step == 2:
    st.markdown("<h1 class='main-title'>DecisionTwin for Decision Making</h1>", unsafe_allow_html=True)
    st.header("Step 2: Review Decision Structure")
    st.info("Review the extracted decision structure, including the process and stakeholders. Modify if needed.")
    
    st.markdown("### Decision Dilemma")
    st.markdown(f'<div class="step-box">{st.session_state.dilemma}</div>', unsafe_allow_html=True)
    
    st.markdown("### Decision-Making Process and Stakeholders")
    st.markdown(f'<div class="step-box">{st.session_state.process_hint}</div>', unsafe_allow_html=True)
    
    st.markdown("### Extracted Structure")
    extracted_str = json.dumps(st.session_state.extracted, indent=2)
    st.code(extracted_str, language="json")
    
    if st.button("Generate Personas"):
        try:
            with st.spinner("Generating personas..."):
                st.session_state.personas = generate_personas(st.session_state.extracted)
                # Save generated personas to database
                for persona in st.session_state.personas:
                    save_persona(persona)
            st.session_state.step = 3
            st.success("Personas generated and saved successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to generate personas: {str(e)}")
    
    if st.button("Back to Step 1"):
        st.session_state.step = 1
        st.rerun()

# Step 3: Review Personas
elif st.session_state.step == 3:
    st.markdown("<h1 class='main-title'>DecisionTwin for Decision Making</h1>", unsafe_allow_html=True)
    st.header("Step 3: Meet Your Stakeholders")
    st.info("Review and modify the AI-crafted personas, or search previously saved personas.")
    
    # Database of Personas
    st.markdown("### Database of Personas")
    if st.button("View/Edit Personas Database"):
        saved_personas = get_all_personas()
        if saved_personas:
            st.markdown("#### Saved Personas")
            for persona in saved_personas:
                with st.expander(f"{persona['name']} (ID: {persona['id']})"):
                    with st.form(f"edit_db_persona_{persona['id']}"):
                        name = st.text_input("Name", value=persona["name"])
                        goals = st.text_area("Goals", value=", ".join(persona["goals"]))
                        biases = st.text_area("Biases", value=", ".join(persona["biases"]))
                        tone = st.text_input("Tone", value=persona["tone"])
                        bio = st.text_area("Bio", value=persona["bio"], height=150)
                        expected_behavior = st.text_area("Expected Behavior", value=persona["expected_behavior"], height=100)
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
                                st.success(f"Persona {name} updated in database!")
                                st.rerun()
                        with col2:
                            if st.form_submit_button("Delete Persona"):
                                delete_persona(persona["id"])
                                st.success(f"Persona {name} deleted from database!")
                                st.rerun()
        else:
            st.write("No personas in database.")

    st.markdown("### Search Saved Personas")
    saved_personas = get_all_personas()
    persona_names = [p["name"] for p in saved_personas]
    search_query = st.text_input("Search for a persona by name:", "")
    filtered_personas = [p for p in saved_personas if search_query.lower() in p["name"].lower()]
    
    if filtered_personas:
        st.markdown("#### Matching Personas")
        selected_persona = st.selectbox("Select a persona to view or edit:", [p["name"] for p in filtered_personas])
        persona_data = next(p for p in filtered_personas if p["name"] == selected_persona)
        with st.form(f"edit_persona_{selected_persona}"):
            st.markdown("#### Edit Persona")
            name = st.text_input("Name", value=persona_data["name"])
            goals = st.text_area("Goals", value=", ".join(persona_data["goals"]))
            biases = st.text_area("Biases", value=", ".join(persona_data["biases"]))
            tone = st.text_input("Tone", value=persona_data["tone"])
            bio = st.text_area("Bio", value=persona_data["bio"], height=150)
            expected_behavior = st.text_area("Expected Behavior", value=persona_data["expected_behavior"], height=100)
            if st.form_submit_button("Save Changes"):
                updated_persona = {
                    "id": persona_data["id"],
                    "name": name,
                    "goals": goals.split(", "),
                    "biases": biases.split(", "),
                    "tone": tone,
                    "bio": bio,
                    "expected_behavior": expected_behavior
                }
                update_persona(updated_persona)
                st.session_state.personas = [updated_persona if p["name"] == selected_persona else p for p in st.session_state.personas]
                st.success(f"Persona {name} updated and saved to database!")
                st.rerun()
    
    st.markdown("### Current Personas")
    stakeholder_titles = {}
    for line in st.session_state.process_hint.split("\n"):
        if ":" in line and any(s["name"] in line for s in st.session_state.extracted.get("stakeholders", [])):
            name, title = line.split(":", 1)
            name = name.strip().split(".")[-1].strip()
            title = title.strip()
            stakeholder_titles[name] = title
    
    cols = st.columns(3)
    for i, persona in enumerate(st.session_state.personas):
        with cols[i % 3]:
            title = stakeholder_titles.get(persona["name"], "Unknown Role")
            emoji = "🌐" if "EAP" in title else "🩺" if "BHA" in title else "🛡️" if "DoD" in title else "💼" if "EB" in title else "📊" if "OMB" in title else "🏛️" if "Senate" in title else "👤"
            st.markdown(f'''
            <div class="persona-card">
                <h3>{persona['name']}</h3>
                <p><strong>Title:</strong> {title} {emoji}</p>
                <p><strong>Goals:</strong> {', '.join(persona['goals'])}</p>
                <p><strong>Biases:</strong> {', '.join(persona['biases'])}</p>
                <p><strong>Tone:</strong> {persona['tone'].capitalize()}</p>
                <p><strong>Bio:</strong> {persona['bio']}</p>
                <p><strong>Expected Behavior:</strong> {persona['expected_behavior']}</p>
            </div>
            ''', unsafe_allow_html=True)

    st.markdown("### Simulation Settings")
    simulation_options = ["Grok-3-Beta"]
    if agentiq_available:
        simulation_options.append("AgentIQ")
    else:
        st.warning("AgentIQ simulation is unavailable because the 'agentiq' package is not installed. Please install 'agentiq==1.0.0' from NVIDIA's repository (build.nvidia.com) or contact NVIDIA support.")
    simulation_type = st.selectbox(
        "Select Simulation Type:",
        simulation_options,
        help="Choose the simulation type. AgentIQ requires the 'agentiq' package and NVIDIA_API_KEY."
    )
    simulation_time_minutes = st.slider(
        "Set Maximum Simulation Time (minutes):",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="Choose how long the simulation can run before it times out."
    )
    simulation_time_seconds = simulation_time_minutes * 60

    if st.button("Launch Simulation", key="launch_simulation"):
        try:
            with st.spinner(f"Initiating {simulation_type} simulation (will timeout after {simulation_time_minutes} minute{'s' if simulation_time_minutes != 1 else ''})..."):
                if simulation_type == "Grok-3-Beta":
                    st.session_state.transcript = simulate_debate(
                        personas=st.session_state.personas,
                        dilemma=st.session_state.dilemma,
                        process_hint=st.session_state.process_hint,
                        extracted=st.session_state.extracted,
                        scenarios=st.session_state.scenarios,
                        max_simulation_time=simulation_time_seconds
                    )
                elif simulation_type == "AgentIQ" and agentiq_available:
                    st.session_state.transcript = simulate_debate_agent_iq(
                        personas=st.session_state.personas,
                        dilemma=st.session_state.dilemma,
                        process_hint=st.session_state.process_hint,
                        extracted=st.session_state.extracted,
                        scenarios=st.session_state.scenarios,
                        max_simulation_time=simulation_time_seconds
                    )
                else:
                    st.error("AgentIQ simulation is not available. Please select Grok-3-Beta.")
                    st.stop()
            st.session_state.step = 4
            st.success("Simulation complete! Watch the debate unfold.")
            st.rerun()
        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")

# Step 4: Watch the Debate Unfold
elif st.session_state.step == 4:
    st.markdown("<h1 class='main-title'>DecisionTwin for Decision Making</h1>", unsafe_allow_html=True)
    st.header("Step 4: Watch the Debate Unfold")
    st.info("Follow the simulated debate among stakeholders as they navigate your decision dilemma.")
    
    for entry in st.session_state.transcript:
        st.markdown(f"**{entry['agent']} (Round {entry['round']}, {entry['step']})**")
        st.write(entry['message'])
        st.markdown("---")
    
    try:
        with st.spinner("Generating summary, suggestions, and visualizations..."):
            summary, suggestion = generate_summary_and_suggestion(st.session_state.transcript)
            st.session_state.summary = summary
            st.session_state.suggestion = suggestion
            keywords = [word for entry in st.session_state.transcript for word in entry['message'].split() if len(word) > 5]
            generate_visuals(keywords, st.session_state.transcript)
        st.session_state.step = 5
        st.success("Summary and visualizations generated!")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to generate summary or visualizations: {str(e)}")
        st.session_state.step = 5
        st.rerun()

# Step 5: View Results
elif st.session_state.step == 5:
    st.markdown("<h1 class='main-title'>DecisionTwin for Decision Making</h1>", unsafe_allow_html=True)
    st.header("Step 5: Unlock Your Insights")
    st.info("Dive into the simulation results, optimization suggestions, and stunning visualizations.")
    st.markdown("### Decision Summary")
    st.markdown(f'<div class="summary-box">{st.session_state.summary}</div>', unsafe_allow_html=True)
    
    st.markdown("### Optimization Suggestion")
    st.markdown(f'<div class="suggestion-box">{st.session_state.suggestion}</div>', unsafe_allow_html=True)
    
    st.markdown("### Visual Insights")
    st.subheader("Word Cloud of Key Themes")
    try:
        st.image("word_cloud.png", use_column_width=True)
    except FileNotFoundError:
        st.warning("Word cloud unavailable.")

    st.subheader("Stakeholder Interaction Network")
    try:
        with open("network_graph.html", "r") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=500, scrolling=True)
    except FileNotFoundError:
        st.warning("Network graph unavailable.")

    st.subheader("Stakeholder Priorities Over Rounds")
    try:
        with open("timeline_chart.html", "r") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=500, scrolling=True)
    except FileNotFoundError:
        st.warning("Timeline chart unavailable.")

    st.subheader("Sentiment Analysis of Contributions")
    try:
        with open("sentiment_chart.html", "r") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=500, scrolling=True)
    except FileNotFoundError:
        st.warning("Sentiment chart unavailable.")

    st.subheader("Top Terms in Identified Topics")
    try:
        with open("topic_modeling_chart.html", "r") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=500, scrolling=True)
    except FileNotFoundError:
        st.warning("Topic modeling chart unavailable.")
    
    st.markdown("### Export Your Results")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.download_button(
            label="📄 Transcript (JSON)",
            data=json.dumps(st.session_state.transcript, indent=2),
            file_name="transcript.json",
            mime="application/json"
        )
    with col2:
        st.download_button(
            label="📝 Summary (TXT)",
            data=st.session_state.summary,
            file_name="summary.txt",
            mime="text/plain"
        )
    with col3:
        try:
            with open("word_cloud.png", "rb") as f:
                st.download_button(
                    label="🖼️ Word Cloud (PNG)",
                    data=f,
                    file_name="word_cloud.png",
                    mime="image/png"
                )
        except FileNotFoundError:
            st.warning("Word cloud unavailable.")
    with col4:
        try:
            with open("network_graph.html", "r") as f:
                st.download_button(
                    label="📊 Network Graph (HTML)",
                    data=f,
                    file_name="network_graph.html",
                    mime="text/html"
                )
        except FileNotFoundError:
            st.warning("Network graph unavailable.")
    with col5:
        try:
            with open("timeline_chart.html", "r") as f:
                st.download_button(
                    label="⏳ Timeline Chart (HTML)",
                    data=f,
                    file_name="timeline_chart.html",
                    mime="text/html"
                )
        except FileNotFoundError:
            st.warning("Timeline chart unavailable.")
    with col6:
        try:
            with open("sentiment_chart.html", "r") as f:
                st.download_button(
                    label="😊 Sentiment Chart (HTML)",
                    data=f,
                    file_name="sentiment_chart.html",
                    mime="text/html"
                )
        except FileNotFoundError:
            st.warning("Sentiment chart unavailable.")
    
    st.markdown('''
    <div class="cta-box">
        <h3>Loved the Experience?</h3>
        <p>Share your feedback or start a new simulation to explore more possibilities!</p>
        <button onclick="restartSimulation()">Start New Simulation</button>
    </div>
    <script>
        function restartSimulation() {
            window.location.reload();
        }
    </script>
    ''', unsafe_allow_html=True)
