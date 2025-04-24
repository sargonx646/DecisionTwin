import streamlit as st
import json
import os
import random
import PyPDF2
from io import BytesIO
from typing import List, Dict
from agents.extractor import extract_decision_structure
from agents.persona_builder import generate_personas
from agents.debater import simulate_debate
from agents.summarizer import generate_summary_and_suggestion
from agents.transcript_analyzer import transcript_analyzer
from utils.visualizer import generate_visuals
from utils.db import save_persona, get_all_personas, init_db, update_persona, delete_persona

# Initialize database
init_db()

# Check for API key
if not os.getenv("XAI_API_KEY"):
    st.error("XAI_API_KEY environment variable is not set. Please configure it in .env.")
    st.stop()

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

def display_persona_cards(personas: List[Dict]):
    """Display personas as a card deck with expandable details."""
    cols = st.columns(3)
    for i, persona in enumerate(personas):
        with cols[i % 3]:
            with st.expander(f"{persona['name']} ({persona.get('role', 'Unknown Role')})", expanded=False):
                st.markdown(f"**Goals**: {', '.join(persona['goals'])}")
                st.markdown(f"**Biases**: {', '.join(persona['biases'])}")
                st.markdown(f"**Tone**: {persona['tone'].capitalize()}")
                st.markdown(f"**Bio**: {persona['bio']}")
                st.markdown(f"**Expected Behavior**: {persona['expected_behavior']}")
                if st.button("Save to Library", key=f"save_persona_{i}"):
                    save_persona(persona)
                    st.success(f"Persona {persona['name']} saved to library.")
                if st.button("Replace with Library Persona", key=f"replace_persona_{i}"):
                    st.session_state[f"replace_index_{i}"] = True
                if st.session_state.get(f"replace_index_{i}", False):
                    saved_personas = get_all_personas()
                    if saved_personas:
                        library_options = [p["name"] for p in saved_personas]
                        selected_persona = st.selectbox("Select Persona", library_options, key=f"select_persona_{i}")
                        if st.button("Confirm Replace", key=f"confirm_replace_{i}"):
                            persona_index = library_options.index(selected_persona)
                            personas[i] = saved_personas[persona_index]
                            st.session_state[f"replace_index_{i}"] = False
                            st.rerun()
                    else:
                        st.warning("No personas in library.")

def display_process_visualization(process: List[str]):
    """Display the decision-making process as a stylized ASCII and markdown flowchart."""
    st.markdown("### Decision-Making Process")
    ascii_timeline = "=== Process Timeline ===\n"
    for i, step in enumerate(process, 1):
        ascii_timeline += f"{i}. {step}\n"
    ascii_timeline += "======================="
    st.code(ascii_timeline)
    st.markdown("#### Process Flowchart")
    flowchart = "```mermaid\ngraph TD\n"
    for i, step in enumerate(process, 1):
        flowchart += f"    S{i}[{step}] -->|Step {i}| "
        flowchart += f"S{i+1}[{process[i] if i < len(process) else 'End'}]\n"
    flowchart += "```"
    st.markdown(flowchart)

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
        
        # Mock dilemma button
        if st.button("Generate Mock Dilemma", key="mock_dilemma"):
            st.session_state.dilemma = generate_mock_dilemma()
            st.rerun()

        # Decision input form
        with st.form(key="decision_form"):
            context_input = st.text_area(
                "Describe the decision dilemma, process, and stakeholders:",
                height=200,
                value=st.session_state.dilemma,
                placeholder="E.g., Allocate $10M budget across departments. Involves CEO, CFO, HR, and department heads.",
                key="context_input"
            )
            uploaded_file = st.file_uploader("Upload a PDF with additional context (optional)", type="pdf", key="pdf_upload")
            submitted = st.form_submit_button("Extract Decision Structure")
            if submitted:
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
                        for persona in st.session_state.personas:
                            save_persona(persona)
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
                    with st.form(f"edit_db_persona_{persona['id']}", clear_on_submit=True):
                        name = st.text_input("Name", value=persona["name"], key=f"name_{persona['id']}")
                        goals = st.text_area("Goals", value=", ".join(persona["goals"]), key=f"goals_{persona['id']}")
                        biases = st.text_area("Biases", value=", ".join(persona["biases"]), key=f"biases_{persona['id']}")
                        tone = st.text_input("Tone", value=persona["tone"], key=f"tone_{persona['id']}")
                        bio = st.text_area("Bio", value=persona["bio"], height=150, key=f"bio_{persona['id']}")
                        expected_behavior = st.text_area("Expected Behavior", value=persona["expected_behavior"], height=100, key=f"behavior_{persona['id']}")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.form_submit_button("Update Persona", key=f"update_persona_{persona['id']}"):
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
                            if st.form_submit_button("Delete Persona", key=f"delete_persona_{persona['id']}"):
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
        st.info("Simulate the debate among stakeholders using Grok-3-Beta.")
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
                with st.spinner(f"Running Grok-3-Beta simulation (timeout: {simulation_time_minutes} minutes)..."):
                    st.session_state.transcript = simulate_debate(
                        personas=st.session_state.personas,
                        dilemma=st.session_state.dilemma,
                        process_hint=st.session_state.dilemma,
                        extracted=st.session_state.extracted,
                        scenarios="",
                        max_simulation_time=simulation_time_seconds
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
                    keywords = [word for entry in st.session_state.transcript for word in entry['message'].split() if len(word) > 5]
                    generate_visuals(keywords, st.session_state.transcript)
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
        if analysis.get("themes"):
            st.markdown("**Key Themes**")
            st.write(", ".join(analysis["themes"]))
        if analysis.get("conflicts"):
            st.markdown("**Negotiation Conflicts**")
            for conflict in analysis["conflicts"]:
                st.write(f"- {conflict} - Potential issue: Misaligned priorities or power imbalance.")
        if analysis.get("negotiation_issues"):
            st.markdown("**Negotiation Issues**")
            for issue in analysis["negotiation_issues"]:
                st.write(f"- {issue}")
        if analysis.get("insights"):
            st.markdown("**Insights**")
            st.write(analysis["insights"])
        st.markdown("### Visual Insights")
        st.subheader("Word Cloud")
        try:
            st.image("word_cloud.png", use_column_width=True)
        except FileNotFoundError:
            st.warning("Word cloud unavailable.")
        st.subheader("Stakeholder Interaction Network")
        try:
            with open("network_graph.png", "rb") as f:
                st.image(f, use_column_width=True)
        except FileNotFoundError:
            st.warning("Network graph unavailable.")
        st.subheader("Negotiation Conflict Heatmap")
        try:
            st.image("conflict_heatmap.png", use_column_width=True)
        except FileNotFoundError:
            st.warning("Conflict heatmap unavailable.")
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
                with open("word_cloud.png", "rb") as f:
                    st.download_button(
                        label="🖼️ Word Cloud (PNG)",
                        data=f,
                        file_name="word_cloud.png",
                        mime="image/png",
                        key="download_word_cloud"
                    )
            except FileNotFoundError:
                st.warning("Word cloud unavailable.")
        with col4:
            try:
                with open("conflict_heatmap.png", "rb") as f:
                    st.download_button(
                        label="📊 Conflict Heatmap (PNG)",
                        data=f,
                        file_name="conflict_heatmap.png",
                        mime="image/png",
                        key="download_heatmap"
                    )
            except FileNotFoundError:
                st.warning("Conflict heatmap unavailable.")
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
