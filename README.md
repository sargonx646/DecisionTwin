# DecisionForge MVP

**DecisionForge** is a revolutionary web app that brings organizational decision-making to life through AI-powered simulations. Using OpenRouter’s **NousResearch/DeepHermes-3-Llama-3-8B-Preview:free** model, it enables leaders to simulate budget allocation decisions, uncover hidden dynamics, and optimize processes with a captivating, user-friendly interface.

## Features
- **Input Dilemma**: Define a budget allocation scenario with stakeholder details.
- **AI-Driven Extraction**: Automatically identify 3–7 stakeholders and process steps.
- **Dynamic Personas**: Generate realistic AI agents with unique goals, biases, and tones.
- **Live Simulation**: Watch a thrilling, real-time debate among agents.
- **Rich Insights**: View summaries, optimization suggestions, and stunning visualizations (word clouds, heatmaps).
- **Export Results**: Download transcripts, summaries, and visuals for analysis.

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your_username/decisionforge-mvp.git
   cd decisionforge-mvp
   ```
2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure Environment**:
   ```bash
   cp .env.example .env
   ```
   Ensure the OpenRouter API key is set in `.env`.
5. **Initialize Database**:
   ```bash
   python -c "from utils.db import init_db; init_db()"
   ```
6. **Run Locally**:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

## Deployment
- Push to a **public** GitHub repository.
- Deploy via [Streamlit Cloud](https://streamlit.io/cloud):
  - Create a new app, select the repository, set `app.py` as the main file.
  - Add `OPENROUTER_API_KEY` to Streamlit Cloud’s environment variables.

## Testing
- Run unit tests:
  ```bash
  pytest tests/
  ```
- Tests use mocked API responses to minimize costs.

## Troubleshooting
- **API Errors**: Verify the OpenRouter API key in `.env`.
- **Deployment Fails**: Check Streamlit Cloud logs for dependency or configuration issues.
- **Visualizations Missing**: Ensure Matplotlib, Seaborn, and WordCloud are installed (`pip install -r requirements.txt`).

## Example Usage
- **Dilemma**: "Allocate $10M budget across departments."
- **Process Hint**: "Involves CEO, CFO, HR, and department heads."
- **Outcome**: See a live debate, get a summary, and receive an optimization suggestion like “Increase HR’s input to balance employee needs.”

## License
MIT License
