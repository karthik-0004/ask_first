# AskFirst — Clary Pattern Detector

## Overview
AskFirst — Clary Pattern Detector analyzes synthetic health conversation logs to uncover cross-session trends that users often miss on their own. It uses GPT-4o to reason over timeline-aware context and identify hidden cause-effect health patterns across multiple timestamps. The final results are shown in a Streamlit interface and exported as structured JSON with confidence scoring.

## Tech Stack
- Python 3.11+
- OpenAI GPT-4o (via openai SDK v1.x)
- Streamlit
- python-dotenv

## Why GPT-4o?
GPT-4o was selected because this assignment depends on reasoning over long, multi-session conversation histories where timing and sequence matter, and the model can reliably handle larger prompt contexts. The task is not simple extraction, it is temporal inference, where a symptom may appear weeks after a behavior shift, and GPT-4o performs better than GPT-3.5 on these nuanced cause-effect chains. The model is also more consistent at producing strict JSON-only outputs under schema-like constraints, which is essential for downstream validation and UI rendering. In practice, GPT-4o delivers stronger medical-style explanatory reasoning when asked to connect behavioral patterns with symptom evolution over time. This made it the most practical choice for a reliable internship-grade prototype.

## Chunking & Context Management Strategy
For each user, the system assembles a single structured context block that includes profile details and conversation sessions in chronological order. Before the full history, it prepends a compact temporal timeline summary so the model gets a quick index of session progression and key tags. Sessions are sorted ascending by timestamp to preserve directionality of events, which is critical for temporal causality analysis. To stay within prompt limits on larger histories, the context builder keeps only the most recent 10 sessions when a user has more than 10 entries. The system prompt explicitly forces temporal-gap reasoning, so Clary evaluates how much time passes between candidate causes and observed effects rather than matching isolated keywords.

## Project Structure
```text
askfirst/
├── app.py
├── config.py
├── data/
│   └── askfirst_synthetic_dataset.json
├── core/
│   ├── __init__.py
│   ├── context_builder.py
│   ├── loader.py
│   ├── pattern_engine.py
│   └── scorer.py
├── utils/
│   ├── __init__.py
│   └── helpers.py
├── requirements.txt
├── README.md
└── WRITEUP.md
```

## Setup
1. Clone the repo.
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
3. Place `askfirst_synthetic_dataset.json` in the `data/` folder.
4. Create a `.env` file in the project root with:
	```env
	OPENAI_API_KEY=your_key_here
	```
5. Run the app:
	```bash
	streamlit run app.py
	```
6. In the sidebar, enter your API key (or let it load from `.env`), choose Single User or All Users, then click Run Pattern Detection.

## How It Works
1. Load dataset records from JSON and select one user or all users.
2. Build structured context with profile data, sorted sessions, and a temporal timeline summary.
3. Send context to GPT-4o with a strict temporal-reasoning system prompt.
4. Parse the model response as JSON patterns with evidence fields.
5. Validate patterns, normalize confidence scores, and filter by threshold.
6. Render streaming output and final pattern cards in Streamlit, with downloadable JSON.
