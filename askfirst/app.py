from dotenv import load_dotenv
load_dotenv()

import json
import os
import time
from html import escape

import streamlit as st

from core.loader import get_all_users, get_user_by_id, load_dataset
from core.pattern_engine import detect_patterns_all_users, detect_patterns_for_user
from core.scorer import (
	build_final_report,
	filter_patterns,
	normalize_confidence_score,
	patterns_to_json_string,
	validate_pattern,
)
from config import DATA_PATH, STREAM_ENABLED

st.set_page_config(
	page_title="AskFirst - Clary Pattern Detector",
	page_icon="🩺",
	layout="wide",
)

st.markdown(
	"""
<style>
/* Sidebar styling */
[data-testid="stSidebar"] {
	background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}
[data-testid="stSidebar"] * {
	color: white !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stNumberInput label {
	color: #a78bfa !important;
	font-size: 0.8rem !important;
	font-weight: 600 !important;
	letter-spacing: 0.05em !important;
	text-transform: uppercase !important;
}

/* Sidebar field readability */
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * {
	color: #111827 !important;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stNumberInput input {
	color: #111827 !important;
	-caret-color: #111827 !important;
}
[data-testid="stSidebar"] .stTextInput input::placeholder,
[data-testid="stSidebar"] .stNumberInput input::placeholder {
	color: #6b7280 !important;
	opacity: 1 !important;
}

/* Main header */
.main-header {
	background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
	padding: 2rem 2.5rem;
	border-radius: 16px;
	margin-bottom: 1.5rem;
	color: white;
}
.main-header h1 {
	font-size: 2.2rem;
	font-weight: 800;
	margin: 0;
	letter-spacing: -0.02em;
}
.main-header p {
	margin: 0.5rem 0 0 0;
	opacity: 0.85;
	font-size: 1rem;
}

/* Pattern cards */
.pattern-card {
	background: white;
	border-radius: 12px;
	padding: 1.5rem;
	margin-bottom: 1rem;
	border-left: 4px solid #6c63ff;
	box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
.pattern-card-high {
	border-left-color: #10b981;
}
.pattern-card-medium {
	border-left-color: #f59e0b;
}
.pattern-card-low {
	border-left-color: #ef4444;
}

/* Confidence badge */
.badge {
	display: inline-block;
	padding: 0.25rem 0.75rem;
	border-radius: 999px;
	font-size: 0.75rem;
	font-weight: 700;
	letter-spacing: 0.05em;
	text-transform: uppercase;
}
.badge-high { background: #d1fae5; color: #065f46; }
.badge-very-high { background: #d1fae5; color: #065f46; }
.badge-medium { background: #fef3c7; color: #92400e; }
.badge-low { background: #fee2e2; color: #991b1b; }

/* Stats row */
.stat-box {
	background: white;
	border-radius: 12px;
	padding: 1.2rem;
	text-align: center;
	box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.stat-number {
	font-size: 2rem;
	font-weight: 800;
	color: #6c63ff;
}
.stat-label {
	font-size: 0.8rem;
	color: #64748b;
	text-transform: uppercase;
	letter-spacing: 0.05em;
}

/* Evidence items */
.evidence-item {
	background: #f1f5f9;
	border-radius: 8px;
	padding: 0.6rem 1rem;
	margin: 0.3rem 0;
	font-size: 0.85rem;
	color: #334155;
	border-left: 3px solid #6c63ff;
}

/* Streaming box */
.stream-box {
	background: #0f172a;
	color: #a78bfa;
	border-radius: 12px;
	padding: 1rem 1.5rem;
	font-family: monospace;
	font-size: 0.85rem;
	min-height: 80px;
	border: 1px solid #1e293b;
}

/* Run button */
.stButton > button {
	background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
	color: white !important;
	border: none !important;
	border-radius: 10px !important;
	padding: 0.6rem 1.5rem !important;
	font-weight: 600 !important;
	width: 100% !important;
	transition: opacity 0.2s !important;
}
.stButton > button:hover {
	opacity: 0.9 !important;
}

/* Main area background */
[data-testid="stAppViewContainer"] {
	background: #f8fafc;
}
</style>
""",
	unsafe_allow_html=True,
)

st.markdown(
	"""
<div class="main-header">
	<h1>🩺 Clary — Health Pattern Detector</h1>
	<p>Cross-conversation temporal reasoning · Powered by GPT-4o</p>
</div>
""",
	unsafe_allow_html=True,
)

st.sidebar.markdown(
	"""
<div style="text-align:center; padding:1rem 0 1.5rem 0;">
	<div style="font-size:2.5rem;">🩺</div>
	<div style="font-size:1.1rem; font-weight:700; color:white;">AskFirst Clary</div>
	<div style="font-size:0.75rem; color:#a78bfa; margin-top:0.2rem;">Health Pattern Detection</div>
	<hr style="border-color:#4c3f8a; margin-top:1rem;">
</div>
""",
	unsafe_allow_html=True,
)

# Read key silently from environment
env_key = os.getenv("OPENAI_API_KEY", "")

if env_key:
	# Key found in .env — don't show it in UI at all
	api_key = env_key
	st.sidebar.success("✅ Good to Go")
else:
	# No .env key — ask user to paste it manually
	api_key = st.sidebar.text_input(
		"OpenAI API Key",
		type="password",
		placeholder="sk-...",
		help="Add OPENAI_API_KEY to your .env file to skip this step"
	)
	if not api_key:
		st.sidebar.warning("⚠️ ERROR ")

analysis_mode = st.sidebar.selectbox("Analysis Mode", ["Single User", "All Users"])

users_for_sidebar: list[dict] = []
dataset_load_error = ""
try:
	sidebar_dataset = load_dataset(DATA_PATH)
	users_for_sidebar = get_all_users(sidebar_dataset)
except FileNotFoundError:
	dataset_load_error = (
		f"Dataset file not found at '{DATA_PATH}'. Place your dataset file there and try again."
	)
except Exception as e:
	dataset_load_error = f"Unable to load dataset for user selection: {e}"

selected_user_name = ""
if analysis_mode == "Single User":
	user_names = [user.get("name", "Unknown") for user in users_for_sidebar]
	if user_names:
		selected_user_name = st.sidebar.selectbox("Select User", user_names)
	else:
		st.sidebar.info(
			"No users available yet. Ensure the dataset file exists and contains a users list."
		)

min_confidence_threshold = st.sidebar.number_input(
	"Min Confidence Threshold",
	min_value=0.0,
	max_value=1.0,
	value=0.4,
	step=0.05,
)

run_button = st.sidebar.button("🔍 Run Pattern Detection")

if run_button:
	try:
		if not api_key:
			st.error("⚠️ Please enter your OpenAI API key to continue")
			st.stop()

		st.info("⏳ Loading dataset...")
		try:
			dataset = load_dataset(DATA_PATH)
		except FileNotFoundError:
			st.error(
				f"Error: Dataset file is missing at '{DATA_PATH}'. Please add the JSON dataset file and rerun analysis."
			)
			st.stop()
		except Exception as e:
			st.error(f"Error: {e}")
			st.stop()

		st.success("✅ Dataset loaded. Starting analysis...")
		tab_patterns, tab_raw = st.tabs(["📊 Detected Patterns", "🔬 Raw JSON Output"])

		all_collected_patterns: list[dict] = []
		analyzed_user_names: list[str] = []
		final_report: dict = {}
		json_output = ""

		with tab_patterns:
			stream_placeholder = st.empty()
			cards_container = st.container()
			stream_buffer = ""

			with st.spinner("Clary is reasoning across sessions..."):
				if analysis_mode == "Single User":
					users = get_all_users(dataset)
					selected_user = None
					for user in users:
						if user.get("name") == selected_user_name:
							selected_user = user
							break

					if selected_user is None and selected_user_name:
						selected_user = get_user_by_id(dataset, selected_user_name)

					if selected_user is None:
						st.error("Error: Could not find the selected user in the dataset.")
						st.stop()

					analyzed_user_names = [selected_user.get("name", "Unknown")]

					if STREAM_ENABLED:
						event_stream = detect_patterns_for_user(
							selected_user,
							api_key=api_key,
							stream=True,
						)
						for item in event_stream:
							item_type = item.get("type", "")
							if item_type == "chunk":
								chunk_text = item.get("content", "")
								stream_buffer += chunk_text
								accumulated_text = escape(stream_buffer).replace("\n", "<br>")
								stream_placeholder.markdown(
									f'<div class="stream-box">🤖 Clary is reasoning...<br>{accumulated_text}</div>',
									unsafe_allow_html=True,
								)
								time.sleep(0.01)
							elif item_type == "result":
								all_collected_patterns.extend(item.get("patterns", []))
					else:
						all_collected_patterns = detect_patterns_for_user(
							selected_user,
							api_key=api_key,
							stream=False,
						)
				else:
					analyzed_user_names = [
						user.get("name", "Unknown") for user in get_all_users(dataset)
					]

					if STREAM_ENABLED:
						event_stream = detect_patterns_all_users(
							dataset,
							api_key=api_key,
							stream=True,
						)
						for item in event_stream:
							item_type = item.get("type", "")
							if item_type == "chunk":
								chunk_text = item.get("content", "")
								stream_buffer += chunk_text
								accumulated_text = escape(stream_buffer).replace("\n", "<br>")
								stream_placeholder.markdown(
									f'<div class="stream-box">🤖 Clary is reasoning...<br>{accumulated_text}</div>',
									unsafe_allow_html=True,
								)
								time.sleep(0.01)
							elif item_type == "result":
								all_collected_patterns.extend(item.get("patterns", []))
							elif item_type == "user_separator":
								st.markdown(f"### 👤 {item.get('user_name', 'Unknown')}")
					else:
						all_collected_patterns = detect_patterns_all_users(
							dataset,
							api_key=api_key,
							stream=False,
						)

			final_report = build_final_report(all_collected_patterns, analyzed_user_names)

			validated_and_normalized: list[dict] = []
			for pattern in all_collected_patterns:
				normalized = normalize_confidence_score(pattern)
				if validate_pattern(normalized):
					validated_and_normalized.append(normalized)

			filtered_raw = filter_patterns(
				validated_and_normalized,
				threshold=float(min_confidence_threshold),
			)
			filtered_ids = {
				(pattern.get("user_id", ""), pattern.get("pattern_id", ""))
				for pattern in filtered_raw
			}
			thresholded_patterns = [
				pattern
				for pattern in final_report.get("patterns", [])
				if (
					pattern.get("user_id", ""),
					pattern.get("pattern_id", ""),
				)
				in filtered_ids
			]
			final_report["patterns"] = thresholded_patterns
			final_report.setdefault("report_metadata", {})[
				"patterns_above_threshold"
			] = len(thresholded_patterns)

			with cards_container:
				st.success(
					f"✅ Analysis complete. Found {len(final_report.get('patterns', []))} patterns."
				)

				display_patterns = final_report.get("patterns", [])
				highest_confidence = "N/A"
				if display_patterns:
					highest_confidence = str(
						display_patterns[0].get("confidence", {}).get("level", "N/A")
					).upper()

				stat_col1, stat_col2, stat_col3 = st.columns(3)
				with stat_col1:
					st.markdown(
						f"""
<div class="stat-box">
	<div class="stat-number">{len(display_patterns)}</div>
	<div class="stat-label">Total Patterns Found</div>
</div>
""",
						unsafe_allow_html=True,
					)
				with stat_col2:
					st.markdown(
						f"""
<div class="stat-box">
	<div class="stat-number">{len(analyzed_user_names)}</div>
	<div class="stat-label">Users Analyzed</div>
</div>
""",
						unsafe_allow_html=True,
					)
				with stat_col3:
					st.markdown(
						f"""
<div class="stat-box">
	<div class="stat-number">{escape(highest_confidence)}</div>
	<div class="stat-label">Highest Confidence</div>
</div>
""",
						unsafe_allow_html=True,
					)

				for pattern in display_patterns:
					confidence = pattern.get("confidence", {})
					confidence_level = str(confidence.get("level", "")).lower().replace(" ", "-")
					if confidence_level not in {"very-high", "high", "medium", "low"}:
						confidence_level = "low"
					confidence_score = confidence.get("score", 0)

					st.markdown(
						f"""
<div class="pattern-card pattern-card-{confidence_level}">
	<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
		<strong style="font-size:1.05rem; color:#1e293b;">{escape(str(pattern.get("title", "Untitled Pattern")))}</strong>
		<span class="badge badge-{confidence_level}">{escape(str(confidence.get("level", "")))} · {escape(str(confidence_score))}</span>
	</div>
	<p style="color:#475569; font-size:0.9rem; margin-bottom:1rem;">{escape(str(pattern.get("description", "")))}</p>
	<div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem;">
		<div>
			<div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Cause</div>
			<div style="color:#1e293b; font-size:0.9rem;">{escape(str(pattern.get("cause", "")))}</div>
		</div>
		<div>
			<div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Effect</div>
			<div style="color:#1e293b; font-size:0.9rem;">{escape(str(pattern.get("effect", "")))}</div>
		</div>
		<div>
			<div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Temporal Gap</div>
			<div style="color:#1e293b; font-size:0.9rem;">{escape(str(pattern.get("temporal_gap", "—")))}</div>
		</div>
		<div>
			<div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Evidence Strength</div>
			<div style="color:#1e293b; font-size:0.9rem;">{escape(str(pattern.get("evidence_strength", "—")))} sessions</div>
		</div>
	</div>
	<div style="font-size:0.75rem; color:#6366f1; font-style:italic; border-top:1px solid #f1f5f9; padding-top:0.8rem;">
		💡 {escape(str(confidence.get("justification", "No confidence justification provided.")))}
	</div>
</div>
""",
						unsafe_allow_html=True,
					)

					with st.expander("🔬 Temporal Reasoning & Evidence"):
						st.markdown("**Temporal Reasoning:**")
						st.write(pattern.get("temporal_reasoning", ""))

						evidence_items = pattern.get("evidence", [])
						for item in evidence_items:
							st.markdown(
								f'<div class="evidence-item">📍 {escape(str(item))}</div>',
								unsafe_allow_html=True,
							)

						if pattern.get("biological_mechanism"):
							st.markdown(
								f"**🧬 Biological Mechanism:** {pattern.get('biological_mechanism', '')}"
							)

						progressive_stages = pattern.get("progressive_stages", None)
						if progressive_stages is not None:
							st.markdown("**Progressive Stages:**")
							for idx, stage in enumerate(progressive_stages, start=1):
								st.markdown(f"{idx}. {stage}")

				with st.expander("🔎 Reasoning Trace"):
					st.write(final_report.get("reasoning_trace", ""))

			json_output = patterns_to_json_string(final_report, indent=2)

		with tab_raw:
			if json_output:
				st.code(json_output, language="json")
				st.download_button(
					label="Download JSON",
					data=json_output,
					file_name="clary_patterns.json",
					mime="application/json",
				)

	except Exception as e:
		st.error(f"Error: {e}")
elif dataset_load_error:
	st.error(dataset_load_error)

st.markdown(
	"""
<div style="text-align:center; padding:2rem 0 1rem 0; color:#94a3b8; font-size:0.8rem;">
	Built for <strong>AskFirst AI Intern Assignment</strong> · 
	Powered by <strong>GPT-4o</strong> · 
	Temporal reasoning across 27 health conversations
</div>
""",
	unsafe_allow_html=True,
)
