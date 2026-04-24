from dotenv import load_dotenv
load_dotenv()

import json
import os
import time

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

st.header("🩺 Clary - Health Pattern Detector")
st.subheader("Cross-conversation temporal reasoning for hidden health patterns")
st.divider()

st.sidebar.title("🩺 AskFirst Clary")
st.sidebar.caption("Health Pattern Detection")

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
								stream_placeholder.text_area(
									"Streaming Output",
									value=stream_buffer,
									height=260,
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
								stream_placeholder.text_area(
									"Streaming Output",
									value=stream_buffer,
									height=260,
								)
								time.sleep(0.01)
							elif item_type == "result":
								all_collected_patterns.extend(item.get("patterns", []))
							elif item_type == "user_separator":
								st.subheader(f"👤 {item.get('user_name', 'Unknown')}")
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

				for pattern in final_report.get("patterns", []):
					confidence = pattern.get("confidence", {})
					confidence_level = str(confidence.get("level", "")).lower()
					confidence_score = confidence.get("score", 0)

					expander_title = (
						f"{pattern.get('title', 'Untitled Pattern')} — "
						f"{str(confidence.get('level', '')).upper()} confidence "
						f"({confidence_score})"
					)

					with st.expander(expander_title, expanded=False):
						st.markdown(pattern.get("description", ""))

						left_col, right_col = st.columns(2)

						with left_col:
							st.markdown(f"**Cause:** {pattern.get('cause', '')}")
							st.markdown(f"**Effect:** {pattern.get('effect', '')}")
							st.markdown(
								f"**Temporal Gap:** {pattern.get('temporal_gap', 'not specified')}"
							)
							st.markdown(
								"**Biological Mechanism:** "
								f"{pattern.get('biological_mechanism', '')}"
							)

						with right_col:
							st.markdown("**Sessions Involved:**")
							sessions = pattern.get("sessions_involved", [])
							if sessions:
								st.markdown("\n".join(f"- {session_id}" for session_id in sessions))
							else:
								st.markdown("- None")
							st.markdown(
								f"**Evidence Strength:** {pattern.get('evidence_strength', 0)}"
							)

						if confidence_level in {"very high", "high"}:
							badge_color = "#1B8A3B"
						elif confidence_level == "medium":
							badge_color = "#B08800"
						else:
							badge_color = "#B42318"

						st.markdown(
							(
								"<div style=\"display:inline-block;padding:6px 10px;"
								f"border-radius:8px;background-color:{badge_color};"
								"color:white;font-weight:600;\">"
								f"Confidence: {str(confidence.get('level', '')).upper()} "
								f"({confidence_score})"
								"</div>"
							),
							unsafe_allow_html=True,
						)
						st.markdown(
							f"*{confidence.get('justification', 'No confidence justification provided.')}*"
						)

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

st.markdown("---\n*Built for AskFirst AI Intern Assignment | Powered by GPT-4o*")
