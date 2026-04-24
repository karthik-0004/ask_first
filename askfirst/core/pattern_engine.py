"""Core LLM-powered temporal pattern detection engine for AskFirst."""

import json

import openai

import config
from core.context_builder import (
	build_chunked_context,
	build_user_profile_header,
	get_temporal_summary,
)
from core.loader import get_user_sessions


PATTERN_DETECTION_SYSTEM_PROMPT = """
You are Clary, an advanced health pattern analysis AI. Your job is to analyze a user's complete health conversation history and detect hidden patterns that the user themselves has not noticed.

You reason across TIME, not just keywords. A symptom appearing 8 weeks after a lifestyle change is medically different from a symptom appearing before one. Direction of time matters.

RULES:
1. Only report patterns that are supported by at least 2 sessions of evidence. Do not speculate on single-session mentions.
2. Always state the SPECIFIC session IDs and timestamps that form your evidence chain.
3. Always reason about the temporal gap between cause and effect. State the gap explicitly.
4. Distinguish between correlation and causation. If there is a plausible biological or behavioral mechanism, state it.
5. Do not restate what Clary already told the user. Find what Clary MISSED or what emerged across multiple sessions.
6. Output ONLY a valid JSON array. No preamble, no explanation outside the JSON.

OUTPUT FORMAT — return a JSON array of pattern objects, each with exactly these fields:
[
  {
	"pattern_id": "auto-generated string like P1, P2...",
	"user_id": "the user's ID",
	"user_name": "the user's name",
	"title": "short title of the pattern",
	"description": "2-3 sentence description of the pattern with temporal reasoning",
	"sessions_involved": ["session_id_1", "session_id_2", ...],
	"timestamps_involved": ["readable timestamp 1", "readable timestamp 2", ...],
	"cause": "what appears to be causing this",
	"effect": "what symptom or outcome results",
	"temporal_gap": "how much time passes between cause and effect, if applicable",
	"biological_mechanism": "brief explanation of WHY this connection is plausible medically or behaviorally",
	"confidence": "very high | high | medium | low",
	"confidence_score": 0.0 to 1.0 as a float,
	"confidence_justification": "one sentence: why this confidence level, citing session evidence",
	"evidence_strength": "how many independent sessions confirm this pattern as an integer"
  }
]

If no patterns are found, return an empty array [].
"""


def _build_parse_error_pattern(user: dict, raw_response: str) -> list[dict]:
	"""Create a standardized parse-error pattern payload.

	Args:
		user: User dictionary used for context and metadata.
		raw_response: Raw model text that failed JSON parsing.

	Returns:
		A single-item list containing a parse error pattern dictionary.
	"""
	return [
		{
			"pattern_id": "PARSE_ERROR",
			"user_id": user.get("user_id", ""),
			"user_name": user.get("name", ""),
			"title": "Parse Error",
			"description": raw_response,
			"sessions_involved": [],
			"timestamps_involved": [],
			"cause": "",
			"effect": "",
			"temporal_gap": "",
			"biological_mechanism": "",
			"confidence": "low",
			"confidence_score": 0.0,
			"confidence_justification": "Model output was not valid JSON.",
			"evidence_strength": 0,
		}
	]


def _parse_patterns_response(user: dict, raw_response: str) -> list[dict]:
	"""Parse model output JSON into a list of pattern dictionaries.

	Args:
		user: User dictionary used for fallback parse-error metadata.
		raw_response: Raw model response text expected to be JSON array.

	Returns:
		Parsed list of pattern dictionaries, or a parse-error pattern list.
	"""
	try:
		parsed = json.loads(raw_response)
		if isinstance(parsed, list):
			return parsed
		return _build_parse_error_pattern(
			user,
			f"Expected JSON array but got {type(parsed).__name__}: {raw_response}",
		)
	except Exception:
		return _build_parse_error_pattern(user, raw_response)


def detect_patterns_for_user(user: dict, api_key: str, stream: bool = True):
	"""Detect temporal health patterns for a single user via GPT-4o.

	Args:
		user: User dictionary with profile and conversation history.
		api_key: OpenAI API key used for this invocation.
		stream: Whether to stream partial tokens and final parsed result.

	Returns:
		If stream is False, returns a list of pattern dictionaries.
		If stream is True, returns a generator yielding chunk/result dictionaries.
	"""
	resolved_api_key = api_key or config.OPENAI_API_KEY
	if not resolved_api_key:
		print("Error: Missing OpenAI API key. Provide api_key or set OPENAI_API_KEY.")
		missing_key_error = [
			{
				"pattern_id": "API_KEY_ERROR",
				"user_id": user.get("user_id", ""),
				"user_name": user.get("name", ""),
				"title": "Parse Error",
				"description": "Missing OpenAI API key.",
				"sessions_involved": [],
				"timestamps_involved": [],
				"cause": "",
				"effect": "",
				"temporal_gap": "",
				"biological_mechanism": "",
				"confidence": "low",
				"confidence_score": 0.0,
				"confidence_justification": "OpenAI call could not be executed.",
				"evidence_strength": 0,
			}
		]
		if stream:
			def _missing_key_stream():
				yield {"type": "result", "patterns": missing_key_error}

			return _missing_key_stream()
		return missing_key_error

	full_context = build_chunked_context(user)
	sessions = get_user_sessions(user)
	temporal_timeline = get_temporal_summary(sessions)
	profile_header = build_user_profile_header(user)

	user_prompt = f"""
Analyze the following user's complete health conversation history and detect ALL hidden patterns with temporal reasoning.

{temporal_timeline}

FULL CONVERSATION HISTORY:
{full_context}

Remember: Output ONLY a valid JSON array of pattern objects. Do not write anything outside the JSON array.
""".strip()

	# Keep profile explicitly available in the prompt payload for robust grounding.
	user_prompt = f"{profile_header}\n\n{user_prompt}"

	client = openai.OpenAI(api_key=resolved_api_key)

	if stream:
		def _stream_detection():
			"""Yield streamed text chunks followed by final parsed pattern result."""
			collected_parts: list[str] = []
			try:
				stream_response = client.chat.completions.create(
					model=config.MODEL_NAME,
					messages=[
						{"role": "system", "content": PATTERN_DETECTION_SYSTEM_PROMPT},
						{"role": "user", "content": user_prompt},
					],
					max_tokens=config.MAX_TOKENS,
					temperature=config.TEMPERATURE,
					stream=True,
				)

				for chunk in stream_response:
					delta = chunk.choices[0].delta.content if chunk.choices else None
					if delta:
						collected_parts.append(delta)
						yield {"type": "chunk", "content": delta}

				raw_response = "".join(collected_parts).strip()
				parsed_list = _parse_patterns_response(user, raw_response)
				yield {"type": "result", "patterns": parsed_list}
			except Exception as exc:
				print(f"Error: OpenAI streaming call failed for user {user.get('name', 'unknown')}: {exc}")
				error_patterns = _build_parse_error_pattern(
					user,
					f"OpenAI streaming call failed: {exc}",
				)
				yield {"type": "result", "patterns": error_patterns}

		return _stream_detection()

	try:
		response = client.chat.completions.create(
			model=config.MODEL_NAME,
			messages=[
				{"role": "system", "content": PATTERN_DETECTION_SYSTEM_PROMPT},
				{"role": "user", "content": user_prompt},
			],
			max_tokens=config.MAX_TOKENS,
			temperature=config.TEMPERATURE,
			stream=False,
		)
		raw_response = (response.choices[0].message.content or "").strip()
		return _parse_patterns_response(user, raw_response)
	except Exception as exc:
		print(f"Error: OpenAI call failed for user {user.get('name', 'unknown')}: {exc}")
		return _build_parse_error_pattern(user, f"OpenAI call failed: {exc}")


def detect_patterns_all_users(dataset: dict, api_key: str, stream: bool = True):
	"""Detect temporal patterns for all users in the dataset.

	Args:
		dataset: Parsed dataset containing a top-level users list.
		api_key: OpenAI API key used for each user detection call.
		stream: Whether to stream chunk/result events user-by-user.

	Returns:
		If stream is False, returns a flat list of all users' pattern dictionaries.
		If stream is True, returns a generator yielding user events and separators.
	"""
	users = dataset.get("users", [])

	if stream:
		def _stream_all_users():
			"""Yield streaming events for each user in sequence."""
			for index, user in enumerate(users):
				user_name = user.get("name", "Unknown")
				user_stream = detect_patterns_for_user(user, api_key=api_key, stream=True)
				for item in user_stream:
					yield item

				if index < len(users) - 1:
					yield {"type": "user_separator", "user_name": user_name}

		return _stream_all_users()

	all_patterns: list[dict] = []
	for user in users:
		user_patterns = detect_patterns_for_user(user, api_key=api_key, stream=False)
		if isinstance(user_patterns, list):
			all_patterns.extend(user_patterns)
	return all_patterns
