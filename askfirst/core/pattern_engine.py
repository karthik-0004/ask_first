"""Core LLM-powered temporal pattern detection engine for AskFirst."""

import json
import re

import openai

import config
from core.context_builder import (
	build_chunked_context,
	build_user_profile_header,
	get_temporal_summary,
)
from core.loader import get_user_sessions


PATTERN_DETECTION_SYSTEM_PROMPT = """
You are Clary, an expert medical reasoning AI specialized in detecting hidden 
health patterns across multiple conversations over time.

Your job is to read a user's FULL conversation history and find ALL repeated 
health patterns — even subtle ones. You must reason about TIME, not just keywords.

CRITICAL RULES:
1. You MUST return at least 2-3 patterns if ANY repeated signals exist in the data.
	 Do NOT return an empty array unless the history has only 1 session.
2. A pattern needs at least 2 supporting sessions as evidence.
3. You MUST explicitly calculate the time gap between cause and effect.
4. Look for: repeated symptoms after lifestyle triggers, delayed reactions 
	 (weeks after a change), symptoms that appear only in specific contexts 
	 (deadlines, stress, diet changes).
5. Do NOT be conservative. A medium-confidence pattern is still worth reporting.
6. Never return [] for a user with 8+ sessions. There are always patterns.

WHAT TO LOOK FOR (non-exhaustive):
- Same symptom appearing multiple times across weeks/months
- Lifestyle change followed by symptom weeks later (temporal delay)
- Symptom that disappears when trigger is removed (confirmation)
- Symptom that worsens when trigger increases (dose-response)
- Multiple symptoms from one root cause (branching effect)

OUTPUT FORMAT:
Return ONLY a valid JSON array. No explanation outside the JSON. No markdown. 
No code fences. Just the raw JSON array starting with [ and ending with ].

Each object in the array must have EXACTLY these fields:
[
  {
		"pattern_id": "P1",
		"user_id": "USR001",
		"user_name": "Arjun",
		"title": "short descriptive title of the pattern",
		"description": "2-3 sentences explaining the pattern with specific session references and dates",
		"cause": "what triggers this pattern",
		"effect": "what symptom or outcome results",
		"temporal_gap": "how much time between cause and effect e.g. within hours, 6 weeks later, same day",
		"biological_mechanism": "brief medical or behavioral reason why this connection makes sense",
		"sessions_involved": ["USR001_S01", "USR001_S04"],
		"timestamps_involved": ["Jan 05 2026", "Jan 28 2026"],
		"evidence_strength": 3,
		"confidence": "high",
		"confidence_score": 0.88,
		"confidence_justification": "One sentence: why this score, citing specific session evidence"
  }
]
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



def safe_parse_json(raw: str, user: dict) -> list:
	"""Try multiple strategies to extract valid JSON from LLM response."""
	# Strategy 1: direct parse
	try:
		return json.loads(raw)
	except json.JSONDecodeError:
		pass

	# Strategy 2: extract JSON array between first [ and last ]
	try:
		start = raw.index("[")
		end = raw.rindex("]") + 1
		return json.loads(raw[start:end])
	except (ValueError, json.JSONDecodeError):
		pass

	# Strategy 3: strip markdown code fences if model added them
	try:
		cleaned = re.sub(r"```json|```", "", raw).strip()
		return json.loads(cleaned)
	except json.JSONDecodeError:
		pass

	# Strategy 4: return error pattern so app doesnt crash silently
	print(f"PARSE FAILED for {user['name']}. Raw response was:\n{raw}")
	return [{
		"pattern_id": "PARSE_ERROR",
		"user_id": user["user_id"],
		"user_name": user["name"],
		"title": "JSON Parse Error",
		"description": f"LLM returned non-JSON response. Raw: {raw[:300]}",
		"cause": "system error",
		"effect": "no patterns extracted",
		"temporal_gap": "N/A",
		"biological_mechanism": "N/A",
		"sessions_involved": [],
		"timestamps_involved": [],
		"evidence_strength": 0,
		"confidence": "low",
		"confidence_score": 0.0,
		"confidence_justification": "Parse failed"
	}]


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
	_ = build_user_profile_header(user)

	user_prompt = f"""
You are analyzing the health conversation history of {user['name']} (ID: {user['user_id']}).
Age: {user['age']}, Occupation: {user['occupation']}.
Background: {user['onboarding_notes']}

TEMPORAL TIMELINE (use this to reason about sequence and gaps):
{temporal_timeline}

FULL CONVERSATION HISTORY (all sessions in chronological order):
{full_context}

TASK:
1. Read ALL sessions carefully from start to end.
2. Identify every repeated pattern — symptoms that recur, causes that keep appearing, 
   lifestyle factors that consistently precede health events.
3. For each pattern, state the exact sessions and dates involved.
4. Calculate the time gap between cause and effect explicitly.
5. Return your findings as a JSON array.

IMPORTANT: This user has {len(sessions)} sessions over 3 months. 
There are definitely patterns here. Do not return an empty array.
Return ALL patterns you find, even medium confidence ones.

Return ONLY the JSON array. Nothing else.
""".strip()

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

				raw_response_text = "".join(collected_parts).strip()
				print("=" * 60)
				print(f"RAW LLM RESPONSE FOR {user['name']}:")
				print("=" * 60)
				print(raw_response_text)
				print("=" * 60)
				parsed_list = safe_parse_json(raw_response_text, user)
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
		raw_response_text = (response.choices[0].message.content or "").strip()
		print("=" * 60)
		print(f"RAW LLM RESPONSE FOR {user['name']}:")
		print("=" * 60)
		print(raw_response_text)
		print("=" * 60)
		return safe_parse_json(raw_response_text, user)
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
