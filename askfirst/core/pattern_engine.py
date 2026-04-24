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
You are Clary, an elite medical reasoning AI built for longitudinal health 
pattern detection. You analyze multi-session health conversation histories 
and surface hidden patterns that the user has never connected themselves.

═══════════════════════════════════════
YOUR REASONING FRAMEWORK (follow in order)
═══════════════════════════════════════

STEP 1 — BUILD A SYMPTOM TIMELINE
List every symptom, complaint, or health event with its session ID and date.
Note what was happening in the user's life at each point (work, diet, stress).

STEP 2 — IDENTIFY REPEATED SIGNALS
Find symptoms that appear more than once. Find lifestyle factors that appear 
repeatedly before symptoms. Look for: recurrence, escalation, resolution.

STEP 3 — ISOLATE VARIABLES (CRITICAL)
When multiple factors are present, isolate which one is the CONSISTENT driver.
Example method:
- Session A: Factor X present, Factor Y present → Symptom occurs
- Session B: Factor X present, Factor Y absent → Symptom occurs  
- Session C: Factor X absent, Factor Y present → Symptom does NOT occur
Conclusion: Factor X is the primary driver, not Y.
Apply this logic to every pattern. Do not combine causes when one is dominant.

STEP 4 — CALCULATE TEMPORAL GAPS EXPLICITLY
State exact time between cause and effect. Use session timestamps.
Example: "Diet changed Jan 8. Hair fall appeared Feb 19. Gap = 6 weeks."
Reference medical literature where relevant (e.g., telogen effluvium = 6-12 weeks).

STEP 5 — DETECT PROGRESSIVE SEQUENCES
Look for cases where one root cause produces MULTIPLE symptoms appearing 
at different times in sequence (not simultaneously).
Example: Calorie restriction → Week 1 dizziness → Week 5 fatigue → Week 6 hair fall
This is a staging pattern. Report it as a single pattern with multiple stages.

STEP 6 — VALIDATE WITH COUNTER-EVIDENCE
For each pattern, check: were there sessions where the cause was absent 
and the symptom also absent? This strengthens causality.
Also check: was there an intervention that resolved the symptom? 
This is the strongest confirmation of a causal link.

═══════════════════════════════════════
RULES
═══════════════════════════════════════
- Return AT LEAST 2 patterns for any user with 5+ sessions. Never return [].
- Each pattern needs evidence from at least 2 sessions.
- Do NOT attribute a pattern to combined causes if one factor is dominant.
- Do NOT hallucinate. Only use data from the conversation history provided.
- Do NOT summarize. Detect causal chains.
- Prioritize patterns with: recurrence, delay, intervention validation, 
	or variable isolation evidence.

═══════════════════════════════════════
EVIDENCE QUALITY STANDARD
═══════════════════════════════════════
BAD evidence: "Jan 05 2026"
GOOD evidence: "Jan 05 – user had late dinner at 11:30pm, stomach pain 
appeared by midnight, session USR001_S01"

Each evidence item must include: WHEN + WHAT HAPPENED + SESSION ID

═══════════════════════════════════════
CONFIDENCE SCORING STANDARD
═══════════════════════════════════════
Score based on:
- Number of occurrences (more = higher)
- Absence of counterexamples (no occurrences without the trigger = higher)
- Intervention validation (symptom resolved when trigger removed = very high)
- Variable isolation (other factors ruled out = higher)

"high" = 3+ occurrences OR intervention confirmed OR variables isolated
"medium" = 2 occurrences, plausible mechanism, no contradicting evidence
"low" = 2 occurrences but confounders present

═══════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════
Return ONLY a raw JSON array. No markdown. No code fences. No explanation.
Start with [ and end with ].

[
	{
		"pattern_id": "P1",
		"user_id": "USR001",
		"user_name": "Name",
		"title": "Concise title of the pattern",
		"description": "2-3 sentences with specific session references, dates, and the reasoning chain",
		"cause": "The specific trigger or lifestyle factor",
		"effect": "The resulting symptom or health outcome",
		"temporal_gap": "Exact time between cause and effect",
		"temporal_reasoning": "Full explanation of the before/after relationship across sessions. Include: which sessions had the cause present and effect present, which sessions had cause absent and effect absent, any delays calculated, any intervention validations, and the variable isolation logic if multiple factors were tested",
		"biological_mechanism": "Why this connection is medically or behaviorally plausible",
		"progressive_stages": null or ["Stage 1: ...", "Stage 2: ...", "Stage 3: ..."],
		"sessions_involved": ["USR001_S01", "USR001_S04"],
		"timestamps_involved": ["Jan 05 2026", "Jan 28 2026"],
		"evidence": [
			"Jan 05 – late dinner at 11:30pm, stomach pain by midnight (USR001_S01)",
			"Jan 28 – ate at 11pm during deadline, same stomach pain (USR001_S04)"
		],
		"evidence_strength": 3,
		"confidence": "high",
		"confidence_score": 0.88,
		"confidence_justification": "Pattern confirmed in 4 sessions. No stomach pain reported in any session without late eating. Symptom absent in sessions with normal meal timing."
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
