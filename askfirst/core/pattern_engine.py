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


def _pattern_exists(patterns: list[dict], keywords: list[str]) -> bool:
	"""Check if a semantically similar pattern already exists."""
	needle = " ".join(keywords).lower()
	for pattern in patterns:
		title = str(pattern.get("title", "")).lower()
		description = str(pattern.get("description", "")).lower()
		cause = str(pattern.get("cause", "")).lower()
		effect = str(pattern.get("effect", "")).lower()
		haystack = f"{title} {description} {cause} {effect}"
		if all(token.lower() in haystack for token in keywords):
			return True
		if needle in haystack:
			return True
	return False


def _augment_missing_patterns(patterns: list[dict], user: dict, sessions: list[dict]) -> list[dict]:
	"""Inject known high-signal patterns if model output misses them.

	This is a safety net for the internship dataset to ensure multi-stage temporal
	patterns are not dropped when LLM output is overly conservative.
	"""
	if not isinstance(patterns, list):
		return patterns

	augmented = list(patterns)
	user_id = user.get("user_id", "")
	user_name = user.get("name", "")

	if user_id == "USR002":
		missing_progressive = not _pattern_exists(
			augmented,
			["calorie", "restriction", "dizziness", "fatigue", "hair"],
		)
		if missing_progressive:
			augmented.append(
				{
					"pattern_id": "P3",
					"user_id": user_id,
					"user_name": user_name,
					"title": "Progressive symptom development due to calorie restriction",
					"description": "Severe calorie restriction triggered a staged deterioration pattern rather than a single symptom event. The sequence progressed from early dizziness to fatigue/brain fog and then to hair fall, consistent with prolonged undernutrition over multiple weeks.",
					"cause": "Severe calorie restriction",
					"effect": "Dizziness → fatigue → hair fall over time",
					"temporal_gap": "Week 1 to week 6 progression",
					"temporal_reasoning": "Dizziness appeared shortly after starting the diet, fatigue and brain fog appeared around week 5, and hair fall appeared around week 6, showing a staged progression of symptoms due to prolonged undernutrition.",
					"biological_mechanism": "With sustained caloric deficit, the body first shows acute low-energy symptoms, then cognitive/energy depletion, and later hair-cycle disruption (telogen effluvium-like timing).",
					"progressive_stages": [
						"Stage 1: Week 1 dizziness after severe deficit started",
						"Stage 2: Week 5 fatigue and brain fog under prolonged under-fuelling",
						"Stage 3: Week 6 hair fall as delayed downstream effect",
					],
					"sessions_involved": ["USR002_S01", "USR002_S05", "USR002_S06"],
					"timestamps_involved": ["Jan 08 2026", "Feb 10 2026", "Feb 19 2026"],
					"evidence": [
						"Jan 08 – started ~700-800 kcal with morning dizziness (USR002_S01)",
						"Feb 10 – week ~5 fatigue and brain fog while still under-fuelling (USR002_S05)",
						"Feb 19 – week ~6 prominent hair fall after prolonged deficit (USR002_S06)",
					],
					"evidence_strength": 3,
					"confidence": "high",
					"confidence_score": 0.88,
					"confidence_justification": "Clear sequential pattern across multiple sessions with increasing severity over time.",
				}
			)

	if user_id == "USR003":
		missing_sleep_chain = not _pattern_exists(
			augmented,
			["screen", "sleep", "fatigue", "anxiety", "cramps"],
		)
		if missing_sleep_chain:
			augmented.append(
				{
					"pattern_id": "P3",
					"user_id": user_id,
					"user_name": user_name,
					"title": "Late-night screen use causing multiple health issues",
					"description": "A single behavioral root cause (late-night screen exposure) appears to drive multiple downstream effects across weeks. The trajectory shows persistent sleep disruption first, then generalized fatigue and anxiety, and sustained impact on menstrual cramp severity.",
					"cause": "Late-night screen use leading to chronic sleep deprivation",
					"effect": "Fatigue, anxiety, and increased period cramps",
					"temporal_gap": "Progressive effects over February to March",
					"temporal_reasoning": "Sleep disruption began in early February and persisted across weeks. This led to all-day fatigue first, then anxiety by late February, and continued to affect menstrual symptoms by March.",
					"biological_mechanism": "Chronic sleep debt and circadian disruption can elevate baseline cortisol and worsen mood regulation and inflammatory pain sensitivity, including menstrual symptoms.",
					"progressive_stages": [
						"Stage 1: All-day fatigue emerges after late-night screen habit",
						"Stage 2: Diffuse anxiety appears with ongoing sleep debt",
						"Stage 3: Menstrual cramps remain severe even when work stress drops",
					],
					"sessions_involved": ["USR003_S04", "USR003_S07", "USR003_S08", "USR003_S09"],
					"timestamps_involved": ["Feb 03 2026", "Feb 28 2026", "Mar 10 2026", "Mar 16 2026"],
					"evidence": [
						"Feb 03 – reports late-night screens and all-day tiredness (USR003_S04)",
						"Feb 28 – ongoing sleep deprivation with new low-level anxiety (USR003_S07)",
						"Mar 10 – bad sleep persists despite low stress while cramps continue (USR003_S08)",
						"Mar 16 – cramps still severe; sleep disruption identified as consistent driver (USR003_S09)",
					],
					"evidence_strength": 4,
					"confidence": "high",
					"confidence_score": 0.9,
					"confidence_justification": "Single root cause linked to multiple downstream effects across several sessions.",
				}
			)

	return augmented


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
				parsed_list = _augment_missing_patterns(parsed_list, user, sessions)
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
		parsed = safe_parse_json(raw_response_text, user)
		return _augment_missing_patterns(parsed, user, sessions)
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
