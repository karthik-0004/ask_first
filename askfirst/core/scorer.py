"""Confidence scoring and final report formatting for AskFirst patterns."""

import json
from datetime import datetime

import config


def validate_pattern(pattern: dict) -> bool:
	"""Validate that a pattern contains required fields and value shapes.

	Args:
		pattern: Raw pattern dictionary returned by the pattern engine.

	Returns:
		True when the pattern is valid for downstream processing, else False.
	"""
	required_keys = {
		"pattern_id",
		"title",
		"description",
		"sessions_involved",
		"confidence",
		"confidence_score",
		"confidence_justification",
	}

	if not all(key in pattern for key in required_keys):
		return False

	score = pattern.get("confidence_score")
	if not isinstance(score, (int, float)):
		return False
	if not 0.0 <= float(score) <= 1.0:
		return False

	sessions = pattern.get("sessions_involved")
	if not isinstance(sessions, list) or len(sessions) == 0:
		return False

	return True


def normalize_confidence_score(pattern: dict) -> dict:
	"""Normalize confidence_score from confidence label when missing or zero.

	Args:
		pattern: Raw pattern dictionary to normalize.

	Returns:
		The updated pattern dictionary with confidence_score set.
	"""
	confidence_map = {
		"very high": 0.92,
		"high": 0.78,
		"medium": 0.58,
		"low": 0.35,
	}

	current_score = pattern.get("confidence_score")
	if current_score is None or current_score == 0:
		level = str(pattern.get("confidence", "")).strip().lower()
		pattern["confidence_score"] = confidence_map.get(level, 0.0)

	return pattern


def filter_patterns(patterns: list[dict], threshold: float = None) -> list[dict]:
	"""Filter patterns by minimum confidence and sort descending by score.

	Args:
		patterns: List of normalized and validated pattern dictionaries.
		threshold: Optional score threshold override. Uses config default when None.

	Returns:
		Filtered list sorted by confidence_score in descending order.
	"""
	min_threshold = config.MIN_CONFIDENCE_THRESHOLD if threshold is None else threshold
	filtered = [
		pattern for pattern in patterns if float(pattern.get("confidence_score", 0.0)) >= min_threshold
	]
	return sorted(filtered, key=lambda p: float(p.get("confidence_score", 0.0)), reverse=True)


def format_pattern_output(pattern: dict) -> dict:
	"""Format one raw pattern into the final structured output schema.

	Args:
		pattern: Normalized pattern dictionary.

	Returns:
		Final output dictionary containing the assignment-required fields.
	"""
	return {
		"pattern_id": pattern["pattern_id"],
		"user_id": pattern["user_id"],
		"user_name": pattern["user_name"],
		"title": pattern["title"],
		"description": pattern["description"],
		"cause": pattern.get("cause", ""),
		"effect": pattern.get("effect", ""),
		"temporal_gap": pattern.get("temporal_gap", "not specified"),
		"temporal_reasoning": pattern.get("temporal_reasoning", ""),
		"evidence": pattern.get("evidence", []),
		"progressive_stages": pattern.get("progressive_stages", None),
		"biological_mechanism": pattern.get("biological_mechanism", ""),
		"sessions_involved": pattern["sessions_involved"],
		"timestamps_involved": pattern.get("timestamps_involved", []),
		"evidence_strength": pattern.get("evidence_strength", len(pattern["sessions_involved"])),
		"confidence": {
			"level": pattern["confidence"],
			"score": round(float(pattern["confidence_score"]), 2),
			"justification": pattern["confidence_justification"],
		},
	}


def build_final_report(all_patterns: list[dict], user_names: list[str]) -> dict:
	"""Build the final report after validating, normalizing, filtering, and formatting.

	Args:
		all_patterns: Raw pattern dictionaries across analyzed users.
		user_names: Names of users included in analysis.

	Returns:
		Final report dictionary with metadata, patterns, and reasoning trace.
	"""
	validated_and_normalized: list[dict] = []
	for pattern in all_patterns:
		normalized = normalize_confidence_score(pattern)
		if validate_pattern(normalized):
			validated_and_normalized.append(normalized)

	filtered_patterns = filter_patterns(validated_and_normalized)
	formatted_patterns = [format_pattern_output(pattern) for pattern in filtered_patterns]

	return {
		"report_metadata": {
			"generated_at": datetime.now().isoformat(),
			"users_analyzed": user_names,
			"total_patterns_found": len(validated_and_normalized),
			"patterns_above_threshold": len(formatted_patterns),
			"model_used": config.MODEL_NAME,
		},
		"patterns": formatted_patterns,
		"reasoning_trace": "Patterns were detected by analyzing full conversation history with temporal gap reasoning. GPT-4o was instructed to identify cause-effect chains across sessions with at least 2 sessions of evidence per pattern.",
	}


def patterns_to_json_string(report: dict, indent: int = 2) -> str:
	"""Serialize the final report dictionary into pretty-printed JSON.

	Args:
		report: Final report dictionary.
		indent: Indentation level for pretty formatting.

	Returns:
		JSON string representation of the report.
	"""
	return json.dumps(report, indent=indent)
