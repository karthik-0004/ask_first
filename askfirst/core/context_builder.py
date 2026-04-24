"""Context assembly utilities for AskFirst LLM pattern analysis."""

from core.loader import get_user_sessions, summarize_session


def build_full_user_context(user: dict) -> str:
	"""Build the full context block for one user including all sessions.

	Args:
		user: User dictionary containing profile fields and conversations.

	Returns:
		A single formatted string with profile header followed by all sessions.
	"""
	profile_header = build_user_profile_header(user)
	session_blocks = [summarize_session(session) for session in get_user_sessions(user)]

	if not session_blocks:
		return profile_header

	return f"{profile_header}\n\n" + "\n\n".join(session_blocks)


def build_user_profile_header(user: dict) -> str:
	"""Build the user profile header block without any sessions.

	Args:
		user: User dictionary containing demographic and onboarding fields.

	Returns:
		A formatted user profile header string.
	"""
	name = user.get("name", "")
	age = user.get("age", "")
	gender = user.get("gender", "")
	occupation = user.get("occupation", "")
	location = user.get("location", "")
	onboarding_notes = user.get("onboarding_notes", "")

	return (
		"USER PROFILE\n"
		f"Name: {name} | Age: {age} | Gender: {gender}\n"
		f"Occupation: {occupation} | Location: {location}\n"
		f"Background: {onboarding_notes}"
	)


def build_chunked_context(user: dict, max_sessions: int = 10) -> str:
	"""Build a context block capped to the most recent sessions.

	Args:
		user: User dictionary containing profile and conversation history.
		max_sessions: Maximum number of most-recent sessions to include.

	Returns:
		A formatted context string with profile header and selected sessions.
	"""
	profile_header = build_user_profile_header(user)
	sessions = get_user_sessions(user)
	selected_sessions = sessions[-max_sessions:] if len(sessions) > max_sessions else sessions
	session_blocks = [summarize_session(session) for session in selected_sessions]

	if not session_blocks:
		return profile_header

	return f"{profile_header}\n\n" + "\n\n".join(session_blocks)


def get_temporal_summary(sessions: list[dict]) -> str:
	"""Build a compact temporal timeline summary for session history.

	Args:
		sessions: List of session dictionaries.

	Returns:
		A timeline string with one line per session showing date and tags.
	"""
	lines = ["TEMPORAL TIMELINE:"]

	for session in sessions:
		session_id = session.get("session_id", "")
		timestamp = session.get("timestamp", "")
		date_only = timestamp.split("T", maxsplit=1)[0]
		date_display = date_only
		if date_only:
			year, month, day = date_only.split("-")
			month_names = {
				"01": "Jan",
				"02": "Feb",
				"03": "Mar",
				"04": "Apr",
				"05": "May",
				"06": "Jun",
				"07": "Jul",
				"08": "Aug",
				"09": "Sep",
				"10": "Oct",
				"11": "Nov",
				"12": "Dec",
			}
			date_display = f"{month_names.get(month, month)} {day}, {year}"

		tags = ", ".join(session.get("tags", []))
		lines.append(
			f"- Session {session_id} → {date_display} → Tags: [{tags}]"
		)

	return "\n".join(lines)
