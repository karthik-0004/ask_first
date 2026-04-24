"""Data ingestion and session formatting utilities for AskFirst."""

import json

from utils.helpers import format_timestamp


def load_dataset(path: str) -> dict:
	"""Load and parse the AskFirst JSON dataset from disk.

	Args:
		path: File path to the dataset JSON file.

	Returns:
		The parsed dataset as a dictionary.
	"""
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def get_all_users(dataset: dict) -> list[dict]:
	"""Return all user objects from the dataset.

	Args:
		dataset: Parsed dataset dictionary.

	Returns:
		A list of user dictionaries from dataset["users"].
	"""
	return dataset.get("users", [])


def get_user_by_id(dataset: dict, user_id: str) -> dict | None:
	"""Find and return a user by user_id.

	Args:
		dataset: Parsed dataset dictionary.
		user_id: User identifier to match.

	Returns:
		The matching user dictionary if found, otherwise None.
	"""
	for user in get_all_users(dataset):
		if user.get("user_id") == user_id:
			return user
	return None


def get_user_sessions(user: dict) -> list[dict]:
	"""Return a user's conversation sessions sorted by timestamp ascending.

	Args:
		user: User dictionary containing a conversations list.

	Returns:
		A new list of session dictionaries ordered by timestamp.
	"""
	sessions = user.get("conversations", [])
	return sorted(sessions, key=lambda session: session.get("timestamp", ""))


def summarize_session(session: dict) -> str:
	"""Format a single session into a readable summary block.

	Args:
		session: Session dictionary with session metadata and messages.

	Returns:
		A formatted multi-line session summary string.
	"""
	session_id = session.get("session_id", "")
	readable_timestamp = format_timestamp(session.get("timestamp", ""))
	severity = session.get("severity", "")
	user_message = session.get("user_message", "")
	user_followup = session.get("user_followup")
	clary_response = session.get("clary_response", "")
	tags = ", ".join(session.get("tags", []))

	lines = [
		f"--- Session {session_id} | {readable_timestamp} | Severity: {severity} ---",
		f'User said: "{user_message}"',
	]

	if user_followup:
		lines.append(f'Follow-up: "{user_followup}"')

	lines.extend(
		[
			f'Clary responded: "{clary_response}"',
			f"Tags: {tags}",
		]
	)

	return "\n".join(lines)
