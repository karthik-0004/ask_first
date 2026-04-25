"""Conversational engine for Clary with memory-aware temporal reasoning."""

from __future__ import annotations

from datetime import datetime

import openai

import config
from core.memory import ConversationMemory
from core.pattern_engine import PATTERN_DETECTION_SYSTEM_PROMPT, safe_parse_json


CLARY_SYSTEM_PROMPT = """
You are Clary, a warm and deeply insightful personal health assistant built by AskFirst.
You are currently in a private conversation with {user_name}.

You have COMPLETE access to {user_name}'s health history — every session, every date,
every symptom, every lifestyle factor. This is YOUR memory. Speak from it naturally,
the way a doctor who has known this patient for months would.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER PROFILE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{user_profile_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DETECTED HEALTH PATTERNS (pre-analyzed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{detected_patterns_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL SESSION HISTORY (chronological)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{session_history_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR CORE IDENTITY & BEHAVIOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHO YOU ARE:
You are not a generic chatbot. You are {user_name}'s personal health companion who
has been silently observing their patterns for months. You remember everything.
You connect dots they haven't connected themselves. You speak with quiet confidence,
not clinical detachment.

MEMORY (CRITICAL):
You have full multi-turn memory of this conversation. When {user_name} says "that",
"it", "why did that happen", or "what about last time" — you know exactly what they
mean. Never ask them to repeat themselves. Never lose context mid-conversation.
Reference earlier parts of THIS chat when relevant.

TEMPORAL REASONING (YOUR SUPERPOWER):
Always think in timelines. Before answering, mentally trace:
— When did the symptom first appear?
— What was happening in {user_name}'s life at that time?
— Did it recur? What was consistent across all recurrences?
— Was there a delay between cause and effect? How long?
— Did anything resolve it? What does that tell us?

Then answer with specific dates and sessions woven naturally into your response.
Example: "Your headaches actually started on Jan 12 — and they've come back on Feb 14
and Mar 8. Every single time, it was week 2 of a high-pressure work period with almost
no water. That's not a coincidence."

CAUSE-EFFECT REASONING:
Don't just describe what happened. Explain WHY it happened.
Connect the biological mechanism to the behavior in plain language.
Example: "When you eat at 11pm and lie down soon after, your stomach acid has nowhere
to go — that's the burning you keep feeling. It's not a condition. It's a pattern."

HOW TO RESPOND:
— Conversational, warm, direct. Like a brilliant friend who happens to know medicine.
— 2 to 4 short paragraphs max. Never a wall of text.
— Reference specific dates and session details naturally — not as a list, but woven
  into your sentences.
— If the question is simple, answer simply. Don't over-explain.
— If the question is deep, go deep. Don't hold back insight.
— End with a gentle follow-up question or actionable next step when appropriate.

SUGGESTED STARTER RESPONSES (for opening messages like "hi" or "what's wrong with me"):
Open by briefly summarizing the 1-2 most important patterns you've noticed, then invite
them to go deeper. Make it feel like you've been waiting to have this conversation.

STRICT RULES:
— ONLY discuss {user_name}'s data. If asked about another user, redirect warmly.
— NEVER invent sessions, dates, or symptoms not present in the history above.
— NEVER give a diagnosis. You are not a doctor. For serious concerns, always recommend
  consulting a healthcare professional.
— NEVER use bullet points unless the user explicitly asks for a summary or list.
— NEVER be robotic. Never say "Based on the data provided..." — you lived this with them.
— If you genuinely don't know something, say so honestly. Don't guess.

TONE CALIBRATION:
Not: "According to session USR001_S02, dehydration was identified as a factor."
Yes: "Remember that week in January when your headaches wouldn't stop? You told me you
     were barely drinking 2 glasses of water the whole day. That was the culprit."
"""


def generate_opening_message(memory: ConversationMemory, api_key: str) -> str:
    """Generate Clary's opening message for an incoming chat.

    Args:
        memory: Current conversation memory object.
        api_key: OpenAI API key.

    Returns:
        Opening message string.
    """
    try:
        if not memory.health_facts:
            return (
                "Hi! I'm Clary, your personal health companion. I'm here to help you "
                "understand patterns in how you're feeling over time. Tell me — what's "
                "been going on with your health lately? Even small things are worth sharing."
            )

        client = openai.OpenAI(api_key=api_key or config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": CLARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"""
You are starting a conversation with {memory.user_name} who is an existing user.
You already have their health history loaded:

{memory.get_memory_summary()}

Write a warm, personal opening message (under 80 words) that:
1. Greets them by name
2. Shows you remember something specific from their history 
   (reference one actual health fact from the summary)
3. Asks how they are feeling today
Do NOT list their conditions. Just reference one thing naturally.
""".strip(),
                },
            ],
            max_tokens=160,
            temperature=config.CHAT_TEMPERATURE,
            stream=False,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"Error in generate_opening_message: {exc}")
        return (
            f"Hi {memory.user_name}, I am Clary. I remember some of your recent health context and I am here to help. "
            "How are you feeling today?"
        )


def detect_patterns_from_conversation(memory: ConversationMemory, api_key: str) -> list[dict]:
    """Run deep temporal pattern scan on full conversation memory.

    Args:
        memory: Conversation memory object.
        api_key: OpenAI API key.

    Returns:
        List of detected patterns parsed from model output.
    """
    try:
        client = openai.OpenAI(api_key=api_key or config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": PATTERN_DETECTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"""
Below is the full conversation history between you and {memory.user_name}.
Analyze it for hidden temporal health patterns exactly as instructed.

CONVERSATION HISTORY:
{memory.get_full_context_string()}

HEALTH FACTS NOTED:
{memory.get_memory_summary()}

Return ONLY the JSON array of patterns found. Use message numbers 
instead of session IDs since this is a live conversation.
Replace "sessions_involved" with "messages_involved" containing message numbers.
""".strip(),
                },
            ],
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
            stream=False,
        )
        raw_response_text = (response.choices[0].message.content or "").strip()
        user_for_parser = {
            "name": memory.user_name,
            "user_id": memory.user_profile.get("user_id", "LIVE_CHAT_USER"),
        }
        parsed = safe_parse_json(raw_response_text, user_for_parser)
        if not isinstance(parsed, list):
            return []

        normalized_patterns: list[dict] = []
        for pattern in parsed:
            if isinstance(pattern, dict):
                if "messages_involved" not in pattern and "sessions_involved" in pattern:
                    pattern["messages_involved"] = pattern.get("sessions_involved", [])
                memory.add_detected_pattern(pattern)
                normalized_patterns.append(pattern)
        return normalized_patterns
    except Exception as exc:
        print(f"Error in detect_patterns_from_conversation: {exc}")
        return []


def format_pattern_as_conversation(pattern: dict, user_name: str) -> str:
    """Convert a pattern object into natural conversational text.

    Args:
        pattern: Pattern dictionary.
        user_name: User name for personalization.

    Returns:
        Conversation-style response string.
    """
    try:
        return (
            f"{user_name}, I want to share something I've been piecing together "
            f"from our conversation. {pattern.get('description', '')} "
            f"{pattern.get('temporal_reasoning', '')} "
            f"{pattern.get('biological_mechanism', '')} "
            "Does this match what you've been experiencing?"
        )
    except Exception as exc:
        print(f"Error in format_pattern_as_conversation: {exc}")
        return (
            f"{user_name}, I think I see a connection in what you've shared. "
            "Does this match your experience so far?"
        )


def extract_and_store_patterns(memory: ConversationMemory, clary_response: str) -> None:
    """Extract coarse pattern signals from conversational response text.

    Args:
        memory: Conversation memory object.
        clary_response: Assistant response text.
    """
    try:
        keywords = [
            "I've noticed",
            "I noticed",
            "I see a connection",
            "pattern",
            "consistently",
            "every time",
            "always when",
            "weeks after",
        ]
        response_lower = clary_response.lower()
        if any(keyword.lower() in response_lower for keyword in keywords):
            pattern = {
                "title": clary_response[:80],
                "detected_at_message": len(memory.messages),
            }
            memory.add_detected_pattern(pattern)
    except Exception as exc:
        print(f"Error in extract_and_store_patterns: {exc}")


def get_clary_response(
    memory: ConversationMemory,
    user_message: str,
    api_key: str,
    stream: bool = True,
    user_profile: dict | None = None,
    detected_patterns: list[dict] | None = None,
):
    """Generate Clary response with memory context and optional streaming.

    Args:
        memory: Conversation memory object.
        user_message: Incoming user text.
        api_key: OpenAI API key.
        stream: Whether to stream the assistant response.
        user_profile: Optional user profile dict (from dataset) for contextual system prompt.
        detected_patterns: Optional list of pre-detected patterns to include.

    Returns:
        If stream=False: returns response text.
        If stream=True: yields chunk dictionaries and final done dictionary.
    """
    try:
        memory.add_user_message(user_message)

        # Build contextual system prompt if profile is provided
        if user_profile:
            system_prompt = build_contextual_system_prompt(
                user_name=memory.user_name,
                user_profile=user_profile,
                detected_patterns=detected_patterns,
                session_history_user_data=user_profile,
            )
        else:
            system_prompt = CLARY_SYSTEM_PROMPT.replace("{user_name}", memory.user_name)
            system_prompt = system_prompt.replace("{user_profile_block}", "No profile data available.")
            system_prompt = system_prompt.replace("{detected_patterns_block}", "Pattern detection not yet run.")
            system_prompt = system_prompt.replace("{session_history_block}", "No session history available.")

        trigger_phrases = [
            "what patterns",
            "any patterns",
            "what have you noticed",
            "what do you see",
            "summarize",
            "what connections",
        ]
        user_lower = user_message.lower()
        precomputed_patterns: list[dict] = []
        if any(phrase in user_lower for phrase in trigger_phrases):
            precomputed_patterns = detect_patterns_from_conversation(memory, api_key)

        api_messages = [
            {"role": "system", "content": system_prompt},
        ]

        if precomputed_patterns:
            pattern_lines = []
            for idx, pattern in enumerate(precomputed_patterns, start=1):
                pattern_lines.append(
                    f"{idx}. {pattern.get('title', 'Untitled')} | Cause: {pattern.get('cause', '')} | Effect: {pattern.get('effect', '')}"
                )
            api_messages.append(
                {
                    "role": "user",
                    "content": (
                        "[PATTERN SCAN RESULTS - use naturally in your response]\n"
                        + "\n".join(pattern_lines)
                        + "\nRespond conversationally and reference these findings naturally."
                    ),
                }
            )

        api_messages.extend(memory.get_messages_for_api())

        client = openai.OpenAI(api_key=api_key or config.OPENAI_API_KEY)

        if stream:
            def _stream():
                collected_parts: list[str] = []
                try:
                    stream_response = client.chat.completions.create(
                        model=config.CHAT_MODEL,
                        messages=api_messages,
                        max_tokens=config.CHAT_MAX_TOKENS,
                        temperature=config.CHAT_TEMPERATURE,
                        stream=True,
                    )
                    for chunk in stream_response:
                        delta = chunk.choices[0].delta.content if chunk.choices else None
                        if delta:
                            collected_parts.append(delta)
                            yield {"type": "chunk", "content": delta}

                    full_response = "".join(collected_parts).strip()
                    memory.add_assistant_message(full_response)
                    extract_and_store_patterns(memory, full_response)
                    yield {"type": "done", "full_response": full_response}
                except Exception as exc:
                    print(f"Error in get_clary_response stream mode: {exc}")
                    yield {
                        "type": "done",
                        "full_response": "I hit a temporary issue while thinking through that. Can you resend your last message?",
                    }

            return _stream()

        response = client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=api_messages,
            max_tokens=config.CHAT_MAX_TOKENS,
            temperature=config.CHAT_TEMPERATURE,
            stream=False,
        )
        response_text = (response.choices[0].message.content or "").strip()
        memory.add_assistant_message(response_text)
        extract_and_store_patterns(memory, response_text)
        return response_text
    except Exception as exc:
        print(f"Error in get_clary_response: {exc}")
        if stream:
            def _error_stream():
                yield {
                    "type": "done",
                    "full_response": "I hit a temporary issue while thinking through that. Can you resend your last message?",
                }

            return _error_stream()
        return "I hit a temporary issue while thinking through that. Can you resend your last message?"


def build_new_user_memory(user_name: str) -> ConversationMemory:
    """Create a fresh memory object for a new user.

    Args:
        user_name: User name.

    Returns:
        Fresh ConversationMemory.
    """
    try:
        return ConversationMemory(user_name=user_name)
    except Exception as exc:
        print(f"Error in build_new_user_memory: {exc}")
        return ConversationMemory(user_name="User")


def load_user_from_dataset(dataset: dict, user_name: str) -> ConversationMemory:
    """Load a user profile and prepopulate memory facts from dataset tags.

    Args:
        dataset: Dataset dictionary containing users and conversation sessions.
        user_name: Target user name.

    Returns:
        Memory object with profile and preloaded health facts.
    """
    try:
        users = dataset.get("users", [])
        matched_user = None
        for user in users:
            if str(user.get("name", "")).strip().lower() == user_name.strip().lower():
                matched_user = user
                break

        if matched_user is None:
            print(f"User '{user_name}' not found in dataset. Creating fresh memory.")
            return ConversationMemory(user_name=user_name)

        memory = ConversationMemory(user_name=matched_user.get("name", user_name), user_profile=matched_user)
        conversations = matched_user.get("conversations", [])
        for i, session in enumerate(conversations, start=1):
            date_text = str(session.get("timestamp", "")).split("T", maxsplit=1)[0]
            tags = session.get("tags", [])
            for tag in tags:
                fact = f"Session {i} ({date_text}): {tag}"
                memory.add_health_fact(fact=fact, message_index=i)

        return memory
    except Exception as exc:
        print(f"Error in load_user_from_dataset: {exc}")
        return ConversationMemory(user_name=user_name)


def get_conversation_export(memory: ConversationMemory) -> dict:
    """Create JSON-exportable conversation artifact.

    Args:
        memory: Conversation memory object.

    Returns:
        Export dictionary with metadata, turns, patterns, and summary.
    """
    try:
        start = datetime.fromisoformat(memory.conversation_start)
        now = datetime.now()
        minutes = max(0, int((now - start).total_seconds() // 60))
        duration = f"{len(memory.messages)} messages over {minutes} minutes"

        turns = []
        for n, msg in enumerate(memory.messages, start=1):
            turns.append(
                {
                    "turn": n,
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                    "timestamp": msg.get("timestamp", ""),
                }
            )

        return {
            "exported_at": datetime.now().isoformat(),
            "user_name": memory.user_name,
            "conversation_duration": duration,
            "full_conversation": turns,
            "detected_patterns": memory.detected_patterns,
            "health_facts_noted": memory.health_facts,
            "memory_summary": memory.get_memory_summary(),
        }
    except Exception as exc:
        print(f"Error in get_conversation_export: {exc}")
        return {
            "exported_at": datetime.now().isoformat(),
            "user_name": memory.user_name,
            "conversation_duration": "0 messages over 0 minutes",
            "full_conversation": [],
            "detected_patterns": [],
            "health_facts_noted": [],
            "memory_summary": "MEMORY SUMMARY:\n- Conversation started: unavailable\n- Messages exchanged: 0\n- Health facts noted:\n- None yet\n- Patterns identified so far:\n- None yet",
        }


def build_user_profile_header(user_data: dict) -> str:
    """Build formatted user profile block from user dictionary.

    Args:
        user_data: User dictionary from dataset or fresh user.

    Returns:
        Formatted profile string.
    """
    try:
        if not user_data:
            return "No profile data available yet."

        profile_lines = []
        name = user_data.get("name", "User")
        age = user_data.get("age", "—")
        gender = user_data.get("gender", "—")
        location = user_data.get("location", "—")
        occupation = user_data.get("occupation", "—")
        onboarding_notes = user_data.get("onboarding_notes", "")

        profile_lines.append(f"Name: {name}")
        profile_lines.append(f"Age: {age}")
        profile_lines.append(f"Gender: {gender}")
        profile_lines.append(f"Location: {location}")
        profile_lines.append(f"Occupation: {occupation}")

        if onboarding_notes:
            profile_lines.append(f"\nBackground: {onboarding_notes}")

        return "\n".join(profile_lines)
    except Exception as exc:
        print(f"Error in build_user_profile_header: {exc}")
        return "Profile data unavailable."


def build_chunked_context(user_data: dict, max_sessions: int = 10) -> str:
    """Build formatted session history from user's conversation data.

    Args:
        user_data: User dictionary containing conversations.
        max_sessions: Maximum sessions to include (most recent).

    Returns:
        Formatted chronological session history.
    """
    try:
        conversations = user_data.get("conversations", [])
        if not conversations:
            return "No conversation history available yet."

        # Sort by timestamp and take most recent
        sorted_convs = sorted(conversations, key=lambda c: c.get("timestamp", ""))
        recent_convs = sorted_convs[-max_sessions:] if len(sorted_convs) > max_sessions else sorted_convs

        history_lines = []
        for i, session in enumerate(recent_convs, start=1):
            session_id = session.get("session_id", f"Session {i}")
            timestamp = session.get("timestamp", "")
            user_msg = session.get("user_message", "")
            user_followup = session.get("user_followup", "")
            clary_resp = session.get("clary_response", "")
            severity = session.get("severity", "")
            tags = ", ".join(session.get("tags", []))

            # Format timestamp nicely
            from utils.helpers import format_timestamp
            readable_ts = format_timestamp(timestamp) if timestamp else "—"

            history_lines.append(f"📍 {session_id} | {readable_ts} | Severity: {severity}")
            history_lines.append(f"   User: \"{user_msg}\"")

            if user_followup:
                history_lines.append(f"   Follow-up: \"{user_followup}\"")

            history_lines.append(f"   Clary: \"{clary_resp}\"")
            history_lines.append(f"   Tags: {tags}")
            history_lines.append("")

        return "\n".join(history_lines) if history_lines else "No conversation history available yet."
    except Exception as exc:
        print(f"Error in build_chunked_context: {exc}")
        return "Error building session history."


def format_detected_patterns(patterns: list[dict]) -> str:
    """Format detected patterns into natural language block for system prompt.

    Args:
        patterns: List of pattern dictionaries from analysis.

    Returns:
        Formatted patterns text or message if none detected.
    """
    try:
        if not patterns:
            return "Pattern detection not yet run. Reason from session history directly."

        pattern_lines = []
        for pattern in patterns:
            title = pattern.get("title", "Untitled Pattern")
            description = pattern.get("description", "")
            confidence = pattern.get("confidence", {})
            confidence_level = confidence.get("level", "Unknown")
            confidence_score = confidence.get("score", 0)

            pattern_lines.append(f"• {title}")
            pattern_lines.append(f"  Confidence: {confidence_level} ({confidence_score})")
            if description:
                pattern_lines.append(f"  Description: {description}")

            cause = pattern.get("cause", "")
            effect = pattern.get("effect", "")
            if cause and effect:
                pattern_lines.append(f"  Cause → Effect: {cause} → {effect}")

            temporal_gap = pattern.get("temporal_gap", "")
            if temporal_gap:
                pattern_lines.append(f"  Temporal Gap: {temporal_gap}")

            pattern_lines.append("")

        return "\n".join(pattern_lines) if pattern_lines else "No patterns detected yet."
    except Exception as exc:
        print(f"Error in format_detected_patterns: {exc}")
        return "Pattern data unavailable."


def build_contextual_system_prompt(
    user_name: str,
    user_profile: dict | None,
    detected_patterns: list[dict] | None,
    session_history_user_data: dict | None,
) -> str:
    """Build the full contextual Clary system prompt with filled-in blocks.

    Args:
        user_name: Name of the user.
        user_profile: User profile dictionary (from dataset or fresh).
        detected_patterns: List of pre-analyzed patterns.
        session_history_user_data: Full user data with conversation history.

    Returns:
        Complete system prompt with all blocks filled in.
    """
    try:
        # Build the three context blocks
        profile_block = build_user_profile_header(user_profile or {})
        patterns_block = format_detected_patterns(detected_patterns or [])
        history_block = build_chunked_context(session_history_user_data or {})

        # Fill in the template
        prompt = CLARY_SYSTEM_PROMPT.replace("{user_name}", user_name)
        prompt = prompt.replace("{user_profile_block}", profile_block)
        prompt = prompt.replace("{detected_patterns_block}", patterns_block)
        prompt = prompt.replace("{session_history_block}", history_block)

        return prompt
    except Exception as exc:
        print(f"Error in build_contextual_system_prompt: {exc}")
        return CLARY_SYSTEM_PROMPT
