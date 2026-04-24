def format_timestamp(ts: str) -> str:
    """Convert ISO timestamp to readable format like 'Jan 5, 2026 at 11:14 PM'"""
    from datetime import datetime
    dt = datetime.fromisoformat(ts)
    return dt.strftime("%b %d, %Y at %I:%M %p")
