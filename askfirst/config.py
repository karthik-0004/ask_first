import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")   # loaded from environment
MODEL_NAME = "gpt-4o"
MAX_TOKENS = 4096
TEMPERATURE = 0.2   # low temperature for precise medical-style reasoning
DATA_PATH = "data/askfirst_synthetic_dataset.json"

# Pattern detection settings
MIN_CONFIDENCE_THRESHOLD = 0.4   # patterns below this are discarded
MAX_SESSIONS_IN_CONTEXT = 10     # max sessions per user to send per reasoning call
STREAM_ENABLED = True
