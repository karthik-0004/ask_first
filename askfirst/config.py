from dotenv import load_dotenv
import os

load_dotenv()  # this reads the .env file automatically

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = "gpt-4o"
CHAT_MODEL = "gpt-4o"
MAX_TOKENS = 8192
TEMPERATURE = 0.2   # low temperature for precise medical-style reasoning
CHAT_TEMPERATURE = 0.7  # warmer for conversational responses
DATA_PATH = "data/askfirst_synthetic_dataset.json"

# Pattern detection settings
MIN_CONFIDENCE_THRESHOLD = 0.2   # patterns below this are discarded
MAX_SESSIONS_IN_CONTEXT = 10     # max sessions per user to send per reasoning call
MAX_MEMORY_MESSAGES = 50         # max conversation turns to keep in memory
CHAT_MAX_TOKENS = 1024           # max tokens per chat response

STREAM_ENABLED = True
