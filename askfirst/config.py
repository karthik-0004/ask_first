from dotenv import load_dotenv
import os

load_dotenv()  # this reads the .env file automatically

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = "gpt-4o"
MAX_TOKENS = 8192
TEMPERATURE = 0.2   # low temperature for precise medical-style reasoning
DATA_PATH = "data/askfirst_synthetic_dataset.json"

# Pattern detection settings
MIN_CONFIDENCE_THRESHOLD = 0.2   # patterns below this are discarded
MAX_SESSIONS_IN_CONTEXT = 10     # max sessions per user to send per reasoning call
STREAM_ENABLED = True
