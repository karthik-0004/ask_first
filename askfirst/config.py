from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = "gpt-4o"
CHAT_MODEL = "gpt-4o"
MAX_TOKENS = 8192
TEMPERATURE = 0.2
CHAT_TEMPERATURE = 0.7

# ✅ Fixed path — works locally AND on Streamlit Cloud
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "askfirst_synthetic_dataset.json")

MIN_CONFIDENCE_THRESHOLD = 0.2
MAX_SESSIONS_IN_CONTEXT = 10
MAX_MEMORY_MESSAGES = 50
CHAT_MAX_TOKENS = 1024
STREAM_ENABLED = True
