import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-20b"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
TRAJECTORY_PATH = os.path.join(BASE_DIR, "data", "trajectories")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.85
NUM_SAMPLES = 2
SAMPLE_TEMPERATURE = 0.7

os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(TRAJECTORY_PATH, exist_ok=True)
