from dotenv import load_dotenv
import os

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")
AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME", "")

REPROMPTING = os.getenv("REPROMPTING", "true").casefold() == "true"
MAX_PROMPTS = int(os.getenv("MAX_PROMPTS", "2"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1024))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 128))