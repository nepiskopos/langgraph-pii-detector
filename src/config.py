from dotenv import load_dotenv
import os

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")
AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME", "")

RETAIN_FALSE_POSITIVES = os.getenv("RETAIN_FALSE_POSITIVES", "true").casefold() == "true"
REPROMPTING = os.getenv("REPROMPTING", "true").casefold() == "true"
MAX_PROMPTS = int(os.getenv("MAX_PROMPTS", "2"))