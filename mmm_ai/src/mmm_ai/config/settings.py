import os
from dotenv import load_dotenv

load_dotenv()

# Conversation Service V2 endpoint
CONVO_API = os.getenv("CONVO_API", "https://gai-conversation-api.annalect.com/v2/api")

# Auth + environment
API_KEY = os.getenv("CONVO_API_KEY")
ENV = os.getenv("ENV", "dev")

if not API_KEY:
    raise RuntimeError("Set CONVO_API_KEY in your environment or .env")

HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
}

# Model name string for raw HTTP calls:
MODEL_NAME_STR = "gpt-5"  # mirrors gai_templates.common.constants.GPT_5

# Default inference config; tune as needed
INFERENCE_DEFAULT = {
    "temperature": 0.2,
    "top_p": 1,
    "max_completion_tokens": 800
}
