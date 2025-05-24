import os
from dotenv import load_dotenv
load_dotenv()

#
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo" # cheap, fast and efficient
OPENAI_TEMPERATURE = 0.7

#
OLLAMA_API = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama2"
OLLAMA_TEMPERATURE = 0.7