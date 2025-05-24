import ollama
from config.settings import OLLAMA_MODEL, OLLAMA_TEMPERATURE


def call_ollama(messages):

    # Ollama API hỗ trợ stream trả về iterator
    response_stream = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        temperature=OLLAMA_TEMPERATURE,
        stream=True
    )

    response = ""
    for chunk in response_stream:
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content", "")
        response += content
        yield response
