"""Local LLM access through Ollama (http://localhost:11434)."""
from typing import Iterator

import ollama

PREFERRED_MODEL = "llama3.2"

SYSTEM_PROMPT = (
    "You are EchoMate, a friendly assistant. The user's message is a question "
    "or prompt captured from speech, so it may contain small transcription "
    "mistakes. Answer clearly and naturally in a concise, human tone "
    "(roughly 3-6 sentences unless more is clearly needed)."
)


class OllamaNotRunning(Exception):
    pass


class ModelNotFound(Exception):
    pass


class LLMService:
    def __init__(self, preferred: str = PREFERRED_MODEL):
        self.preferred = preferred
        self.model_name: str = preferred

    def list_models(self) -> list[str]:
        try:
            resp = ollama.list()
        except Exception as exc:
            raise OllamaNotRunning(str(exc)) from exc
        models = resp["models"] if isinstance(resp, dict) else resp.models
        names = []
        for m in models:
            name = m["model"] if isinstance(m, dict) else getattr(m, "model", None)
            if name:
                names.append(name)
        return names

    def check(self) -> tuple[bool, str]:
        """Returns (ollama_running, selected_model_name or '')."""
        try:
            names = self.list_models()
        except OllamaNotRunning:
            return False, ""
        if not names:
            return True, ""
        match = next((n for n in names if n.startswith(self.preferred)), None)
        self.model_name = match or names[0]
        return True, self.model_name

    def stream_answer(self, question: str) -> Iterator[str]:
        running, model = self.check()
        if not running:
            raise OllamaNotRunning("Ollama is not running")
        if not model:
            raise ModelNotFound("No models installed")
        try:
            stream = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                stream=True,
            )
            for part in stream:
                msg = part["message"] if isinstance(part, dict) else part.message
                content = msg["content"] if isinstance(msg, dict) else msg.content
                if content:
                    yield content
        except ollama.ResponseError as exc:
            raise ModelNotFound(str(exc)) from exc
        except Exception as exc:
            raise OllamaNotRunning(str(exc)) from exc
