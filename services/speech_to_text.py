"""Local speech-to-text with Faster-Whisper (runs fully offline after the
one-time model download)."""
from faster_whisper import WhisperModel

MODEL_SIZE = "base"  # use "small" for better accuracy on a stronger PC


class SpeechToText:
    def __init__(self, model_size: str = MODEL_SIZE):
        self.model_size = model_size
        self._model: WhisperModel | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is None:
            self._model = WhisperModel(
                self.model_size, device="cpu", compute_type="int8"
            )

    def transcribe(self, wav_path: str) -> str:
        self.load()
        segments, _info = self._model.transcribe(
            wav_path, beam_size=5, vad_filter=True
        )
        return " ".join(s.text.strip() for s in segments).strip()
