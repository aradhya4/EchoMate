"""Microphone recording using sounddevice. Audio stays on this machine."""
import os
import tempfile
import threading
import wave

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono


class MicrophoneError(Exception):
    """Raised when the microphone can't be opened."""


class AudioRecorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self.level = 0.0  # latest loudness (0..1), handy for UI

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def _callback(self, indata, frames, time_info, status):
        chunk = indata[:, 0].copy()
        with self._lock:
            self._chunks.append(chunk)
        self.level = float(min(1.0, np.sqrt(np.mean(chunk ** 2)) * 8))

    def start(self) -> None:
        if self._stream is not None:
            return
        self._chunks = []
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:  # no mic, permission denied, device busy...
            self._stream = None
            raise MicrophoneError(str(exc)) from exc

    def stop(self) -> str | None:
        """Stop recording and save a temporary WAV file. Returns its path
        (or None if nothing was recorded). Caller must delete the file."""
        if self._stream is None:
            return None
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
            self.level = 0.0

        with self._lock:
            if not self._chunks:
                return None
            audio = np.concatenate(self._chunks)
            self._chunks = []

        if audio.size < self.sample_rate * 0.3:  # shorter than 0.3s
            return None

        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="echomate_")
        os.close(fd)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())
        return path
