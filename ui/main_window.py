"""EchoMate main window (PySide6)."""
import os

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QTextEdit, QVBoxLayout, QWidget,
)

from services.audio_recorder import AudioRecorder, MicrophoneError
from services.llm_service import LLMService, ModelNotFound, OllamaNotRunning
from services.speech_to_text import SpeechToText

STYLE = """
QMainWindow, #root { background: #F4F6FD; }
QLabel { color: #2B2F4A; }
#title { font-size: 28px; font-weight: 700; color: #4F46E5; }
#subtitle { font-size: 12px; color: #7A7F9C; }
#card { background: white; border: 1px solid #E4E7F7; border-radius: 20px; }
#cardTitle { font-size: 16px; font-weight: 600; color: #2B2F4A; }
#status { font-size: 12px; color: #7A7F9C; }
#footer { font-size: 11px; color: #7A7F9C; }
#pill { background: #EEF0FF; border-radius: 12px; padding: 5px 12px;
        font-size: 11px; color: #4F46E5; }
QTextEdit { background: #F8F9FF; border: 1px solid #E4E7F7; border-radius: 14px;
            padding: 10px; font-size: 14px; color: #2B2F4A; }
QPushButton { border: none; border-radius: 14px; padding: 10px 18px;
              font-size: 13px; font-weight: 600; }
QPushButton:disabled, #primary:disabled, #danger:disabled, #ghost:disabled,
#accent:disabled { background: #E9EBF5; color: #A6AAC3; }
#primary { background: #6366F1; color: white; }
#primary:hover { background: #4F46E5; }
#danger { background: #F43F5E; color: white; }
#danger:hover { background: #E11D48; }
#ghost { background: #EEF0FF; color: #4F46E5; }
#ghost:hover { background: #E0E4FF; }
#accent { background: #8B5CF6; color: white; }
#accent:hover { background: #7C3AED; }
"""


# ---------- background workers ----------
class TranscribeWorker(QThread):
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, stt: SpeechToText, wav_path: str):
        super().__init__()
        self.stt, self.wav_path = stt, wav_path

    def run(self):
        try:
            self.done.emit(self.stt.transcribe(self.wav_path))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            try:
                os.remove(self.wav_path)  # clean up temp audio
            except OSError:
                pass


class LLMWorker(QThread):
    token = Signal(str)
    done = Signal()
    failed = Signal(str)  # "ollama" | "model" | other message

    def __init__(self, llm: LLMService, question: str):
        super().__init__()
        self.llm, self.question = llm, question
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            for piece in self.llm.stream_answer(self.question):
                if self._cancel:
                    return
                self.token.emit(piece)
            self.done.emit()
        except OllamaNotRunning:
            self.failed.emit("ollama")
        except ModelNotFound:
            self.failed.emit("model")
        except Exception as exc:
            self.failed.emit(str(exc))


class StatusWorker(QThread):
    result = Signal(bool, str)

    def __init__(self, llm: LLMService):
        super().__init__()
        self.llm = llm

    def run(self):
        running, model = self.llm.check()
        self.result.emit(running, model)


# ---------- main window ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EchoMate")
        self.resize(820, 780)
        self.setMinimumSize(640, 620)
        self.setStyleSheet(STYLE)

        self.recorder = AudioRecorder()
        self.stt = SpeechToText()
        self.llm = LLMService()
        self.transcribe_worker = None
        self.llm_worker = None
        self.status_worker = None
        self.pulse_on = False
        self.dots = 0

        self._build_ui()

        self.pulse_timer = QTimer(self, interval=500, timeout=self._pulse)
        self.dots_timer = QTimer(self, interval=400, timeout=self._animate_dots)
        self.poll_timer = QTimer(self, interval=5000, timeout=self._refresh_ai_status)
        self.poll_timer.start()
        self._refresh_ai_status()

    # ----- UI construction -----
    def _card(self, title: str):
        card = QFrame(objectName="card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(12)
        lay.addWidget(QLabel(title, objectName="cardTitle"))
        return card, lay

    def _btn(self, text, kind, slot):
        b = QPushButton(text, objectName=kind)
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(slot)
        return b

    def _build_ui(self):
        root = QWidget(objectName="root")
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(28, 22, 28, 16)
        main.setSpacing(16)

        # header
        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(QLabel("🎙 EchoMate", objectName="title"))
        titles.addWidget(QLabel("Your personal offline AI voice companion",
                                objectName="subtitle"))
        head.addLayout(titles)
        head.addStretch()
        self.ai_status = QLabel("● Checking...", objectName="pill")
        head.addWidget(self.ai_status, alignment=Qt.AlignTop)
        main.addLayout(head)

        # card 1: voice input
        c1, l1 = self._card("🎤  Voice Input")
        row = QHBoxLayout()
        self.start_btn = self._btn("▶  Start Listening", "primary", self.start_listening)
        self.stop_btn = self._btn("■  Stop Listening", "danger", self.stop_listening)
        self.stop_btn.setEnabled(False)
        self.clear_btn = self._btn("Clear", "ghost", self.clear_transcription)
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        row.addStretch()
        row.addWidget(self.clear_btn)
        l1.addLayout(row)
        self.rec_status = QLabel("Hey! Ready when you are.", objectName="status")
        l1.addWidget(self.rec_status)
        self.transcript = QTextEdit()
        self.transcript.setPlaceholderText("Your words will show up here...")
        l1.addWidget(self.transcript)
        main.addWidget(c1, 1)

        # card 2: answer
        c2, l2 = self._card("✨  AI Suggested Answer")
        row2 = QHBoxLayout()
        self.suggest_btn = self._btn("💡  Suggest Answer", "accent", self.suggest_answer)
        self.copy_btn = self._btn("Copy Answer", "ghost", self.copy_answer)
        row2.addWidget(self.suggest_btn)
        row2.addStretch()
        row2.addWidget(self.copy_btn)
        l2.addLayout(row2)
        self.ai_msg = QLabel("", objectName="status")
        l2.addWidget(self.ai_msg)
        self.answer = QTextEdit()
        self.answer.setReadOnly(True)
        self.answer.setPlaceholderText("Your suggested answer will appear here...")
        l2.addWidget(self.answer)
        main.addWidget(c2, 1)

        # footer
        foot = QHBoxLayout()
        foot.addWidget(QLabel("🔒 100% Local Processing", objectName="footer"))
        foot.addStretch()
        self.model_label = QLabel("Model: —", objectName="footer")
        foot.addWidget(self.model_label)
        main.addLayout(foot)

    # ----- AI status -----
    def _refresh_ai_status(self):
        if self.status_worker and self.status_worker.isRunning():
            return
        self.status_worker = StatusWorker(self.llm)
        self.status_worker.result.connect(self._set_ai_status)
        self.status_worker.start()

    def _set_ai_status(self, running: bool, model: str):
        if running and model:
            self.ai_status.setText("● Local AI Online")
            self.ai_status.setStyleSheet("background:#DCFCE7;color:#15803D;")
            self.model_label.setText(f"Model: {model}  •  Whisper: {self.stt.model_size}")
        elif running:
            self.ai_status.setText("● No model installed")
            self.ai_status.setStyleSheet("background:#FEF3C7;color:#B45309;")
            self.model_label.setText(f"Model: none  •  Whisper: {self.stt.model_size}")
        else:
            self.ai_status.setText("● Local AI Offline")
            self.ai_status.setStyleSheet("background:#FEE2E2;color:#B91C1C;")
            self.model_label.setText(f"Model: —  •  Whisper: {self.stt.model_size}")

    # ----- recording -----
    def start_listening(self):
        try:
            self.recorder.start()
        except MicrophoneError:
            self.rec_status.setText(
                "😕 I can't reach your microphone. Check that it's plugged in and "
                "that Windows allows microphone access for desktop apps.")
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.rec_status.setText("🔴  I'm listening...")
        self.pulse_timer.start()

    def _pulse(self):
        self.pulse_on = not self.pulse_on
        dot = "🔴" if self.pulse_on else "⭕"
        self.rec_status.setText(f"{dot}  I'm listening...")

    def stop_listening(self):
        self.pulse_timer.stop()
        self.stop_btn.setEnabled(False)
        wav = self.recorder.stop()
        if not wav:
            self.start_btn.setEnabled(True)
            self.rec_status.setText("Hmm, I didn't catch anything. Try again?")
            return
        first = "" if self.stt.is_loaded else " (first time takes a bit longer)"
        self.rec_status.setText(f"Got it! Converting your speech into text...{first}")
        self.transcribe_worker = TranscribeWorker(self.stt, wav)
        self.transcribe_worker.done.connect(self._on_transcribed)
        self.transcribe_worker.failed.connect(self._on_transcribe_failed)
        self.transcribe_worker.start()

    def _on_transcribed(self, text: str):
        self.start_btn.setEnabled(True)
        if not text:
            self.rec_status.setText("I couldn't hear any words. Try speaking a bit closer to the mic.")
            return
        existing = self.transcript.toPlainText().strip()
        self.transcript.setPlainText((existing + " " + text).strip())
        self.rec_status.setText("Done! You can record more or hit Suggest Answer.")

    def _on_transcribe_failed(self, err: str):
        self.start_btn.setEnabled(True)
        self.rec_status.setText(
            "😕 Couldn't load the speech model. If this is your first run, make sure "
            "you're online once so it can download. (" + err[:120] + ")")

    def clear_transcription(self):
        self.transcript.clear()
        self.rec_status.setText("Hey! Ready when you are.")

    # ----- LLM -----
    def suggest_answer(self):
        question = self.transcript.toPlainText().strip()
        if not question:
            self.ai_msg.setText("Say something first, or type your question above 🙂")
            return
        if self.llm_worker and self.llm_worker.isRunning():
            return
        self.answer.clear()
        self.suggest_btn.setEnabled(False)
        self.dots = 0
        self.dots_timer.start()
        self.ai_msg.setText("Thinking of the best answer")
        self.llm_worker = LLMWorker(self.llm, question)
        self.llm_worker.token.connect(self._on_token)
        self.llm_worker.done.connect(self._on_llm_done)
        self.llm_worker.failed.connect(self._on_llm_failed)
        self.llm_worker.start()

    def _animate_dots(self):
        self.dots = (self.dots + 1) % 4
        base = "Thinking of the best answer" if not self.answer.toPlainText() \
            else "Typing"
        self.ai_msg.setText(base + "." * self.dots)

    def _on_token(self, piece: str):
        cursor = self.answer.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(piece)
        self.answer.setTextCursor(cursor)
        self.answer.ensureCursorVisible()

    def _finish_llm(self):
        self.dots_timer.stop()
        self.suggest_btn.setEnabled(True)

    def _on_llm_done(self):
        self._finish_llm()
        self.ai_msg.setText("Here's what I came up with!")

    def _on_llm_failed(self, kind: str):
        self._finish_llm()
        if kind == "ollama":
            self.ai_msg.setText(
                "😕 My brain (Ollama) isn't running. Open the Ollama app or run "
                "'ollama serve', then try again.")
        elif kind == "model":
            self.ai_msg.setText(
                "😕 I couldn't find the AI model. Run 'ollama pull llama3.2' in a "
                "terminal and try again.")
        else:
            self.ai_msg.setText("😕 Something went wrong: " + kind[:150])
        self._refresh_ai_status()

    def copy_answer(self):
        text = self.answer.toPlainText().strip()
        if not text:
            self.ai_msg.setText("Nothing to copy yet!")
            return
        QGuiApplication.clipboard().setText(text)
        self.ai_msg.setText("Copied to clipboard ✅")

    # ----- cleanup -----
    def closeEvent(self, event):
        if self.recorder.is_recording:
            wav = self.recorder.stop()
            if wav:
                try:
                    os.remove(wav)
                except OSError:
                    pass
        if self.llm_worker and self.llm_worker.isRunning():
            self.llm_worker.cancel()
            self.llm_worker.wait(2000)
        for w in (self.transcribe_worker, self.status_worker):
            if w and w.isRunning():
                w.wait(3000)
        super().closeEvent(event)
