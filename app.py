import sys
import json
import requests
from typing import Optional, List, Dict, Any
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QTextCursor, QTextCharFormat, QFont, QColor
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextBrowser,
    QLineEdit, QPushButton, QComboBox, QLabel, QMessageBox, QDoubleSpinBox,
    QCheckBox
)


OLLAMA_HOST = "http://127.0.0.1:11434"
APP_DIR = Path.home() / ".ollama_pyqt"
CONFIG_PATH = APP_DIR / "config.json"


# ---------- Workers (in threads), so as not to block the UI ----------

class GenerateWorker(QObject):
    """
    Universal worker:
    - use_chat=False  => /api/generate (stateless, like `ollama run`)
    - use_chat=True   => /api/chat (with message history)
    """
    chunk = pyqtSignal(str)         # streaming response fragments
    finished = pyqtSignal(str)      # final full response
    failed = pyqtSignal(str)        # error

    def __init__(self,
                 model: str,
                 options: Optional[dict],
                 use_chat: bool,
                 prompt: Optional[str] = None,
                 messages: Optional[List[Dict[str, Any]]] = None):
        super().__init__()
        self.model = model
        self.options = options or {}
        self.use_chat = use_chat
        self.prompt = prompt or ""
        self.messages = messages or []

    def run(self):
        compiled: List[str] = []
        try:
            if not self.use_chat:
                # ----- /api/generate -----
                url = f"{OLLAMA_HOST}/api/generate"
                payload = {
                    "model": self.model,
                    "prompt": self.prompt,
                    "stream": True,
                    "options": self.options
                }
                with requests.post(url, json=payload, stream=True) as r:
                    r.raise_for_status()
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # standard streaming field for generate
                        if "response" in obj:
                            text = obj["response"]
                            compiled.append(text)
                            self.chunk.emit(text)

                        # just in case, support message.content if it arrives
                        elif "message" in obj and obj["message"] and "content" in obj["message"]:
                            text = obj["message"]["content"]
                            compiled.append(text)
                            self.chunk.emit(text)

                        if obj.get("done"):
                            break

            else:
                # ----- /api/chat -----
                url = f"{OLLAMA_HOST}/api/chat"
                payload = {
                    "model": self.model,
                    "messages": self.messages,  # [{'role':'user','content':...}, ...]
                    "stream": True,
                    "options": self.options
                }
                with requests.post(url, json=payload, stream=True) as r:
                    r.raise_for_status()
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # chat-stream increments come in message.content
                        if "message" in obj and obj["message"] and "content" in obj["message"]:
                            text = obj["message"]["content"]
                            compiled.append(text)
                            self.chunk.emit(text)

                        # some builds also put chunks in 'response'
                        elif "response" in obj:
                            text = obj["response"]
                            compiled.append(text)
                            self.chunk.emit(text)

                        if obj.get("done"):
                            break

            self.finished.emit("".join(compiled))
        except Exception as e:
            self.failed.emit(str(e))


# ---------- Config ----------

def load_config() -> dict:
    try:
        if CONFIG_PATH.exists():
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}

def save_config(cfg: dict):
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        with CONFIG_PATH.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------- Main window ----------

class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ollama Chat — PyQt6")
        self.resize(980, 680)

        # ---- state ----
        self._is_generating = False
        self._cfg = load_config()
        self._last_selected_model: Optional[str] = self._cfg.get("last_model")
        self._chat_mode: bool = bool(self._cfg.get("chat_mode", False))  # False = ollama run (stateless)
        self._history: List[Dict[str, str]] = []  # for chat mode

        # ---- UI ----
        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(True)
        self.chat.setReadOnly(True)

        self.input = QLineEdit()
        self.input.setPlaceholderText('Type your prompt and press "Enter" or click "Send"…')
        self.input.returnPressed.connect(self.on_send)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.on_send)

        # model selection
        self.model_combo = QComboBox()
        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setToolTip("Refresh model list (ollama list)")
        self.refresh_btn.clicked.connect(self.refresh_models)

        # mode: History (Chat)
        self.chat_mode_cb = QCheckBox("History (Chat)")
        self.chat_mode_cb.setChecked(self._chat_mode)
        self.chat_mode_cb.stateChanged.connect(self.on_toggle_chat_mode)

        self.clear_history_btn = QPushButton("Clear history")
        self.clear_history_btn.clicked.connect(self.on_clear_history)
        self.clear_history_btn.setEnabled(self._chat_mode)

        # top panel
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Model:"))
        top_bar.addWidget(self.model_combo, stretch=1)
        top_bar.addWidget(self.refresh_btn)
        top_bar.addSpacing(12)
        top_bar.addWidget(self.chat_mode_cb)
        top_bar.addWidget(self.clear_history_btn)

        # ---- Settings (temperature / top_p) ----
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setDecimals(2)
        self.temp_spin.setValue(float(self._cfg.get("temperature", 0.7)))

        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setDecimals(2)
        self.top_p_spin.setValue(float(self._cfg.get("top_p", 0.9)))

        settings_bar = QHBoxLayout()
        settings_bar.addWidget(QLabel("temperature:"))
        settings_bar.addWidget(self.temp_spin)
        settings_bar.addWidget(QLabel("top_p:"))
        settings_bar.addWidget(self.top_p_spin)
        settings_bar.addStretch()

        # bottom panel
        bottom_bar = QHBoxLayout()
        bottom_bar.addWidget(self.input, stretch=1)
        bottom_bar.addWidget(self.send_btn)

        root = QVBoxLayout(self)
        root.addLayout(top_bar)
        root.addLayout(settings_bar)
        root.addWidget(self.chat, stretch=1)
        root.addLayout(bottom_bar)

        # ---- Text formats ----
        self.fmt_normal = QTextCharFormat()
        self.fmt_bold = QTextCharFormat()
        self.fmt_bold.setFontWeight(QFont.Weight.Bold)

        self.fmt_user_label = QTextCharFormat()
        self.fmt_user_label.setFontWeight(QFont.Weight.Bold)
        self.fmt_user_label.setForeground(QColor("#1565c0"))

        self.fmt_model_label = QTextCharFormat()
        self.fmt_model_label.setFontWeight(QFont.Weight.Bold)
        self.fmt_model_label.setForeground(QColor("#2e7d32"))

        self.fmt_sys = QTextCharFormat()
        self.fmt_sys.setForeground(QColor("#666666"))

        self.fmt_err = QTextCharFormat()
        self.fmt_err.setForeground(QColor("#b00020"))
        self.fmt_err.setFontWeight(QFont.Weight.Bold)

        # Markdown parser for stream (only **bold**)
        self._md_in_bold = False
        self._md_carry = ""

        # preload models
        self.refresh_models(initial=True)

    # ---- models ----
    def refresh_models(self, initial: bool = False):
        self.model_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)

        def list_models():
            try:
                r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
                r.raise_for_status()
                data = r.json()
                return [m.get("name") for m in data.get("models", []) if m.get("name")], ""
            except Exception as e:
                return [], str(e)

        # without separate QThread: request is lightweight, but you can move it if desired
        models, error = list_models()
        self.on_models_loaded(models, error)

        if not initial and not error:
            self.append_sys("Model list updated.")

    def on_models_loaded(self, models: List[str], error: str):
        self.model_combo.clear()
        if error:
            self.append_err(f"Could not get model list: {error}")
        elif not models:
            self.append_err("No models found. Make sure Ollama is running and models are installed (e.g., `ollama pull llama3:8b`).")
        else:
            models_sorted = sorted(models)
            self.model_combo.addItems(models_sorted)

            if self._last_selected_model and self._last_selected_model in models_sorted:
                idx = models_sorted.index(self._last_selected_model)
                self.model_combo.setCurrentIndex(idx)

            if not self.chat.toPlainText():
                self.append_sys("Model list loaded. Select a model and ask a question.")

        self.model_combo.setEnabled(True)
        self.refresh_btn.setEnabled(True)

    # ---- output / markdown-bold ----
    def _cursor_to_end(self) -> QTextCursor:
        c = self.chat.textCursor()
        c.movePosition(QTextCursor.MoveOperation.End)
        self.chat.setTextCursor(c)
        return c

    def append_user(self, text: str):
        c = self._cursor_to_end()
        c.insertText("You: ", self.fmt_user_label)
        c.insertBlock()
        c.insertText(text, self.fmt_normal)
        c.insertBlock()  # empty line separator

    def append_ai_header(self):
        c = self._cursor_to_end()
        c.insertText("Model: ", self.fmt_model_label)
        c.insertBlock()
        # reset markdown state
        self._md_in_bold = False
        self._md_carry = ""

    def _insert_text_md(self, text: str):
        """
        Incremental text insertion with **bold** support.
        Handles marker '**' split across chunks.
        """
        s = self._md_carry + text
        self._md_carry = ""

        i = 0
        c = self._cursor_to_end()

        while i < len(s):
            ch = s[i]

            if ch == '*':
                # chunk ended - carry over
                if i == len(s) - 1:
                    self._md_carry = "*"
                    i += 1
                    break
                # check for double star
                if s[i + 1] == '*':
                    # case '***' → '**' toggles mode, and '*' is literal
                    if i + 2 < len(s) and s[i + 2] == '*':
                        self._md_in_bold = not self._md_in_bold
                        i += 2
                        fmt = self.fmt_bold if self._md_in_bold else self.fmt_normal
                        c.insertText("*", fmt)
                        i += 1
                        continue
                    else:
                        self._md_in_bold = not self._md_in_bold
                        i += 2
                        continue
                else:
                    # single * is literal
                    fmt = self.fmt_bold if self._md_in_bold else self.fmt_normal
                    c.insertText("*", fmt)
                    i += 1
                    continue

            # normal segment until next '*'
            j = s.find('*', i)
            if j == -1:
                chunk = s[i:]
                i = len(s)
            else:
                chunk = s[i:j]
                i = j

            fmt = self.fmt_bold if self._md_in_bold else self.fmt_normal
            c.insertText(chunk, fmt)

    def append_ai_stream_chunk(self, text: str):
        self._insert_text_md(text)
        self.chat.ensureCursorVisible()

    def append_sys(self, text: str):
        c = self._cursor_to_end()
        c.insertText(text, self.fmt_sys)
        c.insertBlock()

    def append_err(self, text: str):
        c = self._cursor_to_end()
        c.insertText(f"Error: {text}", self.fmt_err)
        c.insertBlock()

    # ---- chat mode / history management ----
    def on_toggle_chat_mode(self, state: int):
        self._chat_mode = self.chat_mode_cb.isChecked()
        self.clear_history_btn.setEnabled(self._chat_mode)
        # save setting immediately
        save_config(self._current_config())
        # user hint
        mode_text = "enabled (model sees history)" if self._chat_mode else "disabled (stateless)"
        self.append_sys(f"History mode {mode_text}.")

    def on_clear_history(self):
        self._history.clear()
        self.append_sys("Message history cleared.")

    # ---- sending ----
    def on_send(self):
        if self._is_generating:
            return
        prompt = self.input.text().strip()
        if not prompt:
            return
        model = self.model_combo.currentText().strip()
        if not model:
            QMessageBox.warning(self, "No model", "Select a model.")
            return

        options = {
            "temperature": float(self.temp_spin.value()),
            "top_p": float(self.top_p_spin.value())
        }

        self._last_selected_model = model

        # UI output
        self.append_user(prompt)
        self.append_ai_header()

        # prepare request depending on mode
        use_chat = self._chat_mode
        messages = None
        if use_chat:
            # history support: add user message
            self._history.append({"role": "user", "content": prompt})
            messages = list(self._history)  # copy at request time

        # lock elements (input stays active!)
        self.input.clear()
        self.input.setFocus()  # keep typing immediately
        self.send_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.chat_mode_cb.setEnabled(False)
        self.clear_history_btn.setEnabled(False)
        self._is_generating = True

        # launch worker
        self.gen_thread = QThread(self)
        self.gen_worker = GenerateWorker(
            model=model,
            options=options,
            use_chat=use_chat,
            prompt=prompt if not use_chat else None,
            messages=messages if use_chat else None
        )
        self.gen_worker.moveToThread(self.gen_thread)
        self.gen_thread.started.connect(self.gen_worker.run)
        self.gen_worker.chunk.connect(self.on_chunk)
        self.gen_worker.finished.connect(self.on_finished)
        self.gen_worker.failed.connect(self.on_failed)
        self.gen_worker.finished.connect(self.cleanup_gen)
        self.gen_worker.failed.connect(self.cleanup_gen)
        self.gen_thread.start()

    def on_chunk(self, text: str):
        self.append_ai_stream_chunk(text)

    def on_finished(self, full_text: str):
        # finished output — add line breaks
        c = self._cursor_to_end()
        c.insertBlock()
        c.insertBlock()

        # if chat mode was on — add assistant message to history
        if self._chat_mode:
            self._history.append({"role": "assistant", "content": full_text})

    def on_failed(self, err: str):
        self.append_err(err)

    def cleanup_gen(self, *_):
        self._is_generating = False
        # re-enable controls except input (which was never disabled)
        self.send_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.chat_mode_cb.setEnabled(True)
        self.clear_history_btn.setEnabled(self._chat_mode)

        # keep focus in the input field
        self.input.setFocus()

        if self.gen_thread:
            self.gen_thread.quit()
            self.gen_thread.wait(2000)
            self.gen_thread = None
            self.gen_worker = None

    # ---- saving settings ----
    def _current_config(self) -> dict:
        return {
            "temperature": float(self.temp_spin.value()),
            "top_p": float(self.top_p_spin.value()),
            "last_model": self._last_selected_model or self.model_combo.currentText().strip() or "",
            "chat_mode": bool(self._chat_mode),
        }

    def closeEvent(self, event):
        save_config(self._current_config())
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    w = ChatWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
