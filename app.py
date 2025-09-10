import sys
import json
import re
import html
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests
from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt, QSize, QTimer
from PyQt6.QtGui import (
    QTextCursor, QTextCharFormat, QFont, QColor,
    QKeyEvent, QGuiApplication, QTextOption
)
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextBrowser,
    QPushButton, QComboBox, QLabel, QMessageBox, QDoubleSpinBox,
    QCheckBox, QTextEdit
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
                    "options": self.options,
                    "session": ""
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

                        if "response" in obj:
                            text = obj["response"]
                            compiled.append(text)
                            self.chunk.emit(text)
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
                    "messages": self.messages,
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

                        if "message" in obj and obj["message"] and "content" in obj["message"]:
                            text = obj["message"]["content"]
                            compiled.append(text)
                            self.chunk.emit(text)
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


# ---------- Tiny syntax highlighters (no deps) ----------

PY_KEYWORDS = {
    "False","None","True","and","as","assert","async","await","break","class","continue",
    "def","del","elif","else","except","finally","for","from","global","if","import",
    "in","is","lambda","nonlocal","not","or","pass","raise","return","try","while","with","yield"
}

BASH_KW = {
    "if","then","else","elif","fi","for","in","do","done","case","esac","while","until",
    "function","select","time","coproc","return","break","continue","echo","export","local"
}


def _html_span(token: str, style: str) -> str:
    return f'<span style="{style}">{html.escape(token)}</span>'


def highlight_python(code: str) -> str:
    """Very small Python highlighter: strings, comments, numbers, keywords."""
    out: List[str] = []
    i = 0
    n = len(code)

    def take_string(quote: str, triple: bool) -> str:
        nonlocal i
        start = i
        if triple:
            i += 3
            q = quote*3
            while i < n and code[i:i+3] != q:
                i += 1
            i = min(n, i+3)
        else:
            i += 1
            esc = False
            while i < n:
                ch = code[i]
                if esc:
                    esc = False
                    i += 1
                    continue
                if ch == '\\':
                    esc = True
                    i += 1
                    continue
                if ch == quote:
                    i += 1
                    break
                i += 1
        return code[start:i]

    while i < n:
        ch = code[i]

        # comments
        if ch == "#":
            j = code.find("\n", i)
            if j == -1:
                j = n
            out.append(_html_span(code[i:j], "color:#6a9955"))  # green
            i = j
            continue

        # triple strings
        if code.startswith("'''", i) or code.startswith('"""', i):
            q = code[i]
            s = take_string(q, True)
            out.append(_html_span(s, "color:#ce9178"))  # string
            continue

        # single/double strings
        if ch in ("'", '"'):
            s = take_string(ch, False)
            out.append(_html_span(s, "color:#ce9178"))
            continue

        # numbers
        if ch.isdigit():
            j = i + 1
            while j < n and (code[j].isdigit() or code[j] in ".eE_-"):
                j += 1
            out.append(_html_span(code[i:j], "color:#b5cea8"))  # number
            i = j
            continue

        # identifiers / keywords
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (code[j].isalnum() or code[j] == "_"):
                j += 1
            word = code[i:j]
            if word in PY_KEYWORDS:
                out.append(_html_span(word, "color:#c586c0; font-weight:600"))  # keyword
            else:
                out.append(html.escape(word))
            i = j
            continue

        # everything else (operators, spaces, punctuation)
        out.append(html.escape(ch))
        i += 1

    return "".join(out)


def highlight_json(code: str) -> str:
    out = []
    i, n = 0, len(code)

    def take_string():
        nonlocal i
        start = i
        i += 1
        esc = False
        while i < n:
            ch = code[i]
            if esc:
                esc = False
                i += 1
                continue
            if ch == '\\':
                esc = True
                i += 1
                continue
            if ch == '"':
                i += 1
                break
            i += 1
        return code[start:i]

    while i < n:
        ch = code[i]
        if ch == '"':
            s = take_string()
            out.append(_html_span(s, "color:#ce9178"))  # string
            continue
        if ch.isdigit() or (ch == '-' and i + 1 < n and code[i+1].isdigit()):
            j = i + 1
            while j < n and (code[j].isdigit() or code[j] in ".eE+-"):
                j += 1
            out.append(_html_span(code[i:j], "color:#b5cea8"))  # number
            i = j
            continue
        # literals
        if code.startswith("true", i) or code.startswith("false", i) or code.startswith("null", i):
            if code.startswith("true", i):
                lit = "true"
            elif code.startswith("false", i):
                lit = "false"
            else:
                lit = "null"
            out.append(_html_span(lit, "color:#569cd6; font-weight:600"))
            i += len(lit)
            continue
        out.append(html.escape(ch))
        i += 1

    return "".join(out)


def highlight_bash(code: str) -> str:
    out = []
    for line in code.splitlines(keepends=True):
        # comment line-part
        if "#" in line:
            before, hash_, after = line.partition("#")
            out.append(_highlight_bash_segment(before))
            out.append(_html_span("#" + after.rstrip("\n"), "color:#6a9955"))
            out.append(html.escape("\n") if line.endswith("\n") else "")
        else:
            out.append(_highlight_bash_segment(line))
    return "".join(out)


def _highlight_bash_segment(seg: str) -> str:
    out = []
    i, n = 0, len(seg)

    def take_string(q):
        nonlocal i
        start = i
        i += 1
        esc = False
        while i < n:
            ch = seg[i]
            if esc:
                esc = False
                i += 1
                continue
            if ch == '\\':
                esc = True
                i += 1
                continue
            if ch == q:
                i += 1
                break
            i += 1
        return seg[start:i]

    while i < n:
        ch = seg[i]
        if ch in ("'", '"'):
            s = take_string(ch)
            out.append(_html_span(s, "color:#ce9178"))
            continue
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (seg[j].isalnum() or seg[j] in "_-"):
                j += 1
            word = seg[i:j]
            if word in BASH_KW:
                out.append(_html_span(word, "color:#c586c0; font-weight:600"))
            else:
                out.append(html.escape(word))
            i = j
            continue
        if ch.isdigit():
            j = i + 1
            while j < n and (seg[j].isdigit() or seg[j] in "."):
                j += 1
            out.append(_html_span(seg[i:j], "color:#b5cea8"))
            i = j
            continue
        out.append(html.escape(ch))
        i += 1
    return "".join(out)


def highlight_code(lang: str, code: str) -> str:
    lang = (lang or "").lower()
    if lang in ("py", "python"):
        return highlight_python(code)
    if lang in ("json",):
        return highlight_json(code)
    if lang in ("bash", "sh", "shell"):
        return highlight_bash(code)
    # default: no highlight, just escape
    return html.escape(code)


# ---------- Helpers: minimal markdown → HTML with copy buttons + highlight ----------

CODE_BLOCK_RE = re.compile(
    r"```([a-zA-Z0-9_+.-]+)?\s*\n(.*?)\n```",
    re.DOTALL
)
BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")


def md_to_html_with_copy(md_text: str, code_store: Dict[str, str]) -> str:
    safe = md_text.replace("\r\n", "\n")

    code_id_seq = 1
    parts = []
    last = 0

    for m in CODE_BLOCK_RE.finditer(safe):
        before = safe[last:m.start()]
        parts.append(html.escape(before).replace("\n", "<br>"))

        lang = m.group(1) or ""
        code = m.group(2)
        code_id = f"code-{code_id_seq}"
        code_id_seq += 1
        code_store[code_id] = code

        highlighted = highlight_code(lang, code)
        lang_label = f'<span style="font-weight:600; opacity:0.7">{html.escape(lang)}</span>' if lang else ""

        parts.append(
            (
                '<div style="border:1px solid #ddd;border-radius:6px;margin:4px 0;">'
                '<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 8px;background:#fafafa;border-bottom:1px solid #eee;font-size:12px;">'
                f'<div>{lang_label}</div>'
                f'<div><a href="copy://{code_id}" style="text-decoration:none;font-weight:600;">Copy</a></div>'
                '</div>'
                f'<pre style="margin:0;padding:6px 8px;font-family:Consolas,\'Courier New\',monospace;font-size:12.5px;white-space:pre-wrap;background:#f5f5f5;"><code>{highlighted}</code></pre>'
                '</div>'
            )
        )

        last = m.end()

    tail = safe[last:]
    parts.append(html.escape(tail).replace("\n", "<br>"))

    html_text = "".join(parts)

    # Поддержка **жирного**
    html_text = BOLD_RE.sub(r"<b>\1</b>", html_text)
    # Поддержка `inline code`
    html_text = INLINE_CODE_RE.sub(
        r'<code style="background:#f5f5f5;padding:2px 4px;border-radius:4px;font-family:Consolas,\'Courier New\',monospace;font-size:12px;">\1</code>',
        html_text,
    )

    return html_text


# ---------- Expanding multiline input ----------

class ExpandingTextEdit(QTextEdit):
    """
    Auto-expands vertically up to max_lines. Enter = send (via callback),
    Shift+Enter inserts newline.
    Counts visual (wrapped) lines, so it grows both on paste and on typing.
    """
    def __init__(self, send_callback, max_lines=10, parent=None):
        super().__init__(parent)
        self._send_callback = send_callback
        self._max_lines = max_lines
        self._line_height = self.fontMetrics().lineSpacing()

        self.setAcceptRichText(False)
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.setPlaceholderText('Type your prompt. Enter: send, Shift+Enter: newline…')
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.textChanged.connect(self._adjust_height)
        self._adjust_height()

    def sizeHint(self) -> QSize:
        s = super().sizeHint()
        margins = self.contentsMargins()
        h = self._line_height + margins.top() + margins.bottom() + 8
        return QSize(s.width(), h)

    # --- key handling ---
    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                return super().keyPressEvent(e)  # newline
            if callable(self._send_callback):
                self._send_callback()  # send
            return
        return super().keyPressEvent(e)

    # --- sizing helpers ---
    def _visual_line_count(self) -> int:
        """Count wrapped (visual) lines across all blocks."""
        doc = self.document()
        # ensure layout is up-to-date
        doc.adjustSize()
        lines = 0
        block = doc.begin()
        while block.isValid():
            lay = block.layout()
            if lay is not None:
                lines += max(1, lay.lineCount())
            else:
                lines += 1
            block = block.next()
        return max(1, lines)

    def _cap_height_for_lines(self, lines: int) -> int:
        margins = self.contentsMargins()
        frame = int(self.frameWidth()) * 2
        pad = 8
        return int(lines * self._line_height + margins.top() + margins.bottom() + frame + pad)

    def _adjust_height(self):
        lines = self._visual_line_count()
        if lines <= self._max_lines:
            h = self._cap_height_for_lines(lines)
            # grow/shrink freely under the cap
            self.setMinimumHeight(h)
            self.setMaximumHeight(h)
        else:
            # fix height at the cap and let the scrollbar appear
            cap_h = self._cap_height_for_lines(self._max_lines)
            self.setMinimumHeight(cap_h)
            self.setMaximumHeight(cap_h)

    def resizeEvent(self, ev):
        """Re-flow on width changes so wrapping-based height updates too."""
        super().resizeEvent(ev)
        # layout changes after resize; update height next event loop tick
        QTimer.singleShot(0, self._adjust_height)


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

        # heavy render toggle (persisted)
        self._heavy_render: bool = bool(self._cfg.get("heavy_render", True))

        # For replacing the last streamed answer with rich HTML (code copy + highlight)
        self._answer_start_pos: Optional[int] = None
        self._current_answer_raw: List[str] = []
        self._code_blocks_store: Dict[str, str] = {}  # id -> raw code

        # ---- UI ----
        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(False)
        self.chat.anchorClicked.connect(self.on_anchor_clicked)
        self.chat.setReadOnly(True)

        self.input = ExpandingTextEdit(send_callback=self.on_send)
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

        # NEW: Heavy render toggle
        self.heavy_render_cb = QCheckBox("Heavy render (HTML + highlight)")
        self.heavy_render_cb.setChecked(self._heavy_render)
        self.heavy_render_cb.stateChanged.connect(self.on_toggle_heavy_render)

        # top panel
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Model:"))
        top_bar.addWidget(self.model_combo, stretch=1)
        top_bar.addWidget(self.refresh_btn)
        top_bar.addSpacing(12)
        top_bar.addWidget(self.chat_mode_cb)
        top_bar.addWidget(self.clear_history_btn)
        top_bar.addSpacing(12)
        top_bar.addWidget(self.heavy_render_cb)

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

        # ---- Text formats (used for streaming preview) ----
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

        # Markdown (only **bold**) for streaming preview
        self._md_in_bold = False
        self._md_carry = ""

        # preload models
        self.refresh_models(initial=True)

        # (optional) speed-ups for long sessions
        self.chat.document().setUndoRedoEnabled(False)
        self.chat.document().setMaximumBlockCount(3000)

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

    # ---- toggles ----
    def on_toggle_chat_mode(self, state: int):
        self._chat_mode = self.chat_mode_cb.isChecked()
        self.clear_history_btn.setEnabled(self._chat_mode)
        save_config(self._current_config())
        mode_text = "enabled (model sees history)" if self._chat_mode else "disabled (stateless)"
        self.append_sys(f"History mode {mode_text}.")

    def on_toggle_heavy_render(self, state: int):
        self._heavy_render = self.heavy_render_cb.isChecked()
        save_config(self._current_config())
        self.append_sys(f"Heavy render {'ON' if self._heavy_render else 'OFF'}.")

    def on_clear_history(self):
        self._history.clear()
        self.append_sys("Message history cleared.")

    # ---- output / streaming preview (simple **bold**) ----
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
        c.insertBlock()

    def append_ai_header(self):
        c = self._cursor_to_end()
        c.insertText("Model: ", self.fmt_model_label)
        c.insertBlock()
        if self._heavy_render:
            self._answer_start_pos = self.chat.textCursor().position()
            self._current_answer_raw = []
            self._code_blocks_store.clear()
        else:
            self._answer_start_pos = None
            self._current_answer_raw = []
            self._code_blocks_store.clear()
        # reset markdown state for streaming preview
        self._md_in_bold = False
        self._md_carry = ""

    def _insert_text_md(self, text: str):
        """
        Incremental text insertion with **bold** support (preview only).
        """
        s = self._md_carry + text
        self._md_carry = ""

        i = 0
        c = self._cursor_to_end()

        while i < len(s):
            ch = s[i]
            if ch == '*':
                if i == len(s) - 1:
                    self._md_carry = "*"
                    i += 1
                    break
                if s[i + 1] == '*':
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
                    fmt = self.fmt_bold if self._md_in_bold else self.fmt_normal
                    c.insertText("*", fmt)
                    i += 1
                    continue

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
        # preview in plain rich text (bold only)
        self._insert_text_md(text)
        if self._heavy_render:
            self._current_answer_raw.append(text)
        self.chat.ensureCursorVisible()

    def append_sys(self, text: str):
        c = self._cursor_to_end()
        c.insertText(text, self.fmt_sys)
        c.insertBlock()

    def append_err(self, text: str):
        c = self._cursor_to_end()
        c.insertText(f"Error: {text}", self.fmt_err)
        c.insertBlock()

    # ---- anchor handler (copy://<id>) ----
    def on_anchor_clicked(self, url):
        qurl = url.toString()
        if qurl.startswith("copy://"):
            code_id = qurl.split("copy://", 1)[1]
            code = self._code_blocks_store.get(code_id)
            if code is not None:
                QGuiApplication.clipboard().setText(code)
                self.append_sys("Code copied to clipboard.")
        else:
            if qurl.startswith("http://") or qurl.startswith("https://"):
                self.chat.setOpenExternalLinks(True)
                self.chat.setSource(url)
                self.chat.setOpenExternalLinks(False)

    # ---- sending ----
    def on_send(self):
        if self._is_generating:
            return
        prompt = self.input.toPlainText().strip()
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
            self._history.append({"role": "user", "content": prompt})
            messages = list(self._history)

        # lock elements (input stays active but cleared)
        self.input.clear()
        self.input.setFocus()
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

    def _replace_last_answer_with_html(self, full_text: str):
        """
        Replace the streamed plain text with HTML that includes
        syntax highlighting and 'Copy' buttons for code blocks.
        """
        if self._answer_start_pos is None:
            return

        html_text = md_to_html_with_copy(full_text, self._code_blocks_store)

        self.chat.setUpdatesEnabled(False)
        c = self.chat.textCursor()
        c.beginEditBlock()
        c.setPosition(self._answer_start_pos)
        c.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        c.removeSelectedText()
        c.insertHtml(html_text)
        c.insertBlock()
        c.endEditBlock()
        self.chat.setTextCursor(c)
        self.chat.setUpdatesEnabled(True)
        self.chat.ensureCursorVisible()

        self._answer_start_pos = None
        self._current_answer_raw = []

    def on_finished(self, full_text: str):
        if self._heavy_render:
            self._replace_last_answer_with_html("".join(self._current_answer_raw) if self._current_answer_raw else full_text)
        else:
            c = self._cursor_to_end()
            c.insertBlock()
            c.insertBlock()
            self._answer_start_pos = None
            self._current_answer_raw = []

        if self._chat_mode:
            self._history.append({"role": "assistant", "content": full_text})

    def on_failed(self, err: str):
        self.append_err(err)

    def cleanup_gen(self, *_):
        self._is_generating = False
        self.send_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.chat_mode_cb.setEnabled(True)
        self.clear_history_btn.setEnabled(self._chat_mode)
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
            "heavy_render": bool(self._heavy_render),
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
