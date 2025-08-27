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

# ---- Константы ----
OLLAMA_HOST = "http://127.0.0.1:11434"
APP_DIR = Path.home() / ".ollama_pyqt"
CONFIG_PATH = APP_DIR / "config.json"


# ---------- Workers (в потоках), чтобы не блокировать UI ----------

class GenerateWorker(QObject):
    """
    Универсальный воркер:
    - use_chat=False  => /api/generate (без контекста, как `ollama run`)
    - use_chat=True   => /api/chat (с историей сообщений)
    """
    chunk = pyqtSignal(str)         # потоковые фрагменты ответа
    finished = pyqtSignal(str)      # финальный ответ (собранный)
    failed = pyqtSignal(str)        # ошибка

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

                        # стандартное поле стрима generate
                        if "response" in obj:
                            text = obj["response"]
                            compiled.append(text)
                            self.chunk.emit(text)

                        # на всякий случай поддержим message.content если вдруг придёт
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

                        # у chat-стрима инкремент приходит в message.content
                        if "message" in obj and obj["message"] and "content" in obj["message"]:
                            text = obj["message"]["content"]
                            compiled.append(text)
                            self.chunk.emit(text)

                        # некоторые сборки также кладут кусочки в 'response'
                        elif "response" in obj:
                            text = obj["response"]
                            compiled.append(text)
                            self.chunk.emit(text)

                        if obj.get("done"):
                            break

            self.finished.emit("".join(compiled))
        except Exception as e:
            self.failed.emit(str(e))


# ---------- Конфиг ----------

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


# ---------- Основное окно ----------

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
        self._history: List[Dict[str, str]] = []  # для chat-режима

        # ---- UI ----
        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(True)
        self.chat.setReadOnly(True)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Введите запрос и нажмите Enter…")
        self.input.returnPressed.connect(self.on_send)

        self.send_btn = QPushButton("Отправить")
        self.send_btn.clicked.connect(self.on_send)

        # выбор модели
        self.model_combo = QComboBox()
        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setToolTip("Обновить список моделей (ollama list)")
        self.refresh_btn.clicked.connect(self.refresh_models)

        # режим: История (Chat)
        self.chat_mode_cb = QCheckBox("История (Chat)")
        self.chat_mode_cb.setChecked(self._chat_mode)
        self.chat_mode_cb.stateChanged.connect(self.on_toggle_chat_mode)

        self.clear_history_btn = QPushButton("Очистить историю")
        self.clear_history_btn.clicked.connect(self.on_clear_history)
        self.clear_history_btn.setEnabled(self._chat_mode)

        # верхняя панель
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Модель:"))
        top_bar.addWidget(self.model_combo, stretch=1)
        top_bar.addWidget(self.refresh_btn)
        top_bar.addSpacing(12)
        top_bar.addWidget(self.chat_mode_cb)
        top_bar.addWidget(self.clear_history_btn)

        # ---- Настройки (temperature / top_p) ----
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

        # низ
        bottom_bar = QHBoxLayout()
        bottom_bar.addWidget(self.input, stretch=1)
        bottom_bar.addWidget(self.send_btn)

        root = QVBoxLayout(self)
        root.addLayout(top_bar)
        root.addLayout(settings_bar)
        root.addWidget(self.chat, stretch=1)
        root.addLayout(bottom_bar)

        # ---- Форматы текста ----
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

        # Markdown-парсер для потока (только **bold**)
        self._md_in_bold = False
        self._md_carry = ""

        # подгрузка моделей
        self.refresh_models(initial=True)

    # ---- модели ----
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

        # без отдельного QThread: запрос лёгкий, но если хотите – можно как раньше вынести
        models, error = list_models()
        self.on_models_loaded(models, error)

        if not initial and not error:
            self.append_sys("Список моделей обновлён.")

    def on_models_loaded(self, models: List[str], error: str):
        self.model_combo.clear()
        if error:
            self.append_err(f"Не удалось получить список моделей: {error}")
        elif not models:
            self.append_err("Модели не найдены. Убедитесь, что Ollama запущен и модели установлены (например, `ollama pull llama3:8b`).")
        else:
            models_sorted = sorted(models)
            self.model_combo.addItems(models_sorted)

            if self._last_selected_model and self._last_selected_model in models_sorted:
                idx = models_sorted.index(self._last_selected_model)
                self.model_combo.setCurrentIndex(idx)

            if not self.chat.toPlainText():
                self.append_sys("Список моделей загружен. Выберите модель и задайте вопрос.")

        self.model_combo.setEnabled(True)
        self.refresh_btn.setEnabled(True)

    # ---- вывод / markdown-bold ----
    def _cursor_to_end(self) -> QTextCursor:
        c = self.chat.textCursor()
        c.movePosition(QTextCursor.MoveOperation.End)
        self.chat.setTextCursor(c)
        return c

    def append_user(self, text: str):
        c = self._cursor_to_end()
        c.insertText("Вы: ", self.fmt_user_label)
        c.insertBlock()
        c.insertText(text, self.fmt_normal)
        c.insertBlock()  # пустая строка-разделитель

    def append_ai_header(self):
        c = self._cursor_to_end()
        c.insertText("Модель: ", self.fmt_model_label)
        c.insertBlock()
        # сброс markdown состояния
        self._md_in_bold = False
        self._md_carry = ""

    def _insert_text_md(self, text: str):
        """
        Инкрементальная вставка текста с поддержкой **жирного**.
        Учитывает перенос маркера '**' между чанками.
        """
        s = self._md_carry + text
        self._md_carry = ""

        i = 0
        c = self._cursor_to_end()

        while i < len(s):
            ch = s[i]

            if ch == '*':
                # конец чанка – переносим
                if i == len(s) - 1:
                    self._md_carry = "*"
                    i += 1
                    break
                # проверяем двойную звезду
                if s[i + 1] == '*':
                    # случай '***' → '**' переключаем режим, а '*' вставляем как литерал
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
                    # одиночную * считаем литералом
                    fmt = self.fmt_bold if self._md_in_bold else self.fmt_normal
                    c.insertText("*", fmt)
                    i += 1
                    continue

            # обычный сегмент до следующей '*'
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
        c.insertText(f"Ошибка: {text}", self.fmt_err)
        c.insertBlock()

    # ---- управление режимом / историей ----
    def on_toggle_chat_mode(self, state: int):
        self._chat_mode = self.chat_mode_cb.isChecked()
        self.clear_history_btn.setEnabled(self._chat_mode)
        # сохраняем переключение сразу
        save_config(self._current_config())
        # подсказка пользователю
        mode_text = "включён (модель видит историю)" if self._chat_mode else "выключен (без контекста)"
        self.append_sys(f"Режим истории {mode_text}.")

    def on_clear_history(self):
        self._history.clear()
        self.append_sys("История сообщений очищена.")

    # ---- отправка ----
    def on_send(self):
        if self._is_generating:
            return
        prompt = self.input.text().strip()
        if not prompt:
            return
        model = self.model_combo.currentText().strip()
        if not model:
            QMessageBox.warning(self, "Нет модели", "Выберите модель.")
            return

        options = {
            "temperature": float(self.temp_spin.value()),
            "top_p": float(self.top_p_spin.value())
        }

        self._last_selected_model = model

        # UI вывод
        self.append_user(prompt)
        self.append_ai_header()

        # готовим запрос в зависимости от режима
        use_chat = self._chat_mode
        messages = None
        if use_chat:
            # поддержка истории: добавим user сообщение
            self._history.append({"role": "user", "content": prompt})
            messages = list(self._history)  # копия на момент запроса

        # блокируем элементы
        self.input.clear()
        self.input.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.chat_mode_cb.setEnabled(False)
        self.clear_history_btn.setEnabled(False)
        self._is_generating = True

        # запускаем воркер
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
        # завершили вывод — перенос строки + пустая строка
        c = self._cursor_to_end()
        c.insertBlock()
        c.insertBlock()

        # если был чат-режим — добавим assistant в историю
        if self._chat_mode:
            self._history.append({"role": "assistant", "content": full_text})

    def on_failed(self, err: str):
        self.append_err(err)

    def cleanup_gen(self, *_):
        self._is_generating = False
        self.input.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.chat_mode_cb.setEnabled(True)
        self.clear_history_btn.setEnabled(self._chat_mode)
        if self.gen_thread:
            self.gen_thread.quit()
            self.gen_thread.wait(2000)
            self.gen_thread = None
            self.gen_worker = None

    # ---- сохранение настроек ----
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
