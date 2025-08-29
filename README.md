# Ollama PyQt App

A desktop application built with **PyQt6** to interact with [Ollama](https://ollama.ai/) via a clean graphical interface.  
It supports both **stateless text generation** (`/api/generate`, like `ollama run`) and **chat with history** (`/api/chat`).  

The app now features syntax highlighting in code blocks, copy-to-clipboard buttons, an expanding input field, streaming responses, and persistent user settings.

<img width="1340" height="952" alt="image" src="https://github.com/user-attachments/assets/95433bac-c562-44b5-a2c7-299c40496231" />

---

## Features

- **PyQt6 GUI** with input box, output area, model selector, and controls
- **Streaming responses** in real time from the Ollama API
- **Two modes**:
  - **Stateless mode** (`/api/generate`) – every request is independent
  - **Chat mode** (`/api/chat`) – conversation history is preserved
- **Configurable parameters**: `temperature`, `top_p`
- **Persistent settings** in `~/.ollama_pyqt/config.json`
- **Markdown support**:
  - `**bold**` text in responses
  - Inline code ``like this``
  - Fenced code blocks with syntax highlighting (`python`, `json`, `bash`, …)
- **Copy-to-clipboard buttons** for each code block
- **Refreshable model list** (via `/api/tags`)
- **Clearable history** in chat mode
- **Threaded workers** so the UI remains responsive

---

## Requirements

- **Python 3.9+**
- **Ollama** running locally (`http://127.0.0.1:11434`)

---

## Installation

1. Install [Ollama](https://ollama.ai) and ensure it is running.
   - Pull at least one model, e.g.:
     ```bash
     ollama pull gemma2:9b
     ```

2. Clone or copy this repository.

3. Create a virtual environment and install dependencies.


---

## Usage

1. Start Ollama (`ollama serve` or system service). Ensure at least one model is installed.
2. Run the app:
   ```bash
   uv run app.py
   ```
3. In the UI:
   - Select a model (**↻** refreshes the list)
   - Type a prompt and press **Enter** or click **Send**
   - Toggle **History (Chat)** to switch between stateless and chat modes
   - Use **temperature** and **top_p** controls to adjust output style
   - Use **Clear history** in chat mode to reset memory
   - Copy code from responses with the **Copy** button

---

## Configuration

Preferences are stored in:

```
~/.ollama_pyqt/config.json
```

Example:
```json
{
  "temperature": 0.7,
  "top_p": 0.9,
  "last_model": "gemma2:9b",
  "chat_mode": true
}
```

Change `OLLAMA_HOST` in the source (`app.py`) to connect to a custom endpoint.

---

## Project Structure

- `app.py` — full application (PyQt6 GUI, streaming workers, highlighting, copy buttons)
  - `GenerateWorker` — background Ollama requests
  - `ExpandingTextEdit` — auto-resizing input field
  - `ChatWindow` — main window and logic
- `~/.ollama_pyqt/config.json` — persisted settings

---

## Troubleshooting

- **“No models found.”** Run:
  ```bash
  ollama pull gemma2:9b
  ```
  Then click **↻** in the UI.
- **Connection errors.** Verify Ollama is running on `http://127.0.0.1:11434`.
- **Markdown rendering.** Supported: `**bold**`, inline code, fenced code blocks (with highlighting).
- **Copy button missing.** Only available for fenced code blocks (```).

---

## License

Source code is licensed under the **MIT License**.  
This app uses **PyQt6**, licensed under **GNU GPL v3** (or commercially from Riverbank Computing).  

➡️ Practical meaning:  
- You may use and modify this code under MIT.  
- If distributing with PyQt6 (GPLv3), your work must also comply with GPLv3.  
- Or, obtain a commercial PyQt6 license for different terms.  

See [Riverbank PyQt Licensing](https://www.riverbankcomputing.com/commercial/pyqt).