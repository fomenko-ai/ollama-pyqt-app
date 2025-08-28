# Ollama PyQt App

A simple desktop application built with **PyQt6** that provides a graphical interface for interacting with [Ollama](https://ollama.ai/).  
It supports both **stateless text generation** (`/api/generate`, similar to `ollama run`) and **chat with history** (`/api/chat`).  

The app features streaming responses, model selection, adjustable generation parameters (temperature, top_p), and persistent settings.

<img width="1340" height="952" alt="image" src="https://github.com/user-attachments/assets/95433bac-c562-44b5-a2c7-299c40496231" />

---

## Features

- **PyQt6 GUI** with input box, output window, and model selection dropdown
- **Streaming responses** from the Ollama API, displayed in real time
- **Two modes**:
  - **Stateless mode** (`/api/generate`) – every request is independent
  - **Chat mode** (`/api/chat`) – keeps conversation history
- **Configurable parameters**: `temperature`, `top_p`
- **Persistent settings** saved in `~/.ollama_pyqt/config.json`
- **Markdown-like bold support** in responses (`**bold**`)
- **Refreshable model list** (via `/api/tags`)
- **Threaded workers** so the UI stays responsive

---

## Requirements

- **Python 3.9+**
- **Ollama** installed and running locally (`http://127.0.0.1:11434`)

---

## Installation

1. Ensure [Ollama](https://ollama.ai) is installed and running.
   - Pull at least one model, e.g.:
     ```bash
     ollama pull gemma2:9b
     ```

2. Create a virtual environment.

3. Add required dependencies.

---

## Usage

1. Start (or ensure) the Ollama server is running and that you have at least one local model.
2. Run the application:
   ```bash
   uv run app.py
   ```
3. In the GUI:
   - Select a model in the **Model** dropdown (use **↻** to refresh the list)
   - Type your prompt and press **Enter** or click **Send**
   - Toggle **History (Chat)** to switch between stateless and chat modes
   - Adjust **temperature** and **top_p** in **Settings** if needed
   - Use **Clear history** to reset the conversation when in chat mode

---

## Configuration

The application stores user preferences in:

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

To change the Ollama endpoint, edit the `OLLAMA_HOST` constant near the top of the source file.

---

## Project Structure

- `app.py` — entry point with the full PyQt6 app and worker logic
  - `GenerateWorker` — runs Ollama requests on a background `QThread`
  - `ChatWindow` — main window: UI, state management, handlers
- `~/.ollama_pyqt/config.json` — persisted user settings

---

## Troubleshooting

- **“No models found.”** Ensure Ollama is running and at least one model is installed:
  ```bash
  ollama pull gemma2:9b
  ```
  Then click **↻** to refresh the model list.
- **Cannot connect to Ollama.** Verify the endpoint (`http://127.0.0.1:11434`) and that `ollama serve` is running (on Linux), or the Ollama service is active.
- **Markdown rendering.** Only `**bold**` is supported in-stream.

---

## License

This project’s source code is licensed under the **MIT License**.  
However, it depends on **PyQt6**, which is licensed under the **GNU GPL v3** (or a commercial license from Riverbank Computing).  

➡️ This means:  
- You are free to use and modify this code under the terms of MIT.  
- If you **distribute** the application together with **PyQt6** under GPLv3, the combined work must comply with GPLv3.  
- Alternatively, you may obtain a **commercial PyQt6 license** to distribute the application under different terms.  

For more details, see: [Riverbank PyQt Licensing](https://www.riverbankcomputing.com/commercial/pyqt).
