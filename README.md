# Equipment Maintenance Assistant (local, Ollama edition)

A RAG chat app for maintenance teams that runs entirely on your machine: no
API key, no cloud calls. Upload equipment manuals and maintenance logs, then
ask troubleshooting questions in a multi-turn chat. Answers are built only
from your documents, put safety warnings first, say so when the manuals don't
cover something, and cite the file and page for every passage used.

- **UI:** Streamlit chat with session-state memory
- **Ingestion:** pypdf, then `RecursiveCharacterTextSplitter` (700 tokens, 100 overlap)
- **Index:** LangChain FAISS, in memory
- **Embeddings & LLM:** any models already pulled into your local Ollama server
  (the sidebar lists them and lets you pick which to use)
- **Chat flow:** follow-up questions are rewritten into standalone queries using
  the chat history, then retrieved with MMR (relevant but not repetitive)

## Project layout

```
app.py
requirements.txt
.streamlit/config.toml            theme and upload limit
.streamlit/secrets.toml.example   template for OLLAMA_HOST (only needed if
                                   Ollama isn't on localhost:11434)
.vscode/                          VS Code interpreter, debug config, extensions
tests/                            sample manual/log + test scripts (optional)
```

## 1. Install Ollama on Debian

Skip this if you already have it (you mentioned 0.30.0, which is fine).

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &          # or: sudo systemctl enable --now ollama
```

Pull a chat model and an embedding model:

```bash
ollama pull llama3.1          # ~4.7 GB; swap for a smaller model if RAM is tight
ollama pull nomic-embed-text  # ~270 MB embedding model
```

Any models already on your system work too -- the app's sidebar reads
`ollama list` and lets you choose per session.

## 2. Open the project in VS Code

```bash
code /path/to/this/folder
```

Install the recommended extensions if prompted (Python + Pylance). Then, in a
VS Code terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`.vscode/settings.json` points VS Code at `.venv` automatically once it
exists, so reopen the terminal (or run **Python: Select Interpreter**) if it
doesn't pick it up right away.

## 3. Run it

```bash
streamlit run app.py
```

or press **F5** in VS Code (uses `.vscode/launch.json`). It opens at
`http://localhost:8501`.

In the sidebar: confirm the Ollama host (defaults to
`http://localhost:11434`), pick a chat model and an embedding model from what
you've pulled, upload a PDF manual or a `.txt`/`.log`/`.csv`/`.md` log, and
choose **Ingest documents**. Then ask a question in the chat box.

If your Ollama server runs elsewhere (another machine, WSL, Docker), copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set
`OLLAMA_HOST` there instead of using the default.

## Good to know

- **The index lives in each browser session's memory.** Restarting the app
  clears it, so documents must be uploaded again.
- **Don't switch embedding models mid-session.** All the vectors in an index
  come from one embedding model; the app detects a mismatch and asks you to
  choose **Remove all** before re-ingesting with a different one.
- **Everything is local**, so there is no per-request cost, but answer speed
  depends on your CPU/GPU and the model size. A smaller chat model (e.g.
  `llama3.2` or `phi3`) answers faster on modest hardware.
- **Scanned PDFs need OCR first.** The app reads embedded text only, and tells
  you when a PDF has none.
- **Plain-text logs have no page numbers**, so their citations show "text file".
- **Settings** such as chunk size and history length are constants at the top
  of `app.py`.
- The assistant is a reference aid. Technicians should confirm procedures
  against the original manual and follow site lockout/tagout rules.

## Optional: run the test suite

`tests/` has a generated sample manual/log and two scripts that exercise the
app's logic and UI with fake models (no Ollama server needed):

```bash
pip install reportlab  # only needed to regenerate tests/HX-400_manual.pdf
python tests/test_core.py
python tests/test_ui.py
```
# manufacturiing-equipment-maintence-query-agent
