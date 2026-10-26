"""End-to-end UI test with Streamlit's AppTest; the Ollama server is faked locally."""
import itertools, os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.pop("OLLAMA_HOST", None)

import ollama
import langchain_ollama
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

# ---- fake "server down" first ----------------------------------------------
class DownClient:
    def __init__(self, host=None, **kw): pass
    def list(self): raise ConnectionError(
        "Failed to connect to Ollama. Please check that Ollama is downloaded, "
        "running and accessible. https://ollama.com/download")
ollama.Client = DownClient

at = AppTest.from_file("../app.py", default_timeout=60).run()
assert not at.exception, at.exception
print("1. server down -> error:", at.sidebar.error[0].value[:70])
print("               -> info :", at.sidebar.info[0].value[:60])
assert at.chat_input[0].disabled
assert [b for b in at.button if b.label == "Retry connection"]

# ---- fake server up, with two installed models -----------------------------
class Model:
    def __init__(self, name): self.model = name
class ListResp:
    def __init__(self, names): self.models = [Model(n) for n in names]
class UpClient:
    def __init__(self, host=None, **kw): pass
    def list(self): return ListResp(["mistral:latest", "llama3.1:8b", "nomic-embed-text:latest"])
ollama.Client = UpClient

def fake_embeddings(**kw): return DeterministicFakeEmbedding(size=64)
def fake_chat(**kw):
    text = "Rewritten query about filter F2 interval" if kw.get("temperature") == 0.0 else \
           "> ⚠️ **SAFETY WARNING:** Lock out and tag out first.\n\n1. Replace filter F2 every 500 hours [1]."
    return GenericFakeChatModel(messages=itertools.cycle([AIMessage(text)]))
langchain_ollama.OllamaEmbeddings = fake_embeddings
langchain_ollama.ChatOllama = fake_chat

pdf = open("tests/HX-400_manual.pdf", "rb").read()
log = open("tests/press_log.txt", "rb").read()

at = AppTest.from_file("../app.py", default_timeout=60).run()
assert not at.exception, at.exception
print("2. models found -> chat select:", at.sidebar.selectbox[0].value, "| embed select:", at.sidebar.selectbox[1].value)
assert at.sidebar.selectbox[0].value == "llama3.1:8b"       # preferred chat default
assert at.sidebar.selectbox[1].value == "nomic-embed-text:latest"  # preferred embed default
assert at.chat_input[0].disabled  # no docs yet

at.file_uploader[0].set_value([("HX-400_manual.pdf", pdf, "application/pdf"),
                               ("press_log.txt", log, "text/plain"),
                               ("bad.pdf", b"nope", "application/pdf")]).run()
ingest = [b for b in at.button if b.label == "Ingest documents"][0]
assert not ingest.disabled
ingest.click().run()
assert not at.exception, at.exception
print("3. ingest -> notices:", [m.value for m in at.sidebar.success], [m.value for m in at.sidebar.error])
kb = at.session_state["kb_files"]
assert len(kb) == 2 and at.session_state["vectorstore"] is not None
assert at.session_state["kb_embed_model"] == "nomic-embed-text:latest"
assert not at.chat_input[0].disabled

# chat turn 1 + follow-up
at.chat_input[0].set_value("How do I fix fault E-217?").run()
assert not at.exception, at.exception
msgs = at.session_state["messages"]
print("4. chat1 -> sources:", len(msgs[1]["sources"]), "| safety in answer:", "SAFETY WARNING" in msgs[1]["content"])
assert "SAFETY WARNING" in msgs[1]["content"] and msgs[1]["query"] is None

at.chat_input[0].set_value("and how often do I replace it?").run()
assert not at.exception, at.exception
msgs = at.session_state["messages"]
print("5. chat2 -> rewritten query:", msgs[3]["query"])
assert msgs[3]["query"].startswith("Rewritten")
statuses = [e.label for e in at.status] + [e.label for e in at.expander]
assert any(l.startswith("Sources (") for l in statuses)

# ---- switching embedding model after ingest warns and blocks re-ingest -----
at.sidebar.selectbox[1].set_value("mistral:latest").run()
warn = [w.value for w in at.sidebar.warning]
print("6. embed switch -> warning shown:", bool(warn), warn[:1])
assert warn and "Remove all" in warn[0]
at.file_uploader[0].set_value([("press_log.txt", log, "text/plain")]).run()
ingest2 = [b for b in at.button if b.label == "Ingest documents"][0]
print("   ingest disabled while mismatched:", ingest2.disabled)
assert ingest2.disabled

# remove all clears the mismatch and lets the user start over
[b for b in at.button if b.label == "Remove all"][0].click().run()
assert at.session_state["vectorstore"] is None and at.session_state["kb_embed_model"] is None
print("7. remove all -> cleared OK")

# ---- 404 "model not found" surfaces a pull hint -----------------------------
import httpx
def missing_model(**kw):
    class Missing:
        def embed_documents(self, texts):
            raise ollama.ResponseError('{"error": "model \\"nomic-embed-text\\" not found"}', 404)
    return Missing()
langchain_ollama.OllamaEmbeddings = missing_model
at.sidebar.selectbox[1].set_value("nomic-embed-text:latest").run()
at.file_uploader[0].set_value([("HX-400_manual.pdf", pdf, "application/pdf")]).run()
[b for b in at.button if b.label == "Ingest documents"][0].click().run()
errs = [e.value for e in at.sidebar.error]
print("8. missing model -> error:", errs)
assert at.session_state["vectorstore"] is None
assert any("ollama pull" in e for e in errs)
print("ALL UI TESTS PASSED")
