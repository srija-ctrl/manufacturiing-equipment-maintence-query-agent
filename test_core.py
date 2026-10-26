import itertools, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import app
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

# ---- parsing (unchanged logic, sanity check) --------------------------------
pdf = open("tests/HX-400_manual.pdf", "rb").read()
info, chunks = app.prepare_file("HX-400_manual.pdf", pdf)
assert info["pages"] == 3 and info["chunks"] >= 3
info_t, chunks_t = app.prepare_file("press_log.txt", open("tests/press_log.txt", "rb").read())
assert info_t["kind"] == "text"
print("1. parsing OK:", info, info_t)

for bad_name, bad_data in [("empty.txt", b"  \n"), ("junk.pdf", b"not a pdf")]:
    try:
        app.prepare_file(bad_name, bad_data); raise SystemExit("expected failure")
    except Exception as e:
        print("   expected error:", bad_name, type(e).__name__)

# ---- config / defaults -------------------------------------------------------
cfg = app.OllamaConfig(host="http://localhost:11434", chat_model="llama3.1", embed_model="nomic-embed-text")
print("2. pick_default chat:", app.pick_default(["mistral:latest", "llama3.1:8b"], app.PREFERRED_CHAT_MODELS))
assert app.pick_default(["mistral:latest", "llama3.1:8b"], app.PREFERRED_CHAT_MODELS) == "llama3.1:8b"
assert app.pick_default(["foo", "bar"], app.PREFERRED_CHAT_MODELS) == "foo"  # falls back to first
assert app.pick_default([], app.PREFERRED_CHAT_MODELS) is None

# ---- indexing + retrieval (fake embeddings) --------------------------------
emb = DeterministicFakeEmbedding(size=64)
progress = []
store = app.build_index(chunks + chunks_t, emb, on_progress=lambda d, t: progress.append((d, t)))
print("3. index progress:", progress[-1])
docs = app.retrieve(store, "E-217 low hydraulic pressure", 4)
assert len(docs) == 4
print("   retrieved:", [(d.metadata["source"], d.metadata["page"]) for d in docs])

# batching across multiple embed calls
app.EMBED_BATCH_SIZE = 2
store2 = app.build_index(chunks, emb)
assert len(store2.docstore._dict) == len(chunks)
app.EMBED_BATCH_SIZE = 32

# ---- conversational chain (fake chat model) --------------------------------
calls = []
def fake_llm(config, temperature=0.1):
    calls.append((config.chat_model, temperature))
    reply = "Rewritten: HX-400 E-217 filter F2 replacement" if temperature == 0.0 else \
            "> ⚠️ **SAFETY WARNING:** Lock out first. Replace filter F2 [1]."
    return GenericFakeChatModel(messages=itertools.cycle([AIMessage(reply)]))
app.build_llm = fake_llm

turn = app.answer_turn("How do I fix E-217?", [], store, cfg, 4)
text = "".join(turn.stream)
print("4. turn1 query:", repr(turn.query), "| calls:", calls)
assert turn.query == "How do I fix E-217?" and calls == [("llama3.1", 0.1)] and "SAFETY WARNING" in text

history = [{"role": "user", "content": "How do I fix E-217?"}, {"role": "assistant", "content": text}]
turn2 = app.answer_turn("and how often should I change it?", history, store, cfg, 4)
"".join(turn2.stream)
print("5. turn2 query:", repr(turn2.query), "| calls:", calls[-2:])
assert turn2.query.startswith("Rewritten") and calls[-2:] == [("llama3.1", 0.0), ("llama3.1", 0.1)]
assert turn2.sources and {"n", "source", "page", "text"} <= set(turn2.sources[0])

# ---- error mapping ------------------------------------------------------------
import httpx, ollama
req = httpx.Request("GET", "http://localhost:11434/api/tags")
print("6. conn error :", app.describe_error(httpx.ConnectError("refused"), cfg))
print("   conn error2:", app.describe_error(ConnectionError("Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download"), cfg))
print("   timeout    :", app.describe_error(httpx.TimeoutException("slow"), cfg))
missing = ollama.ResponseError('{"error": "model \\"nomic-embed-text\\" not found, try pulling it first"}', 404)
print("   not found  :", app.describe_error(missing, cfg))
print("   other      :", app.describe_error(ValueError("boom")))
assert "ollama pull" in app.describe_error(missing, cfg)
assert "ollama serve" in app.describe_error(httpx.ConnectError("x"), cfg)

# ---- real constructors accept our kwargs ------------------------------------
import importlib; importlib.reload(app)
e = app.build_embeddings(cfg); l = app.build_llm(cfg)
print("7. constructors ok:", e.model, l.model, l.base_url)
print("ALL CORE TESTS PASSED")
