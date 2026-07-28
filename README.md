# RAGM4 — Local Retrieval-Augmented Generation on Apple Silicon

A minimal, fully-local RAG project built with:

| Layer              | Tech                                     |
| ------------------ | ---------------------------------------- |
| Orchestration      | **LlamaIndex**                           |
| LLM engine         | **mlx-lm** (Apple Silicon)               |
| Chat model         | **Qwen2.5-1.5B** (local, `./qwen2.5`)    |
| Embedding model    | **nomic-embed-text** (local, `./nomic_embed`) |
| Vector database    | **ChromaDB** (persistent, `./storage/chroma`) |
| Corpus             | Plain-text essays in `./data/`           |

Everything runs on-device — no API keys, no network calls at query time.

---

## Project layout

```
RAGM4/
├── main.py                 # interactive RAG chat REPL
├── pyproject.toml
├── scripts/
│   └── build_index.py      # one-shot script to (re)build the vector index
├── ragm4/                  # the package
│   ├── config.py           # paths, chunking, retrieval & generation params
│   ├── embeddings.py       # nomic-embed loader (with search_document/query prefixes)
│   ├── llm.py              # LlamaIndex CustomLLM wrapping mlx-lm + Qwen2.5
│   └── pipeline.py         # read → chunk → embed → persist → retrieve
├── data/                   # raw .txt corpus
├── nomic_embed/            # HF snapshot of nomic-embed-text
├── qwen2.5/                # HF snapshot of Qwen2.5-1.5B
└── storage/                # created at runtime — ChromaDB lives here
```

---

## Setup

Requires Python **3.12+** and Apple Silicon (mlx-lm is macOS-only).

```bash
# From the repo root
uv sync
```

---

## Usage

### 1. Build the vector index (first run only)

```bash
uv run python scripts/build_index.py
```

This reads every `.txt` file in `data/`, splits it into ~512-token chunks with
overlap, embeds every chunk with nomic-embed, and persists the vectors to
`storage/chroma/`. The step is skipped on subsequent runs.

To rebuild from scratch (e.g. after changing the corpus):

```bash
uv run python scripts/build_index.py --rebuild
```

### 2. Chat

```bash
uv run python main.py
```

Sample session:

```
you > What did the author do growing up?
bot > Before college the two main things I worked on, outside of school, were writing and programming …
```

Type `exit` or press Ctrl-D to quit.

---

## Tuning

All knobs live in [ragm4/config.py](ragm4/config.py):

- `CHUNK_SIZE` / `CHUNK_OVERLAP` — controls chunking granularity.
- `TOP_K` — number of chunks retrieved per query.
- `MAX_NEW_TOKENS` / `TEMPERATURE` — generation settings for Qwen2.5.

After changing chunking settings, rebuild the index with `--rebuild`.
