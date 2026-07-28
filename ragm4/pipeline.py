"""Ingestion + retrieval pipeline.

- `build_or_load_index()` is the single entry point used by both the offline
  build script and the interactive chat.
- On first run it reads every `.txt` file from `/data`, splits it into chunks,
  embeds them with nomic-embed, and persists the vectors to ChromaDB.
- On subsequent runs it detects the existing collection and simply reopens it,
  skipping the (expensive) embedding step.
"""

from __future__ import annotations

import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
)


def _get_chroma_collection() -> tuple[chromadb.api.ClientAPI, chromadb.Collection]:
    """Open (or create) the persistent ChromaDB collection on disk."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    return client, collection


def build_or_load_index(force_rebuild: bool = False) -> VectorStoreIndex:
    """Return a ready-to-query `VectorStoreIndex`.

    Parameters
    ----------
    force_rebuild:
        When True the on-disk collection is dropped and rebuilt from scratch.
        Use this if you change the corpus or the chunking settings.
    """
    client, collection = _get_chroma_collection()

    # Wipe & recreate when the caller asks for a full rebuild.
    if force_rebuild and collection.count() > 0:
        client.delete_collection(CHROMA_COLLECTION)
        collection = client.get_or_create_collection(CHROMA_COLLECTION)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # ---- Fast path: collection already populated ----------------------------
    if collection.count() > 0:
        print(f"[pipeline] Reusing {collection.count()} vectors from Chroma.")
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )

    # ---- Slow path: read → split → embed → persist --------------------------
    print(f"[pipeline] Reading corpus from {DATA_DIR} …")
    documents = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        required_exts=[".txt"],
    ).load_data()
    print(f"[pipeline] Loaded {len(documents)} document(s).")

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print("[pipeline] Chunking + embedding + persisting to Chroma …")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )
    print(f"[pipeline] Done — {collection.count()} vectors written.")
    return index
