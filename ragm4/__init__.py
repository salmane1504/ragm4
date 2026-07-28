"""RAGM4 — a local RAG pipeline for Apple Silicon.

Components:
- LlamaIndex   : orchestration (chunking, retrieval, query engine)
- mlx-lm       : local LLM inference on Apple Silicon (Qwen2.5-1.5B)
- nomic-embed  : local sentence-transformer embedding model
- ChromaDB     : persistent vector store
"""
