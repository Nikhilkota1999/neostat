import os

# Load environment variables from a .env file if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Optionally source values from Streamlit secrets if running in Streamlit
try:
    import streamlit as st  # type: ignore
    _SECRETS = dict(st.secrets)
except Exception:
    _SECRETS = {}


def _get(key: str, default: str = "") -> str:
    return str(_SECRETS.get(key, os.getenv(key, default)))


def _get_int(key: str, default: int) -> int:
    val = _get(key, str(default))
    try:
        return int(str(val))
    except Exception:
        return default


def _get_bool(key: str, default: bool = False) -> bool:
    val = str(_SECRETS.get(key, os.getenv(key, str(default)))).lower()
    return val in ("1", "true", "yes", "on")


# LLM provider settings
LLM_PROVIDER = _get("LLM_PROVIDER", "auto").lower()  # 'auto' | 'groq' | 'gemini' | 'openai' | 'openrouter' | 'mistral'

# Groq
GROQ_API_KEY = _get("GROQ_API_KEY", "")
GROQ_MODEL = _get("GROQ_MODEL", "llama-3.1-8b-instant")

# Google Gemini
GOOGLE_API_KEY = _get("GOOGLE_API_KEY", "")
# Use a widely supported default; user can override in .env
GEMINI_MODEL = _get("GEMINI_MODEL", "gemini-1.5-pro")

# OpenRouter (OpenAI-compatible API)
OPENROUTER_API_KEY = _get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = _get("OPENROUTER_MODEL", "openrouter/auto")
# Optional headers recommended by OpenRouter (not required)
OPENROUTER_SITE_URL = _get("OPENROUTER_SITE_URL", "")
OPENROUTER_APP_NAME = _get("OPENROUTER_APP_NAME", "")

# Mistral
MISTRAL_API_KEY = _get("MISTRAL_API_KEY", "")
MISTRAL_MODEL = _get("MISTRAL_MODEL", "mistral-small-latest")

# Embeddings settings
EMBEDDINGS_PROVIDER = _get("EMBEDDINGS_PROVIDER", "huggingface")  # 'openai' or 'huggingface'

# OpenAI (if using OpenAI embeddings)
OPENAI_API_KEY = _get("OPENAI_API_KEY", "")
OPENAI_EMBEDDINGS_MODEL = _get("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")

# HuggingFace sentence-transformers model (runs locally)
HF_EMBEDDINGS_MODEL = _get("HF_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# RAG chunking
RAG_CHUNK_SIZE = _get_int("RAG_CHUNK_SIZE", 800)
RAG_CHUNK_OVERLAP = _get_int("RAG_CHUNK_OVERLAP", 120)

# Search
SEARCH_PROVIDER = _get("SEARCH_PROVIDER", "duckduckgo")
SEARCH_TOP_K = _get_int("SEARCH_TOP_K", 3)

# General
DEBUG = _get_bool("DEBUG", False)
