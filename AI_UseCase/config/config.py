import os

# Load environment variables from a .env file if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


# LLM provider settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()  # 'auto' | 'groq' | 'gemini' | 'openai' | 'openrouter' | 'mistral'

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
# Use a widely supported default; user can override in .env
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

# OpenRouter (OpenAI-compatible API)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
# Optional headers recommended by OpenRouter (not required)
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "")

# Mistral
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

# Embeddings settings
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "huggingface")  # 'openai' or 'huggingface'

# OpenAI (if using OpenAI embeddings)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDINGS_MODEL = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")

# HuggingFace sentence-transformers model (runs locally)
HF_EMBEDDINGS_MODEL = os.getenv("HF_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# RAG chunking
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))

# Search
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "duckduckgo")
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "3"))

# General
DEBUG = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")
