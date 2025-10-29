from typing import Any

from config.config import (
    EMBEDDINGS_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_EMBEDDINGS_MODEL,
    HF_EMBEDDINGS_MODEL,
)


def get_embeddings_model() -> Any:
    """
    Return a LangChain-compatible embeddings object based on configuration.

    - If EMBEDDINGS_PROVIDER == 'openai': uses OpenAIEmbeddings (requires OPENAI_API_KEY)
    - Otherwise: uses HuggingFaceEmbeddings (local, downloads model on first run)
    """
    provider = (EMBEDDINGS_PROVIDER or "").strip().lower()
    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except Exception as e:
            raise RuntimeError(f"Failed to import OpenAI embeddings: {e}")

        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set but EMBEDDINGS_PROVIDER=openai")

        return OpenAIEmbeddings(model=OPENAI_EMBEDDINGS_MODEL, api_key=OPENAI_API_KEY)

    # Default to HuggingFace sentence-transformers for local embeddings
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception as e:
        raise RuntimeError(f"Failed to import HuggingFace embeddings: {e}")

    return HuggingFaceEmbeddings(model_name=HF_EMBEDDINGS_MODEL)

