import os
import sys

# Ensure the app root directory (AI_UseCase) is importable
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from config.config import (
    LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL,
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_SITE_URL,
    OPENROUTER_APP_NAME,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
)


def get_chatgroq_model():
    """Initialize and return the Groq chat model using env/config."""
    try:
        from langchain_groq import ChatGroq
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Please set it in your environment or Streamlit secrets.")
        model_name = GROQ_MODEL or "llama-3.1-8b-instant"
        groq_model = ChatGroq(
            api_key=GROQ_API_KEY,
            model=model_name,
        )
        return groq_model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")


def get_gemini_model():
    """Initialize and return the Google Gemini chat model using env/config."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set. Please set it in your environment or Streamlit secrets.")
        # Validate model via google-generativeai list_models to give better errors
        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            available = []
            try:
                for m in genai.list_models():
                    name = getattr(m, "name", "")
                    methods = set(getattr(m, "supported_generation_methods", []) or [])
                    if name and ("generateContent" in methods or not methods):
                        # Strip the leading 'models/' prefix when collecting
                        available.append(name.split("/")[-1])
            except Exception:
                # list_models can be restricted; fall back to configured model
                available = []
        except Exception:
            available = []

        model_name = GEMINI_MODEL or "gemini-1.5-pro"
        # If we have a list of available models and the configured one isn't present, raise useful error
        if available and model_name not in available:
            # Also allow '-latest' alias mapping
            alt = model_name + "-latest"
            if alt in available:
                model_name = alt
            else:
                raise RuntimeError(
                    f"Gemini model '{model_name}' not available for this API key. Available: {', '.join(sorted(set(available)))}"
                )

        # ChatGoogleGenerativeAI reads GOOGLE_API_KEY from env; pass explicitly too.
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini model: {str(e)}")


def get_chat_model():
    """
    Return the chat model based on LLM_PROVIDER preference.
    - LLM_PROVIDER=groq | gemini | openai (openai not implemented here) | auto
    - auto: prefer Groq if configured, else Gemini.
    """
    provider = (LLM_PROVIDER or "auto").lower()
    if provider == "groq":
        return get_chatgroq_model()
    if provider == "gemini":
        return get_gemini_model()
    if provider == "openrouter":
        return get_openrouter_model()
    if provider == "mistral":
        return get_mistral_model()

    # auto
    if GROQ_API_KEY:
        try:
            return get_chatgroq_model()
        except Exception:
            pass
    if GOOGLE_API_KEY:
        return get_gemini_model()
    if OPENROUTER_API_KEY:
        return get_openrouter_model()
    if MISTRAL_API_KEY:
        return get_mistral_model()
    raise RuntimeError("No supported LLM configured. Set GROQ_API_KEY or GOOGLE_API_KEY (and optionally LLM_PROVIDER).")


def check_llm_health() -> tuple[bool, str, str]:
    """
    Check whether the configured LLM is reachable and the model is available.
    Returns (ok, provider, detail_message).
    """
    provider = (LLM_PROVIDER or "auto").lower()
    # Try explicit provider first
    if provider == "groq":
        if not GROQ_API_KEY:
            return False, "groq", "GROQ_API_KEY is not set"
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            # Check model existence
            m = client.models.retrieve(GROQ_MODEL)
            return True, "groq", f"Model '{m.name}' available"
        except Exception as e:
            return False, "groq", f"Groq check failed: {e}"

    if provider == "gemini" or provider == "auto":
        if not GOOGLE_API_KEY:
            if provider == "gemini":
                return False, "gemini", "GOOGLE_API_KEY is not set"
        else:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GOOGLE_API_KEY)
                target = GEMINI_MODEL or "gemini-1.5-pro"
                names = []
                for m in genai.list_models():
                    name = getattr(m, "name", "")
                    methods = set(getattr(m, "supported_generation_methods", []) or [])
                    if name and ("generateContent" in methods or not methods):
                        names.append(name.split("/")[-1])
                if target in names or (target + "-latest") in names:
                    return True, "gemini", f"Model '{target}' available"
                return False, "gemini", (
                    f"Gemini model '{target}' not found. Available: {', '.join(sorted(set(names)))}"
                    if names else "Gemini list_models unavailable for this key/project"
                )
            except Exception as e:
                return False, "gemini", f"Gemini check failed: {e}"

    # OpenRouter explicit or auto
    
    # Fallback
    if provider == "openrouter" or provider == "auto":
        if not OPENROUTER_API_KEY:
            if provider == "openrouter":
                return False, "openrouter", "OPENROUTER_API_KEY is not set"
        else:
            try:
                import requests
                headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
                if OPENROUTER_SITE_URL:
                    headers["HTTP-Referer"] = OPENROUTER_SITE_URL
                if OPENROUTER_APP_NAME:
                    headers["X-Title"] = OPENROUTER_APP_NAME
                resp = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=15)
                if resp.status_code != 200:
                    return False, "openrouter", f"OpenRouter models list failed: HTTP {resp.status_code}"
                data = resp.json()
                ids = [d.get("id") for d in data.get("data", []) if d.get("id")]
                target = OPENROUTER_MODEL or "openrouter/auto"
                if target == "openrouter/auto" or target in ids:
                    return True, "openrouter", f"Model '{target}' available"
                return False, "openrouter", (
                    f"Model '{target}' not found. Example available: {', '.join(ids[:10])}"
                )
            except Exception as e:
                return False, "openrouter", f"OpenRouter check failed: {e}"

    # Mistral explicit or auto
    if provider == "mistral" or provider == "auto":
        if not MISTRAL_API_KEY:
            if provider == "mistral":
                return False, "mistral", "MISTRAL_API_KEY is not set"
        else:
            try:
                import requests
                headers = {
                    "Authorization": f"Bearer {MISTRAL_API_KEY}",
                    "Accept": "application/json",
                }
                resp = requests.get("https://api.mistral.ai/v1/models", headers=headers, timeout=15)
                if resp.status_code != 200:
                    return False, "mistral", f"Mistral models list failed: HTTP {resp.status_code}"
                data = resp.json()
                ids = [d.get("id") for d in data.get("data", []) if d.get("id")]
                target = MISTRAL_MODEL or "mistral-small-latest"
                if target in ids:
                    return True, "mistral", f"Model '{target}' available"
                return False, "mistral", (
                    f"Model '{target}' not found. Example available: {', '.join(ids[:10])}"
                )
            except Exception as e:
                return False, "mistral", f"Mistral check failed: {e}"

    return False, provider, "No supported LLM configured"


def get_openrouter_model():
    """Initialize and return an OpenRouter-backed ChatOpenAI model."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to .env or env vars.")
    from langchain_openai import ChatOpenAI
    model = OPENROUTER_MODEL or "openrouter/auto"
    default_headers = {}
    if OPENROUTER_SITE_URL:
        default_headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_APP_NAME:
        default_headers["X-Title"] = OPENROUTER_APP_NAME

    # ChatOpenAI is OpenAI-compatible; route via OpenRouter base_url
    return ChatOpenAI(
        model=model,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers=default_headers or None,
    )


def get_mistral_model():
    """Initialize and return a Mistral Chat model."""
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY is not set. Add it to .env or env vars.")
    from langchain_mistralai import ChatMistralAI
    model = MISTRAL_MODEL or "mistral-small-latest"
    return ChatMistralAI(model=model, api_key=MISTRAL_API_KEY)
