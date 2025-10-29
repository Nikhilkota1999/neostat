import streamlit as st
import os
import sys
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Ensure the app's directory is importable (for models/, utils/, config/)
APP_DIR = os.path.abspath(os.path.dirname(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
from models.llm import get_chat_model, check_llm_health
from models.embeddings import get_embeddings_model
from utils.rag import load_docs_from_uploads, chunk_documents, build_vectorstore, retrieve
from utils.search import web_search
from config.config import DEBUG


def get_chat_response(chat_model, messages, system_prompt: str, context_blocks: List[str]):
    """Get response from the chat model with optional context blocks."""
    try:
        context_text = "\n\n".join([f"[Context {i+1}]\n{blk}" for i, blk in enumerate(context_blocks)]) if context_blocks else ""
        sys_content = system_prompt + (f"\n\nUse the following context if relevant:\n{context_text}" if context_text else "")

        # Prepare messages for the model
        formatted_messages = [SystemMessage(content=sys_content)]

        # Add conversation history
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))

        # Get response from model
        response = chat_model.invoke(formatted_messages)
        return response.content

    except Exception as e:
        return f"Error getting response: {str(e)}"

def instructions_page():
    """Instructions and setup page"""
    st.title("The Chatbot Blueprint")
    st.markdown("Welcome! Follow these instructions to set up and use the chatbot.")
    
    st.markdown("""
    ## 🔧 Installation
                
    
    First, install the required dependencies: (Add Additional Libraries base don your needs)
    
    ```bash
    pip install -r requirements.txt
    ```
    
    ## API Key Setup
    
    You'll need API keys from your chosen provider. Get them from:
    
    ### OpenAI
    - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
    - Create a new API key
    - Set the variables in config
    
    ### Groq
    - Visit [Groq Console](https://console.groq.com/keys)
    - Create a new API key
    - Set the variables in config
    
    ### Google Gemini
    - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
    - Create a new API key
    - Set the variables in config
    
    ## 📝 Available Models
    
    ### OpenAI Models
    Check [OpenAI Models Documentation](https://platform.openai.com/docs/models) for the latest available models.
    Popular models include:
    - `gpt-4o` - Latest GPT-4 Omni model
    - `gpt-4o-mini` - Faster, cost-effective version
    - `gpt-3.5-turbo` - Fast and affordable
    
    ### Groq Models
    Check [Groq Models Documentation](https://console.groq.com/docs/models) for available models.
    Popular models include:
    - `llama-3.1-70b-versatile` - Large, powerful model
    - `llama-3.1-8b-instant` - Fast, smaller model
    - `mixtral-8x7b-32768` - Good balance of speed and capability
    
    ### Google Gemini Models
    Check [Gemini Models Documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for available models.
    Popular models include:
    - `gemini-1.5-pro` - Most capable model
    - `gemini-1.5-flash` - Fast and efficient
    - `gemini-pro` - Standard model
    
    ## How to Use
    
    1. **Go to the Chat page** (use the navigation in the sidebar)
    2. **Start chatting** once everything is configured!
    
    ## Tips
    
    - **System Prompts**: Customize the AI's personality and behavior
    - **Model Selection**: Different models have different capabilities and costs
    - **API Keys**: Can be entered in the app or set as environment variables
    - **Chat History**: Persists during your session but resets when you refresh
    
    ## Troubleshooting
    
    - **API Key Issues**: Make sure your API key is valid and has sufficient credits
    - **Model Not Found**: Check the provider's documentation for correct model names
    - **Connection Errors**: Verify your internet connection and API service status
    
    ---
    
    Ready to start chatting? Navigate to the **Chat** page using the sidebar! 
    """)

def chat_page():
    """Main chat interface page"""
    st.title("🤖 AI ChatBot")
    # Sidebar controls for mode and tools
    with st.sidebar:
        st.subheader("Response Mode")
        mode = st.radio("Style", ["Concise", "Detailed"], index=0)
        st.subheader("Tools")
        use_rag = st.toggle("Use RAG (local docs)", value=False, help="Retrieve from uploaded files")
        use_search = st.toggle("Use Web Search", value=False, help="Search the web for context")

        if use_rag:
            st.caption("Upload documents to index")
            uploaded = st.file_uploader("Add documents", type=["pdf", "txt", "md"], accept_multiple_files=True)
            col_a, col_b = st.columns(2)
            with col_a:
                build_idx = st.button("Build / Rebuild Index", use_container_width=True)
            with col_b:
                clear_idx = st.button("Clear Index", use_container_width=True)

            if clear_idx:
                st.session_state.pop("vectorstore", None)
                st.session_state.pop("indexed_files", None)
                st.session_state.pop("doc_count", None)
                st.success("Cleared vector index")

            if build_idx and uploaded:
                try:
                    # Load and index
                    payload = [(f.name, f.getvalue()) for f in uploaded]
                    docs = load_docs_from_uploads(payload)
                    chunks = chunk_documents(docs)
                    embeddings = get_embeddings_model()
                    vs = build_vectorstore(chunks, embeddings)
                    st.session_state.vectorstore = vs
                    st.session_state.indexed_files = [f.name for f in uploaded]
                    st.session_state.doc_count = len(chunks)
                    st.success(f"Indexed {len(chunks)} chunks from {len(uploaded)} file(s)")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

            if st.session_state.get("doc_count"):
                st.info(f"Index ready: {st.session_state.get('doc_count')} chunks | files: {', '.join(st.session_state.get('indexed_files', []))}")

    # Preflight: check LLM health to avoid long retries
    ok, provider, detail = check_llm_health()
    if ok:
        st.success(f"LLM ready ({provider}): {detail}")
    else:
        st.warning(f"LLM not ready ({provider}): {detail}")

    # Default system prompt (updated based on mode)
    if mode == "Concise":
        system_prompt = (
            "You are a helpful assistant. Respond concisely (1-3 sentences). "
            "If citing sources, mark them as [S1], [S2], etc."
        )
    else:
        system_prompt = (
            "You are a helpful assistant. Provide detailed, step-by-step answers, "
            "including key assumptions and brief citations as [S1], [S2], etc."
        )

    # Determine which provider to use based on available API keys
    try:
        chat_model = get_chat_model()
    except Exception as e:
        st.error(str(e))
        chat_model = None

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display bot response
        with st.chat_message("assistant"):
            with st.spinner("Getting response..."):
                context_blocks: List[str] = []
                gathered_sources: List[Dict] = []

                # RAG retrieval
                if use_rag and st.session_state.get("vectorstore"):
                    try:
                        docs = retrieve(st.session_state.vectorstore, prompt, top_k=4)
                        if docs:
                            rag_block = "\n\n".join([
                                f"[S{i+1}] (local) {d.metadata.get('source','file')} p.{d.metadata.get('page','-')}\n{d.page_content.strip()}"
                                for i, d in enumerate(docs)
                            ])
                            context_blocks.append(rag_block)
                            for i, d in enumerate(docs):
                                gathered_sources.append({
                                    "label": f"S{i+1}",
                                    "type": "local",
                                    "source": d.metadata.get("source", "file"),
                                    "page": d.metadata.get("page"),
                                    "snippet": d.page_content[:500],
                                })
                    except Exception as e:
                        if DEBUG:
                            st.warning(f"RAG retrieval error: {e}")

                # Web search
                if use_search:
                    try:
                        results = web_search(prompt, k=3)
                        if results:
                            search_block = "\n\n".join([
                                f"[W{i+1}] {r.get('title')}\n{r.get('body')}\n{r.get('href')}"
                                for i, r in enumerate(results)
                            ])
                            context_blocks.append(search_block)
                            offset = len(gathered_sources)
                            for i, r in enumerate(results):
                                gathered_sources.append({
                                    "label": f"S{offset + i + 1}",
                                    "type": "web",
                                    "title": r.get("title"),
                                    "url": r.get("href"),
                                    "snippet": r.get("body"),
                                })
                    except Exception as e:
                        if DEBUG:
                            st.warning(f"Search error: {e}")

                if chat_model is None or not ok:
                    response = (
                        "Model is not configured or unavailable. Check API keys and model name in .env."
                    )
                else:
                    response = get_chat_response(chat_model, st.session_state.messages, system_prompt, context_blocks)
                st.markdown(response)

                # Show sources if any
                if gathered_sources:
                    st.session_state.last_sources = gathered_sources
                    with st.expander("Sources"):
                        for s in gathered_sources:
                            if s.get("type") == "web":
                                st.markdown(f"- [{s['label']}] {s.get('title','(web)')} — {s.get('url','')}")
                            else:
                                page = f", p.{s['page']}" if s.get('page') else ""
                                st.markdown(f"- [{s['label']}] {s.get('source','(local)')}{page}")
        
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("Set your API keys and configure tools in the sidebar.")

def main():
    st.set_page_config(
        page_title="LangChain Multi-Provider ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to:",
            ["Chat", "Instructions"],
            index=0
        )
        
        # Add clear chat button in sidebar for chat page
        if page == "Chat":
            st.divider()
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.session_state.last_sources = []
                st.rerun()
    
    # Route to appropriate page
    if page == "Instructions":
        instructions_page()
    if page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()
