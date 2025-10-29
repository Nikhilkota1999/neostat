from __future__ import annotations

from io import BytesIO
from typing import List, Tuple, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config.config import RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP


def _load_pdf(file_like: BytesIO, name: str) -> List[Document]:
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError(f"PDF support requires pypdf: {e}")

    reader = PdfReader(file_like)
    docs: List[Document] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": name, "page": i + 1}))
    return docs


def _load_text(file_like: BytesIO, name: str) -> List[Document]:
    content = file_like.read().decode("utf-8", errors="ignore")
    return [Document(page_content=content, metadata={"source": name})]


def load_docs_from_uploads(uploads: List[Tuple[str, bytes]]) -> List[Document]:
    """
    Accepts a list of tuples (filename, file_bytes) and returns LangChain Documents.
    Supports PDF and text-like files.
    """
    all_docs: List[Document] = []
    for name, data in uploads:
        lower = name.lower()
        bio = BytesIO(data)
        if lower.endswith(".pdf"):
            all_docs.extend(_load_pdf(bio, name))
        else:
            all_docs.extend(_load_text(bio, name))
    return all_docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vectorstore(docs: List[Document], embeddings) -> Chroma:
    if not docs:
        raise ValueError("No documents provided to build the vector store.")
    return Chroma.from_documents(documents=docs, embedding=embeddings)


def retrieve(vectorstore: Chroma, query: str, top_k: int = 4) -> List[Document]:
    return vectorstore.similarity_search(query, k=top_k)
