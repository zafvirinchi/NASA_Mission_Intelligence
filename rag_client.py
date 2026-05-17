import os
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from typing import Dict, List, Optional
from pathlib import Path


VOCAREUM_BASE_URL = "https://openai.vocareum.com/v1"
EMBEDDING_MODEL = "text-embedding-3-small"


def get_openai_base_url(api_key: str):
    if api_key and api_key.startswith("voc"):
        return "https://openai.vocareum.com/v1"
    return None

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # Look for ChromaDB directories
    # TODO: Create list of directories that match specific criteria (directory type and name pattern)
    chroma_dirs = [
        directory for directory in current_dir.iterdir()
        if directory.is_dir()
        and directory.name.lower().startswith("chroma_db")
    ]

    # TODO: Loop through each discovered directory
    for chroma_dir in chroma_dirs:

        # TODO: Wrap connection attempt in try-except block for error handling
        try:
            # TODO: Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )

            # TODO: Retrieve list of available collections from the database
            collections = client.list_collections()

            # TODO: Loop through each collection found
            for collection in collections:

                # ChromaDB versions may return collection objects or collection names
                collection_name = collection.name if hasattr(collection, "name") else str(collection)

                # Skip invalid or empty collection names
                if not collection_name:
                    continue

                # TODO: Create unique identifier key combining directory and collection names
                backend_key = f"{chroma_dir.name}_{collection_name}"

                # TODO: Build information dictionary containing:
                # TODO: Store directory path as string
                # TODO: Store collection name
                # TODO: Create user-friendly display name
                # TODO: Get document count with fallback for unsupported operations
                try:
                    collection_obj = client.get_collection(name=collection_name)
                    document_count = str(collection_obj.count())
                except Exception:
                    document_count = "unknown"

                # TODO: Add collection information to backends dictionary
                backends[backend_key] = {
                    "chroma_dir": str(chroma_dir),
                    "directory": str(chroma_dir),
                    "collection_name": collection_name,
                    "display_name": f"{collection_name} | {chroma_dir} | {document_count} chunks",
                    "document_count": str(document_count)
                }

        # TODO: Handle connection or access errors gracefully
        except Exception as error:

            # TODO: Create fallback entry for inaccessible directories
            backend_key = f"{chroma_dir.name}_error"

            # TODO: Include error information in display name with truncation
            error_message = str(error)
            if len(error_message) > 80:
                error_message = error_message[:80] + "..."

            # TODO: Set appropriate fallback values for missing information
            backends[backend_key] = {
                "chroma_dir": str(chroma_dir),
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir} | Error: {error_message}",
                "document_count": "0"
            }

    # TODO: Return complete backends dictionary with all discovered collections
    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = get_openai_base_url(api_key)

    embedding_function = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
        api_base=base_url
    )

    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=False
        )
    )

    collection = client.get_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    return collection

def create_query_embedding(query: str) -> List[float]:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = get_openai_base_url(api_key)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    return response.data[0].embedding

def retrieve_documents(collection, query: str, n_results: int = 3,
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    where_filter = None

    if mission_filter and mission_filter.lower() not in ["all", "any", "none", ""]:
        where_filter = {"mission": mission_filter}

    query_embedding = create_query_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter
    )

    return results


def format_context(documents, metadatas):
    """
    Format retrieved documents into structured context for the LLM.
    """

    if not documents:
        return ""

    formatted_sections = []

    seen_docs = set()

    for i, doc in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) else {}

        mission = metadata.get("mission", "unknown")
        source = metadata.get("source", "unknown_source")
        file_path = metadata.get("file_path", "unknown_path")

        # Deduplicate repeated chunks
        dedupe_key = f"{mission}_{source}_{doc[:100]}"

        if dedupe_key in seen_docs:
            continue

        seen_docs.add(dedupe_key)

        formatted_section = f"""
==============================
MISSION: {mission}
SOURCE: {source}
FILE: {file_path}
==============================

{doc}
"""

        formatted_sections.append(formatted_section)

    return "\n\n".join(formatted_sections)