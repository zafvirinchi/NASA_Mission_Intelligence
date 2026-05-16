import os
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path


VOCAREUM_BASE_URL = "https://openai.vocareum.com/v1"
EMBEDDING_MODEL = "text-embedding-3-small"


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
                    document_count = "unknown"
                except Exception:
                    document_count = 0

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

    if not chroma_dir:
        raise ValueError("ChromaDB directory is missing.")

    if not collection_name:
        raise ValueError("ChromaDB collection name is missing.")

    # TODO: Create a chomadb persistentclient
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=False
        )
    )

    # TODO: Return the collection with the collection_name
    collection = client.get_collection(name=collection_name)

    return collection


def create_query_embedding(query: str) -> List[float]:
    """Create query embedding using Vocareum/OpenAI instead of ChromaDB default ONNX"""

    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY is missing. Please set it before querying.")

    client = OpenAI(
        api_key=openai_key,
        base_url=VOCAREUM_BASE_URL
    )

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )

    return response.data[0].embedding


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None
) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    # TODO: Initialize filter variable to None (represents no filtering)
    where_filter = None

    # TODO: Check if filter parameter exists and is not set to "all" or equivalent
    # TODO: If filter conditions are met, create filter dictionary with appropriate field-value pairs
    if mission_filter and mission_filter.lower() not in ["all", "any", "none", ""]:
        where_filter = {"mission": mission_filter}

    # Create OpenAI/Vocareum query embedding manually.
    # This avoids ChromaDB default ONNX embedding and fixes onnxruntime errors.
    query_embedding = create_query_embedding(query)

    # TODO: Execute database query with the following parameters:
    # TODO: Pass search query in the required format
    # TODO: Set maximum number of results to return
    # TODO: Apply conditional filter (None for no filtering, dictionary for specific filtering)
    if where_filter:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
    else:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

    # TODO: Return query results to caller
    return results


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    if not metadatas:
        metadatas = [{} for _ in documents]

    # TODO: Initialize list with header text for context section
    context_parts = ["Retrieved NASA Mission Context:"]

    seen_documents = set()

    # TODO: Loop through paired documents and their metadata using enumeration
    for index, (document, metadata) in enumerate(zip(documents, metadatas), start=1):

        if not document:
            continue

        if metadata is None:
            metadata = {}

        cleaned_document = document.strip()

        if cleaned_document in seen_documents:
            continue

        seen_documents.add(cleaned_document)

        # TODO: Extract mission information from metadata with fallback value
        mission = metadata.get("mission", "unknown")

        # TODO: Clean up mission name formatting (replace underscores, capitalize)
        mission = str(mission).replace("_", " ").title()

        # TODO: Extract source information from metadata with fallback value
        source = metadata.get("source", metadata.get("filepath", metadata.get("file_path", "unknown")))

        # TODO: Extract category information from metadata with fallback value
        category = metadata.get("document_category", metadata.get("category", "general"))

        # TODO: Clean up category name formatting (replace underscores, capitalize)
        category = str(category).replace("_", " ").title()

        # TODO: Create formatted source header with index number and extracted information
        source_header = (
            f"\n--- Source {index} ---\n"
            f"Mission: {mission}\n"
            f"Category: {category}\n"
            f"Source: {source}\n"
        )

        # TODO: Add source header to context parts list
        context_parts.append(source_header)

        # TODO: Check document length and truncate if necessary
        max_document_length = 1800
        if len(cleaned_document) > max_document_length:
            cleaned_document = cleaned_document[:max_document_length] + "..."

        # TODO: Add truncated or full document content to context parts list
        context_parts.append(cleaned_document)

    # TODO: Join all context parts with newlines and return formatted string
    return "\n".join(context_parts)