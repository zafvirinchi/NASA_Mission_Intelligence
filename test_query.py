import os
import chromadb
from chromadb.config import Settings
import rag_client

print("key:", bool(os.getenv("OPENAI_API_KEY")))

client = chromadb.PersistentClient(
    path="./chroma_db_openai",
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_collection("nasa_space_missions_text")
print("collection loaded")

result = rag_client.retrieve_documents(
    collection=collection,
    query="What was the main objective of Apollo 11?",
    n_results=3,
    mission_filter="all"
)

print("query done")
print(result["documents"][0][0][:500])