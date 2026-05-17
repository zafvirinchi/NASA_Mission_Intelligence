import os
import json
import pandas as pd

import rag_client
import llm_client
import ragas_evaluator


CHROMA_DIR = "./chroma_db_submission"

MISSION_COLLECTIONS = {
    "overview": "nasa_apollo11",
    "technical": "nasa_apollo11",
    "timeline": "nasa_apollo11",
    "emergency": "nasa_apollo13",
    "crew": "nasa_apollo13",
    "disaster_analysis": "nasa_challenger",
    "communications": "nasa_challenger"
}

mission_keywords = {
    "Apollo 11": ["apollo 11", "apollo11"],
    "Apollo 13": ["apollo 13", "apollo13"],
    "Challenger": ["challenger", "sts-51l", "sts 51l"]
}

TEST_FILE = "test_questions.json"


def load_questions(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def run_batch_evaluation():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing")

    questions = load_questions(TEST_FILE)

    results = []

    for item in questions:
        category = item.get("category", "overview")
        collection_name = MISSION_COLLECTIONS.get(category, "nasa_apollo11")
        question = item["question"]

        print(f"\nEvaluating: {question}")
        print(f"Using collection: {collection_name}")

        collection = rag_client.initialize_rag_system(
            CHROMA_DIR,
            collection_name
        )

        retrieved = rag_client.retrieve_documents(
            collection=collection,
            query=question,
            n_results=4,
            mission_filter="all"
        )

        contexts = retrieved.get("documents", [[]])[0]
        metadatas = retrieved.get("metadatas", [[]])[0]

        context_text = rag_client.format_context(contexts, metadatas)

        answer = llm_client.generate_response(
            openai_key=api_key,
            user_message=question,
            context=context_text,
            conversation_history=[],
            model="gpt-3.5-turbo"
        )

        scores = ragas_evaluator.evaluate_response_quality(
            question=question,
            answer=answer,
            contexts=contexts
        )

        row = {
            "category": category,
            "collection": collection_name,
            "question": question,
            "answer": answer
        }

        row.update(scores)
        results.append(row)

    df = pd.DataFrame(results)

    print("\n==============================")
    print("Per Question Results")
    print("==============================")
    print(df)

    numeric_columns = df.select_dtypes(include="number").columns

    print("\n==============================")
    print("Aggregate Mean Scores")
    print("==============================")
    print(df[numeric_columns].mean())

    df.to_csv("batch_evaluation_results.csv", index=False)
    print("\nSaved: batch_evaluation_results.csv")


if __name__ == "__main__":
    run_batch_evaluation()