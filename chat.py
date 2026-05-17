#!/usr/bin/env python3

import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "True"

import logging
from typing import Dict, List, Optional

import streamlit as st
import chromadb
from chromadb.config import Settings

import rag_client
import llm_client
import ragas_evaluator


logging.getLogger("chromadb").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)


st.set_page_config(
    page_title="NASA RAG Chat with Evaluation",
    page_icon="🚀",
    layout="wide"
)


CHROMA_DIR = "./chroma_db_submission"

MISSION_COLLECTIONS = {
    "Apollo 11": {
        "collection_name": "nasa_apollo11",
        "mission_filter": "apollo_11"
    },
    "Apollo 13": {
        "collection_name": "nasa_apollo13",
        "mission_filter": "apollo_13"
    },
    "Challenger": {
        "collection_name": "nasa_challenger",
        "mission_filter": "challenger"
    }
}


def detect_question_mission(question: str) -> Optional[str]:
    question_lower = question.lower()

    if "apollo 11" in question_lower or "apollo11" in question_lower:
        return "Apollo 11"

    if "apollo 13" in question_lower or "apollo13" in question_lower:
        return "Apollo 13"

    if "challenger" in question_lower or "sts-51l" in question_lower or "sts 51l" in question_lower:
        return "Challenger"

    return None


def initialize_collection(collection_name: str):
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=False,
            is_persistent=True
        )
    )

    return client.get_collection(name=collection_name)


def display_evaluation_metrics(scores: Dict[str, float]):
    if not scores:
        return

    if "error" in scores:
        st.sidebar.error(f"Evaluation Error: {scores['error']}")
        return

    st.sidebar.subheader("📊 Response Quality")

    for metric_name, score in scores.items():
        if isinstance(score, (int, float)):
            safe_score = min(max(float(score), 0.0), 1.0)
            st.sidebar.metric(
                label=metric_name.replace("_", " ").title(),
                value=f"{safe_score:.3f}"
            )
            st.sidebar.progress(safe_score)


def main():
    st.title("🚀 NASA Space Mission Chat with Evaluation")
    st.markdown(
        "Ask questions about Apollo 11, Apollo 13, or Challenger using NASA mission documents."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = None

    with st.sidebar:
        st.header("🔧 Configuration")

        openai_key = st.text_input(
            "OpenAI / Vocareum API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", "")
        )

        if not openai_key:
            st.warning("Please enter your OpenAI/Vocareum API key.")
            st.stop()

        os.environ["OPENAI_API_KEY"] = openai_key

        selected_mission = st.selectbox(
            "Select Mission Collection",
            options=list(MISSION_COLLECTIONS.keys())
        )

        selected_collection_name = MISSION_COLLECTIONS[selected_mission]["collection_name"]
        selected_mission_filter = MISSION_COLLECTIONS[selected_mission]["mission_filter"]

        model_choice = st.selectbox(
            "OpenAI Model",
            options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            index=0
        )

        n_docs = st.slider(
            "Documents to retrieve",
            min_value=1,
            max_value=10,
            value=5
        )

        enable_evaluation = st.checkbox("Enable RAGAS Evaluation", value=True)

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.last_evaluation = None
            st.rerun()

        if st.session_state.last_evaluation and enable_evaluation:
            display_evaluation_metrics(st.session_state.last_evaluation)

    try:
        collection = initialize_collection(selected_collection_name)
        st.success(f"Connected to `{selected_collection_name}` from `{CHROMA_DIR}`")
    except Exception as e:
        st.error(f"Failed to connect to ChromaDB collection: {e}")
        st.stop()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about NASA missions...")

    if prompt:
        detected_mission = detect_question_mission(prompt)

        if detected_mission and detected_mission != selected_mission:
            st.warning(
                f"You selected **{selected_mission}**, but your question appears to be about **{detected_mission}**. "
                f"Please select **{detected_mission}** from the sidebar and ask again."
            )
            st.stop()

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving NASA context and generating answer..."):
                docs_result = rag_client.retrieve_documents(
                    collection=collection,
                    query=prompt,
                    n_results=n_docs,
                    mission_filter=selected_mission_filter
                )

                documents = docs_result.get("documents", [[]])[0] if docs_result else []
                metadatas = docs_result.get("metadatas", [[]])[0] if docs_result else []

                context = rag_client.format_context(documents, metadatas)

                if not context:
                    st.warning("No context retrieved from ChromaDB.")

                answer = llm_client.generate_response(
                    openai_key=openai_key,
                    user_message=prompt,
                    context=context,
                    conversation_history=st.session_state.messages[:-1],
                    model=model_choice
                )

                st.markdown(answer)

                with st.expander("📚 Retrieved Context"):
                    st.text(context)

                if enable_evaluation:
                    with st.spinner("Evaluating response quality..."):
                        evaluation_scores = ragas_evaluator.evaluate_response_quality(
                            question=prompt,
                            answer=answer,
                            contexts=documents
                        )

                    st.session_state.last_evaluation = evaluation_scores

                    with st.expander("📊 RAGAS Evaluation Scores"):
                        st.json(evaluation_scores)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


if __name__ == "__main__":
    main()