import os
import streamlit as st

import chromadb
from chromadb.config import Settings
import rag_client
import llm_client

st.set_page_config(page_title="NASA RAG Chat", page_icon="🚀")

st.title("🚀 NASA Space Mission Chat")

openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    st.error("OPENAI_API_KEY missing. Set it before running Streamlit.")
    st.stop()

client = chromadb.PersistentClient(
    path="./chroma_db_openai",
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_collection("nasa_space_missions_text")

st.success("ChromaDB connected successfully")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about NASA missions...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        docs_result = rag_client.retrieve_documents(
            collection=collection,
            query=prompt,
            n_results=3,
            mission_filter="all"
        )

        documents = docs_result["documents"][0]
        metadatas = docs_result["metadatas"][0]
        context = rag_client.format_context(documents, metadatas)

        answer = llm_client.generate_response(
            openai_key=openai_key,
            user_message=prompt,
            context=context,
            conversation_history=st.session_state.messages[:-1],
            model="gpt-3.5-turbo"
        )

        st.markdown(answer)

        with st.expander("Retrieved Context"):
            st.text(context)

    st.session_state.messages.append({"role": "assistant", "content": answer})