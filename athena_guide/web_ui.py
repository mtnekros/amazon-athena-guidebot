from typing import Generator

import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM
from main import get_sources_from_documents, store

st.title("Amazon Athena Assistant")
llm = OllamaLLM(
    model="llama3.2",
    base_url="http://localhost:11434",
)
retriever = store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5}
)
welcome_msg = "Hi! Please ask any questions related to Amazon Athena"
system_prompt = """
    You are an AWS Athena Service Guide. Answer the user's
    question about AWS Athena using the information in the message as context.
    If the answer cannot be found in the message, just say "I don't know"
    Do not hallucinate any false answer.
    <document>{context}</document>
"""

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": welcome_msg},
    ]
    st.session_state.history = [ AIMessage(welcome_msg) ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def yield_res(_rag_chain, _history, _input) -> Generator[str, None, None]:  # noqa: ANN001
    """Stream response from rag_chain."""
    context = []
    answer = ""
    for chunk in _rag_chain.stream({ "history": _history, "input": _input }):
        if "context" in chunk:
            context = chunk["context"]
        if "answer" in chunk:
            answer += chunk["answer"]
            yield chunk["answer"]
    if "i don't know" not in answer.lower():
        yield "\n\nSources: \n"
        yield "\n".join(f"* {src}" for src in get_sources_from_documents(context))

# Accept user input
if prompt := st.chat_input("Ask questions"):
    # Display user message in chat message container
    if prompt == "/clear":
        # Display chat messages from history on app rerun
        with st.chat_message("assistant"):
            st.markdown(welcome_msg)
        st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
        st.session_state.history = [AIMessage(welcome_msg)]
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        prompt_template = ChatPromptTemplate([
            ("system", system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = st.write_stream(yield_res(rag_chain, st.session_state.history, prompt))
        st.session_state.messages.append({"role": "assistant", "content": str(response)})
        st.session_state.history.extend([
            HumanMessage(prompt), # type: ignore
            AIMessage(str(response).split("Sources: ")[0]),
        ])
