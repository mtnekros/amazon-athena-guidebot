from textwrap import dedent
from typing import TYPE_CHECKING, Iterator, List, cast

import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_ollama import OllamaLLM
from load_data import example_store, get_sources_from_documents, store

if TYPE_CHECKING:
    from chromadb.api.types import Document


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

def get_system_prompt_with_example(user_question: str) -> str:
    """Combine releavant example with system prompt and return it as a string."""
    documents = example_store.similarity_search_with_score(user_question, k=1)
    examples = "\n".join(
        doc.page_content.replace("{", "").replace("}","").replace("\\\\", "\\")
        for doc,score in documents
        if score > 0.5
    )
    system_msg = dedent(f"""
        You are an AWS Athena Service Guide. Answer the user's
        question about AWS Athena using the information in the message as context.
        If the answer cannot be found in the message, just say "I don't know"
        Do not hallucinate any false answer.
        <document>{{context}}</document>
        <example>{examples}</example>
    """)
    from pprint import pprint
    pprint(system_msg)
    return system_msg

# Initialize chat history
if "st_history" not in st.session_state:
    st.session_state.st_history = [
        {"role": "assistant", "content": welcome_msg},
    ]
    st.session_state.message_history = [ AIMessage(welcome_msg) ]
    # casting is just purely for static type checking for python, it doesn't
    # change the actual data or the type of the data.
    st.session_state.message_history = cast(List[BaseMessage], st.session_state.message_history)

# Display chat messages from history on app rerun
for message in st.session_state.st_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def stream_formatted_answer(_rag_chain: Runnable, _history: List[BaseMessage], _input: str) -> Iterator[str]:
    """Stream response from rag_chain."""
    context: List[Document] = []
    answer = ""
    for chunk in _rag_chain.stream({"history": _history, "input": _input}):
        if "context" in chunk:
            context = chunk["context"]
        if "answer" in chunk:
            answer += chunk["answer"]
            yield chunk["answer"]
    if "i don't know" not in answer.lower():
        yield "\n\nSources: \n"
        yield "\n".join(f"* {src}" for src in get_sources_from_documents(context)) # type: ignore

# Accept user input
if prompt := st.chat_input("Ask questions"):
    # Display user message in chat message container
    if prompt == "/clear":
        # Display chat messages from history on app rerun
        with st.chat_message("assistant"):
            st.markdown(welcome_msg)
        st.session_state.st_history = [{"role": "assistant", "content": welcome_msg}]
        st.session_state.message_history = [AIMessage(welcome_msg)]
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.st_history.append({"role": "user", "content": prompt})
        get_system_prompt_with_example(prompt)
        prompt_template = ChatPromptTemplate([
            ("system", get_system_prompt_with_example(prompt)),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = st.write_stream(stream_formatted_answer(rag_chain, st.session_state.message_history, prompt))
        st.session_state.st_history.append({"role": "assistant", "content": str(response)})
        st.session_state.message_history.extend([
            HumanMessage(prompt),
            AIMessage(str(response).split("Sources: ")[0]),
        ])
