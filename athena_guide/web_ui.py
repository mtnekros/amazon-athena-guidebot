from textwrap import dedent
from typing import TYPE_CHECKING, Iterator, List, cast

import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
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
    temperature=0,
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
        question about AWS Athena using the information in the <document> tag
        as context. If the answer cannot be found in the document, just say
        "I don't know" Do not hallucinate any false answer.
        <document>{{context}}</document>
        <example>{examples}</example>

        Answer as an AWS Athena expert. Start by breaking down the question,
        then provide detailed steps or relevant explanations.
    """)
    print(system_msg)
    return system_msg

def create_stand_alone_history_aware_prompt(history: List[BaseMessage], question: str) -> str:
    """Create a sub chain that reformulates the question to be history aware."""
    if not history:
        return question
    contextualize_q_system_prompt = dedent("""
        As the Contextual Question Reformulator, your sole task is to make each user
        question fully understandable on its own, without relying on prior
        conversation. Given the latest question and the conversation history, either
        rephrase the question to be standalone or return it unchanged if it already
        stands alone. Do not answer, add information, or seek clarification—simply
        focus on rephrasing for standalone clarity when needed.

        Example 1:
        User: "Can you explain how partitioning works in Athena?"
        System: Provides an explanation of partitioning in Athena.
        User: "Does it improve query performance?"

        "Does partitioning improve query performance in Amazon Athena?"

        Example 2:
        User: "Can you explain how partitioning works in Athena?"
        System: Provides an explanation of partitioning in Athena.
        User: "This is not correct."

        "Your explanation of partitioning in Amazon Athena is not correct."
    """)
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    chain = contextualize_q_prompt | llm | StrOutputParser()
    return chain.invoke({"chat_history": history, "input": question})

def get_history_aware_retriever() -> Runnable:
    """Create a sub chain that reformulates the question to be history aware."""
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

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
    for chunk in _rag_chain.stream({"chat_history": _history, "input": _input}):
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
        standalone_prompt = create_stand_alone_history_aware_prompt(st.session_state.message_history, prompt)
        print(standalone_prompt)
        prompt_template = ChatPromptTemplate([
            ("system", get_system_prompt_with_example(standalone_prompt)),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = st.write_stream(stream_formatted_answer(
            _rag_chain=rag_chain,
            _history=st.session_state.message_history,
            _input=standalone_prompt,
        ))
        st.session_state.st_history.append({"role": "assistant", "content": str(response)})
        st.session_state.message_history.extend([
            HumanMessage(prompt),
            AIMessage(str(response).split("Sources: ")[0]),
        ])
