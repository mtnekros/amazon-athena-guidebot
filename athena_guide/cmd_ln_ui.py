from textwrap import dedent
from typing import List

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM
from load_data import get_sources_from_documents, store


def main() -> None:
    """Run the main chat function with Ollama."""
    llm = OllamaLLM(
        model="llama3.2",
        base_url="http://localhost:11434",
    )
    welcome_msg = "Hi! Please ask any questions related to Amazon Athena"
    system_prompt = dedent("""
        You are an AWS Athena Service Guide. Answer the user's
        question about AWS Athena using the information in the document
        message as context. If the answer cannot be found in the system message,
        just say you don't know.
        <document>{context}</document>
    """)
    history: List[BaseMessage] = [AIMessage(welcome_msg)]
    print(welcome_msg)
    retriever = store.as_retriever(search_type="similarity", k=3)
    while True:
        user_msg_str = input("\nYou: ")
        if user_msg_str == "/exit":
            print("exiting chat")
            return
        if user_msg_str == "/clear":
            history = []
            print("history cleared")
            continue

        prompt = ChatPromptTemplate([
            ("system", system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        # response = rag_chain.invoke({"history": history, "input": user_msg_str})
        res_str = ""
        context = []
        print("\nAthena Assistant:")
        for res_chunk in rag_chain.stream({
            "history": history,
            "input": user_msg_str,
        }):
            if "context" in res_chunk:
                context = res_chunk["context"]
            if "answer" in res_chunk:
                res_str += res_chunk["answer"]
                # print(res_chunk["answer"], end="", flush=True)
            print(f"{res_chunk=}")
        print("\nSources:", get_sources_from_documents(context))
        history.extend([
            HumanMessage(user_msg_str),
            AIMessage(res_str),
        ])
        history = history[:-25]


if __name__ == "__main__":
    main()

