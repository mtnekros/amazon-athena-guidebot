"""Athena User GuideBot."""
import os

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:116.0) Gecko/20100101 Firefox/116.0"

import re
from textwrap import dedent
from typing import List

import links
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

persist_directory = "./stores/"
store = Chroma(
    embedding_function=HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        model_kwargs = {"device": "cuda:0"},
    ),
    collection_name="athena-user-guides",
    persist_directory=persist_directory,
)

def remove_extra_lines(text: str) -> str:
    """Remove extra lines."""
    return re.sub(r"\n\n+", "\n\n", text).strip()

def load_data() -> None:
    """Load data into chroma db using ATHENA USER GUIDE LINKS."""
    if store._collection.count():
        print("store already exists")
        return
    loader = WebBaseLoader(web_paths=list(set(links.all_links)))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    for doc in loader.lazy_load():
        doc.page_content = remove_extra_lines(doc.page_content)
        splitted_docs = text_splitter.split_documents(documents=[doc])
        if splitted_docs:
            store.add_documents(splitted_docs)
            print(f"Added {len(splitted_docs)} documents from {splitted_docs[0].metadata}")

def search(query: str, k: int=5) -> str:
    """Search for related text in the store using similarity search."""
    results = store.similarity_search(query, k=k)
    for i, result in enumerate(results, start=1):
        print(f"Results {i}")
        print(f"Text:, {result.page_content}")
        print(f"Metadata: {result.metadata}\n")
    return "/n".join(f"{result.page_content}" for result in results)

def get_sources_from_documents(documents: List[Document]) -> List[str]:
    """Get the source from the documnents."""
    return list({item.metadata["source"] for item in documents})


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
        message as context. If the answer cannot be found in the system
        message, just say you don't know.
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
        response = rag_chain.invoke({"history": history, "input": user_msg_str})
        print("\nAthena Assistant:", response["answer"])
        print("\nSources:", get_sources_from_documents(response["context"]))
        history.extend([
            HumanMessage(user_msg_str),
            AIMessage(response["answer"]),
        ])


if __name__ == "__main__":
    load_data()
    # main()
