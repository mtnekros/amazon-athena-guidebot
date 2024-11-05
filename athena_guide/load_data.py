"""Athena User GuideBot."""
import os

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:116.0) Gecko/20100101 Firefox/116.0"

import json
import re
from typing import List

import links
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

persist_directory = "./stores/"
embeddings_st = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_kwargs = {"device": "cuda:0"},
)
embeddings_llama32 = OllamaEmbeddings(
    model="llama3.2",
    base_url="http://localhost:11434",
)
athena_docs_collection_st = "athena-user-guides" # not used at the moment
athena_docs_collection_llama = "athena-user-guides-llama-3.2"
store = Chroma(
    embedding_function=embeddings_llama32,
    collection_name=athena_docs_collection_llama,
    persist_directory=persist_directory,
    collection_metadata={"hnsw:space": "cosine"},
)
example_store = Chroma(
    embedding_function=embeddings_st,
    collection_name="example-question-answers",
    collection_metadata={"hnsw:space": "cosine"},
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    for doc in loader.lazy_load():
        doc.page_content = remove_extra_lines(doc.page_content)
        splitted_docs = text_splitter.split_documents(documents=[doc])
        if splitted_docs:
            store.add_documents(splitted_docs)
            print(f"Added {len(splitted_docs)} documents from {splitted_docs[0].metadata}")

def load_examples() -> None:
    """Load example for question answering."""
    if example_store._collection.count():
        print("example store already exists")
        return
    with open("./examples/question-answer-examples.json") as f:
        examples = json.load(f)
    for example in examples:
        example_store.add_texts([str(example)])
    print(f"Added {len(examples)} examples into the question answer store.")


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


if __name__ == "__main__":
    # load_data()
    # load_examples()
    pass
