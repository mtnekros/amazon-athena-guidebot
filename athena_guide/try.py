import os

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:116.0) Gecko/20100101 Firefox/116.0"

import re

import links
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings

persist_directory = "./stores/"
store = Chroma(
    embedding_function=HuggingFaceEmbeddings(),
    collection_name="athena-user-guides",
    persist_directory=persist_directory,
)

def remove_extra_lines(text: str) -> str:
    """Remove extra lines."""
    return re.sub(r"\n\n+", "\n\n", text).strip()

def load_data() -> None:
    """Load data into chroma db using ATHENA USER GUIDE LINKS."""
    loader = WebBaseLoader(web_paths=links.user_guide_links)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=["\n\n", "\n", ".", " "],
    )
    for doc in loader.lazy_load():
        doc.page_content = remove_extra_lines(doc.page_content)
        splitted_docs = text_splitter.split_documents(documents=[doc])
        if splitted_docs:
            store.add_documents(splitted_docs)
            print(f"Added {len(splitted_docs)} documents from {splitted_docs[0].metadata}")

def search(query: str, k: int=5) -> None:
    """Search for related text in the store using similarity search."""
    results = store.similarity_search(query, k=k)
    for i, result in enumerate(results, start=1):
        print(f"Results {i}")
        print(f"Text:, {result.page_content}")
        print(f"Metadata: {result.metadata}\n")

