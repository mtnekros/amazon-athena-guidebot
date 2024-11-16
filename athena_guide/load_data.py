"""Athena User GuideBot."""
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:116.0) Gecko/20100101 Firefox/116.0"

import json
import re
from typing import Any, Iterator, List

import links
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings, OllamaLLM

persist_directory = "./stores/"
embeddings_st = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_kwargs = {"device": "cuda:0"},
)
# embeddings_llama32 = OllamaEmbeddings(
#     model="llama3.2",
#     base_url="http://localhost:11434",
# )
athena_docs_collection_st = "athena-user-guides" # not used at the moment
# athena_docs_collection_llama = "athena-user-guides-llama-3.2"
store = Chroma(
    embedding_function=embeddings_st,
    collection_name=athena_docs_collection_st,
    # embedding_function=embeddings_llama32,
    # collection_name=athena_docs_collection_llama,
    persist_directory=persist_directory,
    collection_metadata={"hnsw:space": "cosine"},
)
example_store = Chroma(
    embedding_function=embeddings_st,
    collection_name="example-question-answers",
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory=persist_directory,
)

def remove_extra_white_spaces(text: str) -> str:
    """Remove extra white spaces."""
    return re.sub( r"\n[ \t]+", "\n", re.sub(r"\n\n+", "\n\n", text))

def add_page_info_to_docs_content(docs: List[Document]) -> None:
    """Append title info at the end of each document's page_content."""
    for doc in docs:
        doc.page_content += (
            f"\n- All the document content taken from webpage titled \"{doc.metadata['title']}\""
            f"and it is about \"{doc.metadata['title']}\""
        )
    print("Document Example:", docs and docs[0].page_content)

def _build_metadata(soup: Any, url: str) -> dict:  # noqa: ANN401
    """Build metadata from BeautifulSoup output."""
    metadata = {"source": url}
    if title := soup.find("title"):
        metadata["title"] = title.get_text()
    if description := soup.find("meta", attrs={"name": "description"}):
        metadata["description"] = description.get("content", "No description found.")
    if html := soup.find("html"):
        metadata["language"] = html.get("lang", "No language found.")
    return metadata

def nav_item_selector(tag: Any) -> bool:  # noqa: ANN401
    """Select useless sidebar & navigation rtd elements."""
    return (
        tag.name in ("nav", "header", "footer", "aside", "button")
        or "blog-sidebar" in tag.get("class", [])
        or tag.get("id") == "awsdocs-header"
    )

class MainContentWebBaseLoader(WebBaseLoader):
    """Custom WebBaseLoader for Athena docs.

    This only fetches the #main-content div which contains the main information
    of the page.
    """

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load text from the url(s) in web_path."""
        for path in self.web_paths:
            soup = self._scrape(path, bs_kwargs=self.bs_kwargs)
            main_content = soup.find("div", {"id": "main-content"})
            if main_content:  # noqa: SIM108
                text = main_content.get_text(**self.bs_get_text_kwargs)
            else:
                print(f"\n### main-content not found for {path}")
                # removing unwanted elements like sidebar & navigations
                for element in soup.find_all(nav_item_selector):
                    element.extract()
                text = soup.get_text(**self.bs_get_text_kwargs)
            metadata = _build_metadata(soup, path)
            yield Document(page_content=text, metadata=metadata)

def load_data() -> None:
    """Load data into chroma db using ATHENA USER GUIDE LINKS."""
    if store._collection.count():
        print("store already exists")
        return
    loader = MainContentWebBaseLoader(web_paths=list(set(links.all_links)))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for doc in loader.lazy_load():
        doc.page_content = remove_extra_white_spaces(doc.page_content)
        splitted_docs = text_splitter.split_documents(documents=[doc])
        if splitted_docs:
            add_page_info_to_docs_content(splitted_docs)
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

def save_to_file(title: str, content: str) -> None:
    """Create the file name from title and store the contents to local path."""
    file_name = "".join(
        "-" if char.isspace() else char.lower()
        for char in title
        if char.isalnum() or char.isspace()
    )
    with open(f"summarized/{file_name}.txt", "w") as f:
        f.write(content)
    print(f"Saved to {file_name}")


def create_shortener_chain() -> Runnable:
    """Return a prompt template | LLM chain."""
    prompt = PromptTemplate.from_template("""
    I have a piece of text, and I need it to be concise and clear while keeping
    only the relevant information. Please remove any unnecessary details,
    repetitions, or filler words. Ensure the main message remains intact.

    <text>{page_content}<text>
    """)
    llm = OllamaLLM(
        model="llama3.2",
        base_url="http://localhost:11434",
        temperature=0,
    )
    return prompt | llm | StrOutputParser()


def summarize_splitted_content(shortener_chain: Runnable, splitted_docs: List[Document]) -> List[Document]:
    """Sumarize mutliple docs into one."""
    summarized_docs = []
    for splitted_doc in splitted_docs:
        page_content = (
            splitted_doc.metadata.get("title", "") +
            splitted_doc.metadata.get("description", "") +
            "\n\n" +
            splitted_doc.page_content
        )
        summarized_docs.append(Document(
            shortener_chain.invoke({"page_content": page_content}),
            metadata=splitted_doc.metadata,
        ))
    return summarized_docs

def save_summarized_docs_from_links() -> None:
    """Summarize all links and save to local folder."""
    loader = MainContentWebBaseLoader(web_paths=list(set(links.extra_blogs)))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=15_000, chunk_overlap=100)
    shortener_chain = create_shortener_chain()
    for doc in loader.lazy_load():
        doc.page_content = remove_extra_white_spaces(doc.page_content)
        splitted_docs = text_splitter.split_documents(documents=[doc])
        if splitted_docs:
            summarized_docs = summarize_splitted_content(shortener_chain, splitted_docs)
            save_to_file(doc.metadata["title"], f"/n{'#'*100}/n".join(item.page_content for item in summarized_docs))
            store.add_documents(summarized_docs)

if __name__ == "__main__":
    # load_data()
    # load_examples()
    save_summarized_docs_from_links()
    pass
