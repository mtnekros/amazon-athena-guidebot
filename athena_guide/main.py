"""Loading and Saving the Amazon Athena User Guide to Vector Database."""
import links
from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

persist_directory = "./stores/"
store = Chroma(
    embedding_function=HuggingFaceEmbeddings(),
    collection_name="athena-user-guides",
    persist_directory=persist_directory,
)
if not store._collection.count():
    pdf_path = "./pdfs/athena-ug.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=500,
        separators=["\n\n", "\n", ".", " "],
    )
    splitted_chunks = text_splitter.split_documents(documents=docs)
    store.add_documents(splitted_chunks)

def search(query: str, k: int=5) -> None:
    """Search for related text in the store using similarity search."""
    results = store.similarity_search(query, k=k)
    for i, result in enumerate(results, start=1):
        print(f"Results {i}")
        print(f"Text:, {result.page_content}")
        print(f"Metadata: {result.metadata}\n")

def split_pdf_using_headers() -> None:
    """Split the pdf using header."""
    pdf_path = "./pdfs/athena-ug.pdf"
    print("Initializing PDFMinerPDFasHTMLLoader")
    loader = PDFMinerPDFasHTMLLoader(file_path=pdf_path)
    print("Initializing HTMLHeaderTextSplitter")
    splitter = HTMLHeaderTextSplitter([("h1", "H1"), ("h2", "H2"), ("h3", "H3")])
    print("Loading HTML from PDF")
    documents = loader.load()
    splitted_docs = []
    print("Splitting by headers")
    for doc in documents:
        splitted_docs.extend(splitter.split_text(doc.page_content))
    # Maybe add the CharacterTextSplitter here again
    print("Adding the documents to the store")
    store.add_texts(splitted_docs)

