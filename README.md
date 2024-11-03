# Amazon Athena GPT

Implementing RAG that uses Athena User Guide + Athena API Documentations +
boto3 athena client information that will answer any questions related to
Athena

# Task list
* [X] Tries when loading the data in VectorStores
    * [X] Try loading with PDF with PyPDFLoader
        * [X] Sources are just pages in the PDF
            * [X] I would have like some headers or title instead
                * [X] Tried converting pdf to markdown
                    * [X] With PyPDFMiner
                    * [X] With pandoc (The process to semantically organize in sections was a bit difficult. But it is possible)
                * [X] In both approaches, I didn't get any h1 tags
                * [X] I could've checked the fontsizes to get the t1 tags.
                * [X] (The process to semantically organize in sections was a bit difficult. But it is possible)
            * [X] Ditching this approach
            * [X] Also pretty janky text format
    * [X] Getting the links from the documentation webpages.
        * [X] Only need level 2 links (Because it's all in the same page)
        * [X] Using JS in the console to get the links
            * [X] Because many links were generated with JS (BS4 wasn't an option)
            * [X] It's just a lot easier to run `querySelectorAll('a')` in a loaded page
        * [X] Can use langchain's WebBaseLoader class to get the content
            * [X] Benefit: descriptive metadata (link_url, title, ...)
* [X] Vector Data Store Creation
    * [X] Create a vector store with following data
        * [X] Amazon Athena HTML User Guide
        * [ ] Amazon API Documentation
        * [ ] boto3 Athena Client Documentation
        * [ ] boto3 Athena Resource Documentation
* [ ] Integrate this with locally running ollama (Llama3.2 3B Model)
    * [ ] Try out with Ollama python Client
    * [ ] Try out with LangChain's Ollama Client
* [ ] Integrate the Streamlit UI
