# data_loader.py

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # <-- Import Document
import config

# Import the main chunking function from your new script
from advanced_chunker import chunk_text as advanced_chunker_function

def load_and_split_documents():
    """
    Loads documents and splits them using either the simple or advanced chunker
    based on the configuration.
    """
    print("Loading documents...")
    loader = TextLoader(config.KNOWLEDGE_BASE_PATH, encoding="utf-8")
    docs = loader.load()

    # The raw text content is in the first document
    text_content = docs[0].page_content
    
    # --- CHUNKER SWITCH ---
    if config.USE_ADVANCED_CHUNKER:
        print("Using Advanced Semantic Chunker...")
        # Use the imported function from your advanced_chunker.py
        chunks_as_strings = advanced_chunker_function(
            text_content,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            use_semantic=True # Assuming you want the semantic part
        )
        # Convert the string chunks into LangChain Document objects
        documents = [Document(page_content=chunk) for chunk in chunks_as_strings]

    else:
        print("Using Simple Recursive Chunker...")
        # This is our original, simple chunker
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE, 
            chunk_overlap=config.CHUNK_OVERLAP
        )
        documents = text_splitter.split_documents(docs)

    print(f"Successfully split into {len(documents)} documents.")
    return documents