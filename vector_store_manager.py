# vector_store_manager.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import config

def create_vector_store(documents):
    """
    Creates a FAISS vector store from document chunks.

    Args:
        documents (list): A list of document chunks.

    Returns:
        FAISS: The created vector store.
    """
    print("Creating vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(documents, embeddings)
    print("Vector store created successfully.")
    return vector_store