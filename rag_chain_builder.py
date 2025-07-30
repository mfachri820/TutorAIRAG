# rag_chain_builder.py

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def create_rag_chain(llm, retriever):
    template = """
    Anda adalah seorang ahli manajemen proyek. Tugas Anda adalah menjawab pertanyaan pengguna berdasarkan teks yang disediakan.

    Gunakan aturan berikut:
    1. Berikan jawaban yang jelas dan ringkas.
    2. Gunakan bahasa yang natural dan mudah dimengerti.
    3. Jawaban Anda harus sepenuhnya berdasarkan konteks yang diberikan. Jika jawaban tidak ada di dalam teks, katakan demikian.

    Konteks:
    {context}

    Pertanyaan:
    {input}

    Jawaban:
    """

    prompt = ChatPromptTemplate.from_template(template)
    
    Youtube_chain = create_stuff_documents_chain(llm, prompt)
    
    rag_chain = create_retrieval_chain(retriever, Youtube_chain)
    print("RAG chain built successfully.")
    return rag_chain