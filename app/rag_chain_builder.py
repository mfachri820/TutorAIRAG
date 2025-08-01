# rag_chain_builder.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_rag_chain(llm, retriever):
    """
    Membuat RAG chain yang menerima pertanyaan dan konteks secara langsung.
    """
    # VVV INI ADALAH PROMPT YANG SUDAH DITERJEMAHKAN VVV
    template = """
    Anda adalah seorang ahli yang menjawab pertanyaan berdasarkan teks yang diberikan.
    Jawab pertanyaan pengguna secara ringkas dalam satu atau dua kalimat.

    Konteks:
    {context}

    Pertanyaan:
    {input}

    Jawaban:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Chain ini sekarang akan menghasilkan jawaban dalam Bahasa Indonesia
    rag_chain = prompt | llm | StrOutputParser()
    
    return rag_chain