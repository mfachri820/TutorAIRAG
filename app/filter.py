from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_relevance_checker_chain(llm):
    """
    Creates a chain that checks if the context is relevant to the question.
    """
    template = """
    Analisis konteks yang diberikan untuk menentukan apakah ada informasi yang cukup untuk menjawab pertanyaan pengguna.
    Jawaban Anda harus diawali dengan salah satu kata kunci berikut: RELEVAN, MUNGKIN_RELEVAN, atau TIDAK_RELEVAN.

    - Gunakan RELEVAN jika konteks secara langsung menjawab pertanyaan.
    - Gunakan MUNGKIN_RELEVAN jika konteksnya terkait dengan topik tetapi mungkin bukan jawaban yang sempurna.
    - Gunakan TIDAK_RELEVAN jika konteksnya sama sekali tidak berhubungan.

    Konteks:
    {context}

    Pertanyaan:
    {question}

    Analisis:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    return prompt | llm | StrOutputParser()

def is_question_relevant(relevance_chain, retrieved_docs, question):
    """
    Uses the relevance chain to check if the retrieved documents are relevant.
    Now considers both RELEVAN and MUNGKIN_RELEVAN as true.
    """
    if not retrieved_docs:
        return False
    
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    response = relevance_chain.invoke({"context": context, "question": question})
    
    response_upper = response.strip().upper()
    
    # This is the less strict check
    if response_upper.startswith('RELEVAN') or response_upper.startswith('MUNGKIN_RELEVAN'):
        return True
    
    return False