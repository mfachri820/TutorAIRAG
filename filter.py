from langchain_core.prompts import ChatPromptTemplate

def create_relevance_checker_chain(llm):
    """
    Creates a chain that checks if the context is relevant to the question.
    """
    template = """
    Analyze the following context and determine if it contains information to answer the user's question.
    If the context is relevant, start your response with the word RELEVAN.
    If the context is not relevant, start your response with the word TIDAK_RELEVAN.

    Context:
    {context}

    Question:
    {question}

    Analysis:
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm

def is_question_relevant(relevance_chain, retrieved_docs, question):
    """
    Uses the relevance chain to check if the retrieved documents are relevant.
    """
    if not retrieved_docs:
        return False
    
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    response = relevance_chain.invoke({"context": context, "question": question})
    
    # We now check if the model's analysis starts with our keyword
    return response.strip().upper().startswith('RELEVAN')