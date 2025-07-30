import config
from data_loader import load_and_split_documents
from vector_store_manager import create_vector_store
from rag_chain_builder import create_rag_chain
from filter import create_relevance_checker_chain, is_question_relevant

def main():
    """
    The main function to run the RAG pipeline.
    """
    llm = config.llm

    documents = load_and_split_documents()
    vector_store = create_vector_store(documents)
    retriever = vector_store.as_retriever()
    rag_chain = create_rag_chain(llm, retriever)
    
    relevance_checker = create_relevance_checker_chain(llm)

    print("\n--- Document Q&A is Ready! ---")
    print("--- Type 'exit' to quit. ---")
    
    while True:
        question = input("\nYour Question: ")
        
        if question.lower() == 'exit':
            print("Exiting... Goodbye!")
            break
        
        retrieved_docs = retriever.invoke(question)
        
        print("\n--- RETRIEVED CHUNKS (DEBUG) ---")
        for i, doc in enumerate(retrieved_docs):
            print(f"--- Chunk {i+1} ---\n{doc.page_content}\n")
        print("--- END OF RETRIEVED CHUNKS ---\n")
        
        if is_question_relevant(relevance_checker, retrieved_docs, question):
            response = rag_chain.invoke({"input": question})
            print("\nAnswer:")
            print(response["answer"])
        else:
            print("\nAnswer:")
            print("Maaf, pertanyaan tersebut di luar lingkup dokumen yang saya miliki.")

if __name__ == "__main__":
    main()