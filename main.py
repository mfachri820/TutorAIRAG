# main.py
import config
from manual_retriever import Retriever
from rag_chain_builder import create_rag_chain
from history_rewriter import create_history_rewriter_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from reranker import ReRanker
from langchain_core.documents import Document

def main():
    llm = config.llm
    rewriter_llm = config.rewriter_llm
    reranker = ReRanker()
    retriever = Retriever()
    rag_chain = create_rag_chain(llm, None)
    
    chat_history = ChatMessageHistory()
    history_rewriter = create_history_rewriter_chain(rewriter_llm, chat_history)

    print("\n--- Document Q&A with Memory is Ready! ---")
    print("--- Type 'exit' to quit. ---")
    
    while True:
        question = input("\nYour Question: ")
        
        if question.lower() == 'exit':
            print("Exiting... Goodbye!")
            break

        final_question = history_rewriter.invoke(
            {"question": question},
            config={"configurable": {"session_id": "user123"}}
        )
        
        print(f"\nFINAL QUESTION (DEBUG): {final_question}")

        # 1. Retrieve a larger number of chunks
        initial_chunks = retriever.retrieve_context(question, top_k=10)
        initial_docs = [Document(page_content=chunk['content']) for chunk in initial_chunks]
        
        # 2. Re-rank to get the best chunks
        reranked_docs = reranker.rerank(question, initial_docs, top_k=4)
        
        print("\n--- RE-RANKED CHUNKS (DEBUG) ---")
        for i, doc in enumerate(reranked_docs):
            print(f"--- Chunk {i+1} ---\n{doc.page_content}\n")
        print("--- END OF RE-RANKED CHUNKS ---\n")
        
        answer = rag_chain.invoke({"input": question, "context": reranked_docs})
        print("\nAnswer:")
        print(answer)

if __name__ == "__main__":
    main()