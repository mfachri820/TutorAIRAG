# main.py
import config
from manual_retriever import retrieve_context
from rag_chain_builder import create_rag_chain
from history_rewriter import create_history_rewriter_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from reranker import ReRanker
from langchain_core.documents import Document

def main():
    llm = config.llm
    rewriter_llm = config.rewriter_llm
    
    reranker = ReRanker()
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

        if len(chat_history.messages) == 0:
            final_question = question
        else:
            final_question = history_rewriter.invoke(
                {"question": question},
                config={"configurable": {"session_id": "user123"}}
            )
        
        print(f"\nFINAL QUESTION (DEBUG): {final_question}")

        # 1. Retrieve using the corrected, final question
        retrieved_chunks, _ = retrieve_context(final_question)
        initial_docs = [Document(page_content=chunk['content']) for chunk in retrieved_chunks]
        
        # 2. Re-rank using the corrected, final question
        reranked_docs = reranker.rerank(final_question, initial_docs, top_k=4)
        
        print("\n--- RE-RANKED CHUNKS (DEBUG) ---")
        for i, doc in enumerate(reranked_docs):
            print(f"--- Chunk {i+1} ---\n{doc.page_content}\n")
        print("--- END OF RE-RANKED CHUNKS ---\n")
        
        if reranked_docs:
            # 3. Generate answer using the corrected, final question
            answer = rag_chain.invoke({"input": final_question, "context": reranked_docs})
            
            # 4. Save the conversation to memory for the next turn
            chat_history.add_user_message(question)
            chat_history.add_ai_message(answer)
            
            print("\nAnswer:")
            print(answer)
        else:
            print("\nAnswer:")
            print("Maaf, saya tidak dapat menemukan informasi yang relevan di dokumen.")

if __name__ == "__main__":
    main()