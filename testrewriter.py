import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# This is our most robust prompt for rewriting
REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Berdasarkan riwayat percapan dan pertanyaan lanjutan, ubah pertanyaan lanjutan tersebut menjadi pertanyaan mandiri yang bisa berdiri sendiri.
Jika ada salah ketik (typo), perbaiki kesalahan tersebut.
Keluaran Anda HARUS HANYA berupa pertanyaan yang sudah diubah. JANGAN berikan jawaban atau penjelasan.

Contoh:
Riwayat: Human: apa itu manajemen proyek?, AI: Manajemen proyek adalah...
Pertanyaan: jelaskan lagi secara singkat
Keluaran Anda: jelaskan manajemen proyek secara singkat
""",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ]
)

def main():
    """
    Main function to test the rewriter chain interactively with Ollama.
    """
    print("🚀 Starting Rewriter Test with Ollama...")
    
    # We use the powerful local llama3:8b model for this test
    rewriter_llm = OllamaLLM(model="qwen:0.5b", temperature=0)

    chat_history = ChatMessageHistory()

    rewriter_chain = RunnableWithMessageHistory(
        REWRITE_PROMPT | rewriter_llm | StrOutputParser(),
        lambda session_id: chat_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    while True:
        question = input("\nYour Question: ")
        if question.lower() == 'exit':
            break

        rewritten_question = rewriter_chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": "test_session"}}
        )

        print(f"✅ REWRITTEN QUESTION: {rewritten_question}")

        chat_history.add_user_message(question)
        chat_history.add_ai_message("Ini adalah jawaban dummy untuk menjaga konteks.")

if __name__ == "__main__":
    main()