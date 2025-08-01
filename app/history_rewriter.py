# history_rewriter.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

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

def create_history_rewriter_chain(llm, chat_history: BaseChatMessageHistory):
    return RunnableWithMessageHistory(
        REWRITE_PROMPT | llm | StrOutputParser(),
        lambda session_id: chat_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )