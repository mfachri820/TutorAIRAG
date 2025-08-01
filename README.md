## Proyek RAG 

Ini adalah proyek *Retrieval-Augmented Generation* (RAG) sederhana yang menggunakan model bahasa lokal (via Ollama) untuk menjawab pertanyaan berdasarkan dokumen yang diberikan.

---
### **Penjelasan File (Bahasa Indonesia)**

* `main.py`: File utama untuk menjalankan aplikasi Q&A interaktif.
* `config.py`: Pusat konfigurasi untuk pengaturan model, *path* file, dan parameter lainnya.
* `retriever.py`: Modul yang bertanggung jawab untuk memuat dokumen, membuat *chunks*, dan membangun indeks pencarian (FAISS).
* `rag_chain_builder.py`: Membuat *chain* LangChain utama yang menggabungkan *prompt*, konteks, dan LLM untuk menghasilkan jawaban.
* `history_rewriter.py`: (Opsional) Mengelola memori percakapan dengan menulis ulang pertanyaan lanjutan agar menjadi pertanyaan yang mandiri.
* `reranker.py`: (Opsional) Menyaring dan mengurutkan ulang hasil pencarian dari *retriever* untuk mendapatkan konteks yang paling relevan.
* `filter.py`: (Opsional) Filter cerdas untuk memeriksa apakah konteks yang ditemukan relevan dengan pertanyaan sebelum menghasilkan jawaban.
* `.gitignore`: Menginstruksikan Git untuk mengabaikan file atau folder tertentu (seperti `env/` dan `.env`).
* `ekstrak/`: Folder yang berisi dokumen sumber pengetahuan (misalnya, `my_knowledge.txt`).

---
### **File Descriptions (English)**

* `main.py`: The main file to run the interactive Q&A application.
* `config.py`: Central configuration hub for model settings, file paths, and other parameters.
* `retriever.py`: The module responsible for loading the document, creating chunks, and building the search index (FAISS).
* `rag_chain_builder.py`: Creates the main LangChain chain that combines the prompt, context, and LLM to generate an answer.
* `history_rewriter.py`: (Optional) Manages conversation memory by rewriting follow-up questions to be standalone.
* `reranker.py`: (Optional) Filters and re-ranks the search results from the retriever to get the most relevant context.
* `filter.py`: (Optional) A smart filter to check if the found context is relevant to the question before generating an answer.
* `.gitignore`: Instructs Git to ignore certain files or folders (like `env/` and `.env`).
* `ekstrak/`: The folder containing the source knowledge documents (e.g., `my_knowledge.txt`).