
# Proyek TutorAI RAG

Sebuah aplikasi RAG (Retrieval-Augmented Generation) yang mengintegrasikan berbagai model LLM untuk kebutuhan tutorial. Proyek ini dibangun dengan Python dan menggunakan *library* LangChain.

## Persyaratan

  * Python 3.9 atau yang lebih baru.
  * Git untuk mengkloning repositori.

-----

## Langkah-langkah Instalasi dan Setup

Ikuti langkah-langkah ini dengan saksama untuk menyiapkan proyek di komputer Anda.

### 1\. Klona Repositori

Buka terminal dan klona repositori ini ke mesin lokal Anda:

```bash
git clone [URL_REPOSITORI_ANDA]
cd rag_project
```

### 2\. Buat dan Aktifkan Virtual Environment

Sangat penting untuk menggunakan *virtual environment* agar tidak mengganggu instalasi Python global Anda.

```bash
# Buat environment baru bernama "env"
python -m venv env

# Aktifkan environment (untuk Windows PowerShell)
.\env\Scripts\Activate.ps1
```

*Catatan: Untuk macOS/Linux, gunakan `source env/bin/activate`.*

Setelah aktif, Anda akan melihat `(env)` di awal baris terminal Anda.

### 3\. Instal Paket dari `requirements.txt`

Dengan *environment* yang sudah aktif, instal semua paket yang dibutuhkan dengan satu perintah:

```bash
pip install -r requirements.txt
```

### 4\. Unduh Model Bahasa spaCy

Proyek ini memerlukan model bahasa dari spaCy yang harus diunduh secara terpisah.

```bash
# Perintah ini akan mengunduh model 'en_core_web_sm'
python -m spacy download en_core_web_sm
```

-----

## Troubleshooting: PydanticUserError

Jika Anda menghadapi `PydanticUserError` terkait `A non-annotated attribute was detected: \`URL = ...\``setelah menjalankan aplikasi, ini disebabkan oleh konflik versi antara`langchain-openrouter`dan`pydantic\`. Berikut adalah cara memperbaikinya secara manual:

1.  **Buka File:** Navigasi dan buka file berikut di editor kode Anda:
    `env\Lib\site-packages\langchain_openrouter\openrouter.py`

2.  **Tambahkan Import:** Di bagian atas file, bersama dengan impor lainnya, tambahkan baris ini:

    ```python
    from typing import ClassVar
    ```

3.  **Ubah Kode:** Cari baris yang mendefinisikan `URL`. Ubah dari:

    ```python
    # Tampilan kode sebelum diubah
    URL: str = "https://openrouter.ai/api/v1/chat/completions"
    ```

    Menjadi:

    ```python
    # Tampilan kode setelah diubah
    URL: ClassVar[str] = "https://openrouter.ai/api/v1/chat/completions"
    ```

4.  **Simpan File** dan jalankan kembali aplikasi Anda.

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