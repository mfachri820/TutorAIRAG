# build_index.py hanya dijalankan sekali untuk membangun indeks awal
# Setelah itu, indeks dan metadata disimpan untuk digunakan di seluruh aplikasi
import faiss
import pickle
import numpy as np
import config
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_and_save_index():
    """Reads the source document, chunks it, and saves the FAISS index and metadata."""
    print("🚀 Starting index build process...")

    # 1. Load and chunk the document from config
    print(f"📄 Loading and chunking document from: {config.KNOWLEDGE_BASE_PATH}")
    with open(config.KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, 
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_text(text)
    
    # 2. Prepare metadata
    metadata = [
        {"chunk_id": f"chunk_{i+1:03d}", "source": "my_knowledge", "content": chunk} 
        for i, chunk in enumerate(chunks)
    ]
    print(f"✅ Document split into {len(metadata)} chunks.")

    # 3. Create embeddings and FAISS index
    print("🧠 Loading embedding model and creating embeddings...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    embeddings = model.encode([chunk['content'] for chunk in metadata])
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))
    print("✅ FAISS index built.")

    # 4. Save the index and metadata to files
    faiss.write_index(index, "wbs_faiss.index")
    with open("wbs_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    print("💾 FAISS index and metadata saved successfully.")
    print("🎉 Build process complete.")

if __name__ == "__main__":
    build_and_save_index()