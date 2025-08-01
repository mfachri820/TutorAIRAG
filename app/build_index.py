# build_index.py
import faiss
import pickle
import numpy as np
import config
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_and_save_index():
    print("🚀 Starting index build process...")

    with open(config.KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, 
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_text(text)
    
    metadata = [
        {"chunk_id": f"chunk_{i+1:03d}", "source": "my_knowledge", "content": chunk} 
        for i, chunk in enumerate(chunks)
    ]
    print(f"✅ Document split into {len(metadata)} chunks.")

    print("🧠 Loading embedding model and creating embeddings...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    
    # --- THIS IS THE FIX ---
    # Add the "passage: " prefix to each chunk before encoding
    passages_to_embed = [f"passage: {chunk['content']}" for chunk in metadata]
    embeddings = model.encode(passages_to_embed)
    # --------------------
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))
    print("✅ FAISS index built.")

    faiss.write_index(index, "../index/faiss.index")
    with open("../index/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    print("💾 FAISS index and metadata saved successfully.")

if __name__ == "__main__":
    build_and_save_index()