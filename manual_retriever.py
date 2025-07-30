# retriever.py
import faiss
import numpy as np
import config
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Retriever:
    def __init__(self):
        print("🔧 Initializing Retriever...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        print(f"📄 Loading and chunking document from: {config.KNOWLEDGE_BASE_PATH}")
        with open(config.KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE, 
            chunk_overlap=config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_text(text)
        
        self.metadata = [{"content": chunk} for chunk in chunks]
        print(f"✅ Document split into {len(self.metadata)} chunks.")

        print("🧠 Creating embeddings and building FAISS index...")
        embeddings = self.model.encode([chunk['content'] for chunk in self.metadata])
        
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings, dtype=np.float32))
        print("✅ FAISS index built successfully.")

    def retrieve_context(self, user_question, top_k=4):
        question_embedding = self.model.encode([user_question])
        _, indices = self.index.search(np.array(question_embedding, dtype=np.float32), top_k)
        return [self.metadata[i] for i in indices[0]]