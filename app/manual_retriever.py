import faiss
import pickle
import json
import textwrap
from sentence_transformers import SentenceTransformer
from langdetect import detect
from deep_translator import GoogleTranslator
import numpy as np
import re

# === Configuration ===
DEFAULT_CONFIG = {
    'faiss_top_k': 15,
    'rerank_top_k': 10,
    'final_top_k': 5,
    'similarity_threshold': 0.5,  # Increased from 0.3 to 0.5
    'min_chunk_length': 50,
    'jaccard_threshold': 0.75,
    'keyword_boost_factor': 0.15,
    'enable_dedup': True,
    'enable_keyword_boost': True
}

# === Load FAISS index ===
index = faiss.read_index("wbs_faiss.index")

# === Load metadata chunks ===
with open("wbs_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# === Load model embedding ===
model = SentenceTransformer("intfloat/e5-base-v2")

# === Translation cache ===
translation_cache = {}

# === Deteksi dan translasi pertanyaan ===
def translate_if_needed(text, target_lang="en"):
    cache_key = f"{text}_{target_lang}"
    if cache_key in translation_cache:
        return translation_cache[cache_key]
    
    try:
        lang = detect(text)
    except:
        lang = "unknown"
    
    if lang == "id" and target_lang == "en":
        translated = GoogleTranslator(source='id', target='en').translate(text)
        result = (translated, "id")
    else:
        result = (text, lang)
    
    translation_cache[cache_key] = result
    return result

# === Enhanced query preprocessing ===
def preprocess_query(self, query: str) -> str:
    """Adds the required prefix to the query for the e5-base-v2 model."""
    return f"query: {query}"

# === Keyword-based filtering ===
def keyword_filter(chunks, query, boost_factor=0.15):
    """Boost chunks that contain query keywords"""
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    
    for chunk in chunks:
        content_words = set(re.findall(r'\b\w+\b', chunk['content'].lower()))
        keyword_matches = len(query_words.intersection(content_words))
        
        if len(query_words) > 0:
            keyword_ratio = keyword_matches / len(query_words)
        else:
            keyword_ratio = 0
        
        chunk['keyword_score'] = keyword_ratio * boost_factor
        chunk['total_score'] = chunk.get('faiss_score', 0) + chunk['keyword_score']
    
    return chunks

# === Content-based deduplication ===
def remove_duplicates(chunks, jaccard_threshold=0.75):
    """Remove duplicate chunks using Jaccard similarity"""
    if len(chunks) <= 1:
        return chunks
    
    unique_chunks = [chunks[0]]
    
    for candidate in chunks[1:]:
        is_unique = True
        candidate_words = set(candidate['content'].lower().split())
        
        for existing in unique_chunks:
            existing_words = set(existing['content'].lower().split())
            intersection = candidate_words.intersection(existing_words)
            union = candidate_words.union(existing_words)
            jaccard_sim = len(intersection) / len(union) if union else 0
            
            if jaccard_sim > jaccard_threshold:
                is_unique = False
                break
        
        if is_unique:
            unique_chunks.append(candidate)
    
    return unique_chunks

# === Improved fallback strategy ===
def apply_fallback_strategy(filtered_chunks, candidate_chunks, final_top_k):
    """Gradual fallback when not enough chunks pass filters"""
    if len(filtered_chunks) >= final_top_k:
        return filtered_chunks[:final_top_k]
    
    print("\n⚠ Not enough chunks passed filters, applying fallback strategy...")
    
    # Try relaxed threshold
    relaxed_threshold = DEFAULT_CONFIG['similarity_threshold'] * 0.8
    relaxed_chunks = [c for c in candidate_chunks 
                     if c.get('total_score', c['faiss_score']) >= relaxed_threshold]
    
    if len(relaxed_chunks) >= final_top_k:
        print(f"Using {len(relaxed_chunks)} chunks with relaxed threshold {relaxed_threshold:.2f}")
        return relaxed_chunks[:final_top_k]
    
    # Final fallback to top FAISS results
    print("Using top FAISS results as final fallback")
    return candidate_chunks[:final_top_k]

# === Main retrieval function ===
def retrieve_context(user_question, config=None):
    """
    Optimized Three-stage retrieval:
    1. FAISS retrieval (fast)
    2. Top-K reranking with keyword boost
    3. Deduplication and final selection
    """
    if config is None:
        config = DEFAULT_CONFIG
    else:
        config = {**DEFAULT_CONFIG, **config}  # Merge dictionarie
    
    # Translate if needed
    question_translated, detected_lang = translate_if_needed(user_question)
    processed_query = preprocess_query(question_translated)
    
    print(f"🔍 Original query: {user_question}")
    print(f"🔍 Processed query: {processed_query}")
    print(f"🌐 Detected language: {detected_lang}")
    
    # STAGE 1: FAISS Retrieval
    question_embedding = model.encode([processed_query], convert_to_numpy=True)
    distances, indices = index.search(question_embedding, config['faiss_top_k'])
    
    print(f"\n📊 Stage 1 - FAISS Retrieved: {config['faiss_top_k']} candidates")
    
    # Convert FAISS results to chunks with proper scoring
    candidate_chunks = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            chunk = metadata[idx].copy()
            chunk['faiss_distance'] = float(distances[0][i])
            chunk['faiss_score'] = 1.0 / (1.0 + chunk['faiss_distance'])
            chunk['rank'] = i + 1
            
            if len(chunk["content"].strip()) >= config['min_chunk_length']:
                candidate_chunks.append(chunk)
    
    print(f"🔧 After length filtering: {len(candidate_chunks)} chunks")
    
    # STAGE 2: Keyword boosting and reranking
    if config['enable_keyword_boost']:
        candidate_chunks = keyword_filter(candidate_chunks, question_translated, 
                                       config['keyword_boost_factor'])
        candidate_chunks.sort(key=lambda x: x['total_score'], reverse=True)
    else:
        candidate_chunks.sort(key=lambda x: x['faiss_score'], reverse=True)
    
    top_candidates = candidate_chunks[:config['rerank_top_k']]
    print(f"\n🎯 Stage 2 - Top {len(top_candidates)} candidates selected")
    
    # Apply similarity threshold
    filtered_chunks = [
        chunk for chunk in top_candidates
        if chunk.get('total_score', chunk['faiss_score']) >= config['similarity_threshold']
    ]
    print(f"🔧 After similarity threshold: {len(filtered_chunks)} chunks")
    
    # STAGE 3: Deduplication
    if config['enable_dedup'] and len(filtered_chunks) > 1:
        final_candidates = remove_duplicates(filtered_chunks, config['jaccard_threshold'])
        print(f"🔧 After deduplication: {len(final_candidates)} chunks")
    else:
        final_candidates = filtered_chunks
    
    # Apply fallback strategy if needed
    final_chunks = apply_fallback_strategy(final_candidates, candidate_chunks, config['final_top_k'])
    
    # Debug output
    print(f"\n🎯 Final Results: {len(final_chunks)} chunks")
    print("\n🔍 Detailed Scoring:")
    
    for i, chunk in enumerate(final_chunks):
        print(f"\n--- Chunk {i+1} from {chunk['source']} ---")
        print(f"FAISS: {chunk.get('faiss_score', 0):.4f} | "
              f"Keyword: {chunk.get('keyword_score', 0):.4f} | "
              f"Total: {chunk.get('total_score', chunk['faiss_score']):.4f}")
        print(f"Length: {len(chunk['content'])} chars | Rank: {chunk.get('rank', 'N/A')}")
        preview = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
        print(f"Preview: {preview}")
    
    # Prepare clean output
    clean_chunks = [{
        'chunk_id': c['chunk_id'],
        'source': c['source'],
        'content': c['content']
    } for c in final_chunks]
    
    return clean_chunks, detected_lang

# === Alternative simple retrieval ===
def simple_retrieve(user_question, top_k=5):
    """Pure FAISS retrieval without post-processing"""
    question_translated, detected_lang = translate_if_needed(user_question)
    processed_query = preprocess_query(question_translated)
    
    question_embedding = model.encode([processed_query], convert_to_numpy=True)
    distances, indices = index.search(question_embedding, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            results.append({
                **metadata[idx],
                'distance': float(distances[0][i]),
                'rank': i + 1
            })
    
    return results, detected_lang

# === Test function ===
def test_retrieval(query, method='enhanced', config=None):
    """Test retrieval with different methods"""
    print(f"\n🧪 Testing query: '{query}'")
    
    if method == 'simple':
        chunks, lang = simple_retrieve(query)
        print("\n📊 Simple Retrieval Results:")
        for i, chunk in enumerate(chunks):
            print(f"{i+1}. {chunk['source']} (distance: {chunk['distance']:.4f})")
            print(f"   {chunk['content'][:100]}...")
    else:
        chunks, lang = retrieve_context(query, config)
        print(f"\n📊 Enhanced Retrieval Results: {len(chunks)} chunks")