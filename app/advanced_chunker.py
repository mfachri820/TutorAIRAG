import os
import json
import re
import spacy
from typing import List, Dict, Tuple, Optional
from tabulate import tabulate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    print("⚠️ Model spaCy belum terinstall. Jalankan: python -m spacy download en_core_web_sm")
    nlp = None


class AdvancedSemanticChunker:
    def __init__(self, chunk_size=500, chunk_overlap=100, min_chunk_size=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.table_pattern = re.compile(r'\|.*\|.*\|', re.MULTILINE)
        self.header_patterns = [
            re.compile(r'^#{1,6}\s+.+$', re.MULTILINE),  # Markdown headers
            re.compile(r'^\d+(\.\d+)*\s+.+$', re.MULTILINE),  # Numbered headers (1.1, 1.2.1)
            re.compile(r'^[A-Z][A-Z\s]+$', re.MULTILINE),  # ALL CAPS headers
            re.compile(r'^[^\w]*[A-Z].{10,}[^\w]*$', re.MULTILINE)  # Title case headers
        ]

    def detect_document_structure(self, text: str) -> Dict[str, List[Tuple[int, int]]]:
        """Enhanced structure detection with better paragraph handling"""
        structure = {'paragraphs': [], 'tables': [], 'lists': [], 'headers': [], 'code_blocks': []}
        lines = text.split('\n')
        current_pos = 0
        
        # Detect paragraphs (separated by double newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        para_pos = 0
        for para in paragraphs:
            if para.strip():
                para_start = text.find(para.strip(), para_pos)
                para_end = para_start + len(para.strip())
                structure['paragraphs'].append((para_start, para_end))
                para_pos = para_end

        i = 0
        while i < len(lines):
            line = lines[i]
            line_start = current_pos
            line_end = current_pos + len(line)

            # Headers detection (multiple patterns)
            if any(pattern.match(line.strip()) for pattern in self.header_patterns):
                structure['headers'].append((line_start, line_end))
            
            # Code blocks detection
            elif line.strip().startswith('```'):
                code_start = line_start
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('```'):
                    i += 1
                    current_pos += len(lines[i-1]) + 1
                code_end = current_pos + len(lines[i]) if i < len(lines) else current_pos
                structure['code_blocks'].append((code_start, code_end))
            
            # Table detection (enhanced)
            elif self.table_pattern.search(line):
                table_start = line_start
                j = i
                # Find complete table
                while j < len(lines) and (self.table_pattern.search(lines[j]) or 
                                        lines[j].strip() == '' or 
                                        lines[j].strip().startswith('|') or
                                        re.match(r'^[-\s|:]+$', lines[j].strip())):
                    j += 1
                table_end = current_pos + sum(len(lines[k]) + 1 for k in range(i, j))
                structure['tables'].append((table_start, table_end))
                i = j - 1
            
            # Lists detection (enhanced)
            elif (re.match(r'^\s*[-*+•]\s', line) or 
                re.match(r'^\s*\d+\.\s', line) or
                re.match(r'^\s*[a-zA-Z]\.\s', line)):
                list_start = line_start
                j = i + 1
                # Continue until non-list item
                while j < len(lines):
                    next_line = lines[j].strip()
                    if (not next_line or 
                        re.match(r'^\s*[-*+•]\s', lines[j]) or
                        re.match(r'^\s*\d+\.\s', lines[j]) or
                        re.match(r'^\s*[a-zA-Z]\.\s', lines[j]) or
                        lines[j].startswith('  ')):  # Indented continuation
                        j += 1
                    else:
                        break
                list_end = current_pos + sum(len(lines[k]) + 1 for k in range(i, j))
                structure['lists'].append((list_start, list_end))
                i = j - 1

            current_pos += len(line) + 1
            i += 1

        return structure

    def extract_and_format_tables(self, text: str, table_positions: List[Tuple[int, int]]) -> Dict[int, str]:
        """Enhanced table extraction with better formatting"""
        formatted_tables = {}
        for i, (start, end) in enumerate(table_positions):
            table_text = text[start:end].strip()
            try:
                lines = [line.strip() for line in table_text.split('\n') if line.strip()]
                table_data = []
                
                for line in lines:
                    # Skip separator lines
                    if re.match(r'^[-\s|:]+$', line):
                        continue
                    
                    if '|' in line:
                        # Clean and split cells
                        cells = [cell.strip() for cell in line.split('|')]
                        # Remove empty cells from start/end
                        while cells and not cells[0]:
                            cells.pop(0)
                        while cells and not cells[-1]:
                            cells.pop()
                        
                        if cells:
                            table_data.append(cells)
                
                if len(table_data) >= 2:  # At least header + 1 row
                    formatted = f"\n[TABEL {i+1}]\n"
                    try:
                        formatted += tabulate(table_data[1:], headers=table_data[0], 
                                        tablefmt='grid', stralign='left')
                    except:
                        # Fallback to simple format
                        formatted += '\n'.join(' | '.join(row) for row in table_data)
                    formatted += f"\n[/TABEL {i+1}]\n"
                    formatted_tables[start] = formatted
                else:
                    # Keep original if can't format properly
                    formatted_tables[start] = f"\n[TABEL {i+1}]\n{table_text}\n[/TABEL {i+1}]\n"
                    
            except Exception as e:
                print(f"⚠️ Error formatting table {i+1}: {e}")
                formatted_tables[start] = f"\n[TABEL {i+1}]\n{table_text}\n[/TABEL {i+1}]\n"
        
        return formatted_tables

    def paragraph_aware_chunking(self, text: str) -> List[str]:
        """Chunk by paragraphs first, then by sentences if needed"""
        structure = self.detect_document_structure(text)
        
        # Process tables first
        formatted_tables = self.extract_and_format_tables(text, structure['tables'])
        processed_text = text
        
        # Replace tables with formatted versions (sorted by position desc to maintain indices)
        for start_pos in sorted(formatted_tables.keys(), reverse=True):
            for start, end in structure['tables']:
                if start == start_pos:
                    processed_text = processed_text[:start] + formatted_tables[start_pos] + processed_text[end:]
                    break

        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', processed_text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If paragraph is too long, split by sentences
            if len(para) > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Split long paragraph by sentences
                para_chunks = self.sentence_based_split(para)
                chunks.extend(para_chunks)
            else:
                # Try to add paragraph to current chunk
                test_chunk = current_chunk + ("\n\n" if current_chunk else "") + para
                
                if len(test_chunk) <= self.chunk_size:
                    current_chunk = test_chunk
                else:
                    # Current chunk is full, save it and start new one
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return self.apply_overlap(chunks)

    def sentence_based_split(self, text: str) -> List[str]:
        """Split text by sentences with semantic awareness"""
        if nlp:
            sentences = [sent.text.strip() for sent in nlp(text).sents if sent.text.strip()]
        else:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        
        chunks = []
        i = 0
        
        while i < len(sentences):
            chunk = ""
            char_count = 0
            j = i
            
            # Build chunk sentence by sentence
            while j < len(sentences):
                sentence = sentences[j]
                if char_count + len(sentence) + (1 if chunk else 0) <= self.chunk_size:
                    chunk += (" " if chunk else "") + sentence
                    char_count += len(sentence) + (1 if chunk else 0)
                    j += 1
                else:
                    break
            
            if chunk:
                chunks.append(chunk.strip())
            
            # Move forward with minimal overlap to prevent getting stuck
            if j == i:  # No progress made, force move forward
                i += 1
            else:
                i = j
        
        return chunks

    def apply_overlap(self, chunks: List[str]) -> List[str]:
        """Apply character-based overlap between chunks"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # Calculate overlap in characters
            overlap_chars = min(self.chunk_overlap, len(prev_chunk) // 2)
            
            if overlap_chars > 0:
                # Find good break point for overlap (end of sentence if possible)
                overlap_text = prev_chunk[-overlap_chars:]
                
                # Try to find sentence boundary
                sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s+', overlap_text)]
                if sentence_ends:
                    # Use the last sentence boundary
                    overlap_point = sentence_ends[-1]
                    overlap_text = overlap_text[overlap_point:]
                
                # Add overlap to current chunk
                overlapped_chunk = overlap_text + " " + current_chunk
            else:
                overlapped_chunk = current_chunk
            
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks

    def hybrid_chunking(self, text: str) -> List[str]:
        """Hybrid approach: paragraph-aware + sentence-based fallback"""
        try:
            # First attempt: paragraph-aware chunking
            chunks = self.paragraph_aware_chunking(text)
            
            # Quality check: if any chunk is still too large, re-split
            final_chunks = []
            for chunk in chunks:
                if len(chunk) > self.chunk_size * 1.5:  # 50% tolerance
                    # Re-split oversized chunks
                    sub_chunks = self.sentence_based_split(chunk)
                    final_chunks.extend(sub_chunks)
                elif len(chunk) >= self.min_chunk_size:
                    final_chunks.append(chunk)
            
            return final_chunks
        except Exception as e:
            print(f"⚠️ Error in hybrid chunking: {e}")
            return self.sentence_based_split(text)


def chunk_text_semantic_v3(text, chunk_size=500, chunk_overlap=100):
    """Enhanced semantic chunking with hybrid approach"""
    chunker = AdvancedSemanticChunker(chunk_size, chunk_overlap)
    return chunker.hybrid_chunking(text)


def chunk_text_fallback(text, chunk_size=500, chunk_overlap=100):
    """Fallback to RecursiveCharacterTextSplitter with enhanced separators"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n\n",  # Multiple newlines
            "\n\n",    # Double newlines (paragraphs)
            "\n",      # Single newlines
            ". ",      # Sentence endings
            "! ",
            "? ",
            "; ",      # Clause separators
            ", ",      # Phrase separators
            " ",       # Word boundaries
            ""         # Character level
        ],
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text)


def chunk_text(text, chunk_size=500, chunk_overlap=100, use_semantic=True, force_fallback=False):
    """Main chunking function with improved error handling"""
    if force_fallback or not nlp:
        print("📝 Menggunakan fallback chunking...")
        return chunk_text_fallback(text, chunk_size, chunk_overlap)
    
    try:
        if use_semantic:
            print("🧠 Menggunakan hybrid semantic chunking...")
            return chunk_text_semantic_v3(text, chunk_size, chunk_overlap)
        else:
            print("📝 Menggunakan enhanced standard chunking...")
            return chunk_text_fallback(text, chunk_size, chunk_overlap)
    except Exception as e:
        print(f"⚠️ Error dalam semantic chunking: {e}")
        print("🔄 Beralih ke fallback chunking...")
        return chunk_text_fallback(text, chunk_size, chunk_overlap)


def load_all_texts_from_folder(folder_path):
    """Load all text files from folder with better error handling"""
    texts = []
    supported_extensions = ['.txt', '.md']
    exclude_patterns = ['.id.', '.en.', '_backup', '_temp']
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder '{folder_path}' tidak ditemukan!")
        return texts
    
    for file in os.listdir(folder_path):
        # Check if file has supported extension
        if any(file.lower().endswith(ext) for ext in supported_extensions):
            # Skip excluded patterns
            if not any(pattern in file.lower() for pattern in exclude_patterns):
                base_name = os.path.splitext(file)[0]
                file_path = os.path.join(folder_path, file)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        if text.strip():
                            texts.append({
                                "source": base_name, 
                                "text": text,
                                "file_size": len(text),
                                "file_path": file_path
                            })
                            print(f"✅ Loaded: {file} ({len(text)} chars)")
                except Exception as e:
                    print(f"❌ Error reading {file}: {e}")
    
    print(f"\n📚 Total {len(texts)} dokumen berhasil dimuat")
    return texts


def save_chunks(chunks, out_file="chunks_wbs_v5.jsonl"):
    """Save chunks with enhanced metadata"""
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                # Enhanced metadata
                chunk['char_count'] = len(chunk['content'])
                chunk['word_count'] = len(chunk['content'].split())
                chunk['sentence_count'] = len([s for s in re.split(r'[.!?]+', chunk['content']) if s.strip()])
                chunk['has_table'] = '[TABEL' in chunk['content']
                chunk['has_list'] = bool(re.search(r'^\s*[-*+•]\s|^\s*\d+\.\s', chunk['content'], re.MULTILINE))
                chunk['has_code'] = '```' in chunk['content']
                
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"💾 Chunks berhasil disimpan ke '{out_file}'")
        return True
    except Exception as e:
        print(f"❌ Error saving chunks: {e}")
        return False


def analyze_chunking_quality(chunks):
    """Enhanced quality analysis"""
    if not chunks:
        return {}
    
    char_lengths = [len(c['content']) for c in chunks]
    word_lengths = [len(c['content'].split()) for c in chunks]
    
    stats = {
        'total_chunks': len(chunks),
        'avg_char_length': sum(char_lengths) / len(chunks),
        'avg_word_length': sum(word_lengths) / len(chunks),
        'min_length': min(char_lengths),
        'max_length': max(char_lengths),
        'chunks_with_tables': sum(1 for c in chunks if '[TABEL' in c['content']),
        'chunks_with_lists': sum(1 for c in chunks if bool(re.search(r'^\s*[-*+•]\s|^\s*\d+\.\s', c['content'], re.MULTILINE))),
        'very_short_chunks': sum(1 for c in chunks if len(c['content']) < 100),
        'very_long_chunks': sum(1 for c in chunks if len(c['content']) > 800),
        'optimal_chunks': sum(1 for c in chunks if 200 <= len(c['content']) <= 700)
    }
    
    print("\n📊 Analisis Kualitas Chunking:")
    print(f"   📦 Total chunks: {stats['total_chunks']}")
    print(f"   📏 Rata-rata panjang: {stats['avg_char_length']:.1f} karakter")
    print(f"   📝 Rata-rata kata: {stats['avg_word_length']:.1f} kata")
    print(f"   📊 Range: {stats['min_length']} - {stats['max_length']} karakter")
    print(f"   🏆 Chunks optimal (200-700 char): {stats['optimal_chunks']}")
    print(f"   📋 Chunks dengan tabel: {stats['chunks_with_tables']}")
    print(f"   📝 Chunks dengan list: {stats['chunks_with_lists']}")
    print(f"   ⚠️ Chunks pendek (<100): {stats['very_short_chunks']}")
    print(f"   ⚠️ Chunks panjang (>800): {stats['very_long_chunks']}")
    
    return stats