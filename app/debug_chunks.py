# debug_chunks.py
from data_loader import load_and_split_documents

print("--- Loading and splitting documents to inspect chunks ---")

documents = load_and_split_documents()

for i, doc in enumerate(documents):
    print(f"\n----- CHUNK {i+1} -----")
    print(doc.page_content)
    print(f"----- END CHUNK (Length: {len(doc.page_content)}) -----\n")

print(f"Total chunks created: {len(documents)}")