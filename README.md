Of course. Here is a README file that explains the purpose of each file in your project.

-----

## README.md

### Simple RAG Project

This is a simple Retrieval-Augmented Generation (RAG) project that uses a local language model (via Ollama) to answer questions based on a provided document.

### How to Run

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the application**:
    ```bash
    python main.py
    ```

### File Descriptions

  * `main.py`: The main file to run the interactive Q\&A application.
  * `config.py`: Central configuration for model settings, file paths, and other parameters.
  * `retriever.py`: Module responsible for loading the document, creating chunks, and building the search index (FAISS).
  * `rag_chain_builder.py`: Creates the main LangChain chain that combines the prompt, context, and LLM to generate an answer.
  * `history_rewriter.py`: Manages conversation memory by rewriting follow-up questions to be standalone.
  * `reranker.py`: Filters and re-ranks the search results from the retriever to get the most relevant context.
  * `filter.py`: A smart filter to check if the found context is relevant to the question before generating an answer.
  * `requirements.txt`: A list of all the Python packages required to run the project.