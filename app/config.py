import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_openrouter import OpenRouterLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# --- MAIN SWITCH ---
LLM_PROVIDER = "ollama"

# --- PROVIDER-SPECIFIC CONFIGURATION ---
MODEL_NAME = None
llm = None

# --- DEDICATED REWRITER LLM ---
# This uses a more powerful model to reliably handle instructions
rewriter_llm = OllamaLLM(model="llama3:8b", temperature=0)
print("✨ Question Rewriter is configured with local model: llama3:8b")


if LLM_PROVIDER == "ollama":
    MODEL_NAME = "llama3:8b"
    llm = OllamaLLM(model=MODEL_NAME)
    print("🧠 Using local Ollama model.")

elif LLM_PROVIDER == "openrouter":
    MODEL_NAME = "google/gemini-flash-1.5"
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    # Use LangChain's ChatOpenAI class with OpenRouter config
    llm = OpenRouterLLM(
        model="google/gemini-flash-1.5",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        headers={
            "HTTP-Referer": os.getenv("YOUR_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("YOUR_SITE_NAME", "LangChainApp")
        }
    )
    print("☁️ Using OpenRouter API.")

else:
    raise ValueError("❌ Invalid LLM_PROVIDER specified. Use 'ollama' or 'openrouter'.")

# --- PROMPT TEMPLATE ---
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

# --- LLM CHAIN ---
llm_chain = prompt | llm | StrOutputParser()

# --- EXTRA CONFIG (still available for use elsewhere) ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
KNOWLEDGE_BASE_PATH = "../ekstrak/my_knowledge.txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
USE_ADVANCED_CHUNKER = False

# import os
# from langchain_ollama import OllamaLLM
# from langchain_openrouter import OpenRouterLLM 
# from dotenv import load_dotenv

# # Load environment variables from .env file (optional, for OpenRouter)
# load_dotenv()

# # --- MAIN SWITCH ---
# # Choose your provider: "ollama" or "openrouter"
# LLM_PROVIDER = "openrouter" 

# # --- PROVIDER-SPECIFIC CONFIGURATION ---
# if LLM_PROVIDER == "ollama":
#     # Use a local model
#     MODEL_NAME = "qwen2:1.5b"
#     llm = OllamaLLM(model=MODEL_NAME)
#     print("🧠 Using local Ollama model.")
# elif LLM_PROVIDER == "openrouter":
#     # Use OpenRouter API
#     MODEL_NAME = "google/gemini-flash-1.5"
#     OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
#     llm = OpenRouterLLM(
#         model_name=MODEL_NAME,
#         openrouter_api_key=OPENROUTER_API_KEY
#     )
#     print("☁️ Using OpenRouter API.")
# else:
#     raise ValueError("Invalid LLM_PROVIDER specified in config.py")

# # --- EMBEDDING AND KNOWLEDGE BASE ---
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# KNOWLEDGE_BASE_PATH = "ekstrak/my_knowledge.txt"

# # --- CHUNKER CONFIGURATION ---
# CHUNK_SIZE = 500
# CHUNK_OVERLAP = 100
# USE_ADVANCED_CHUNKER = True