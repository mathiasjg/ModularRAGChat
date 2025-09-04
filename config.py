import os

MODEL_NAME = "qwen2.5:7b"
RAW_DIR = "raw_contents"
os.makedirs(RAW_DIR, exist_ok=True)
FAISS_PATH = "faiss_index"
MAX_URLS = 10
MAX_DISPLAY_CHARS = 1000  # For cleaned text display
DEFAULT_CONTEXT_LENGTH = 8192  # Recommended for Qwen2.5-7B on 4090: up to 128K, but 8K-32K safe for VRAM
DEFAULT_BATCH_SIZE = 1  # For inference on 4090, can be 1-4
DEFAULT_PRECISION = "fp16"  # FP16 or BF16 for 4090 (24GB VRAM handles 7B easily)