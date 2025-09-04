# vectorstore_manager.py
import os
from config import FAISS_PATH
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from config import MODEL_NAME

embeddings = OllamaEmbeddings(model=MODEL_NAME)

def get_vectorstore(tag=None):
    if tag is None:
        # Return an empty vectorstore if no tag
        vs = FAISS.from_texts(["dummy"], embeddings)  # Dummy
        vs.delete([vs.index_to_docstore_id[0]])  # Remove dummy
        print(f"Debug: Created empty vectorstore (no tag provided). ntotal: {vs.index.ntotal}")
        return vs
    path = os.path.join(FAISS_PATH, tag)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Debug: Created new directory for vectorstore tag '{tag}' at {path}.")
    index_path = os.path.join(path, "index.faiss")
    if os.path.exists(index_path):
        vs = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        print(f"Debug: Loaded existing vectorstore for tag '{tag}' from {index_path}. ntotal: {vs.index.ntotal}")
        return vs
    else:
        vs = FAISS.from_texts(["dummy"], embeddings)  # Dummy
        vs.delete([vs.index_to_docstore_id[0]])  # Remove dummy
        vs.save_local(path)
        print(f"Debug: Created and saved new empty vectorstore for tag '{tag}' at {path}. ntotal: {vs.index.ntotal}")
        return vs