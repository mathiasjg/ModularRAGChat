# vectorstore_utils.py
import os
from config import FAISS_PATH
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from config import MODEL_NAME

embeddings = OllamaEmbeddings(model=MODEL_NAME)

def get_vectorstore(tag=None):
    if tag:
        path = os.path.join(FAISS_PATH, tag)
    else:
        path = FAISS_PATH
    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.exists(os.path.join(path, "index.faiss")):
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    else:
        vs = FAISS.from_texts(["dummy"], embeddings)  # Dummy with non-empty text
        vs.delete([vs.index_to_docstore_id[0]])  # Remove dummy
        vs.save_local(path)
        return vs

# Global vectorstore (for backward compatibility if needed)
vectorstore = get_vectorstore()