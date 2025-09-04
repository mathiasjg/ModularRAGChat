# file_utils.py
import os
import threading
import sqlite3
import PyPDF2
import spacy
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import FAISS_PATH, RAW_DIR
from db_utils import add_chunk_if_new, store_content, get_stored_content, add_collection
from vectorstore_manager import get_vectorstore
from utils import lock
from urllib.parse import quote
import html
import re  # Added for sanitization

nlp = spacy.load("en_core_web_sm")

def sanitize_tag(name):
    # Replace invalid path characters with '_'
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', name)
    # Strip leading/trailing whitespace
    sanitized = sanitized.strip()
    # Collapse multiple '_' into single
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized

def extract_text_from_file(file_path):
    if file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    elif file_path.lower().endswith('.pdf'):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
    else:
        raise ValueError("Unsupported file type. Only TXT and PDF are supported.")
    return text

def process_file_content(text, use_ollama=False):
    # NLP Processing
    doc = nlp(text)
    processed_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]
    processed_text = ' '.join(processed_tokens)

    # Chunking
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    chunk_size = 200  # Words per chunk
    chunks = []
    current_chunk = []
    current_word_count = 0
    for sent in sentences:
        word_count = len(sent.split())
        if current_word_count + word_count > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sent]
            current_word_count = word_count
        else:
            current_chunk.append(sent)
            current_word_count += word_count
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    # Optional Ollama enhancement
    if use_ollama:
        ollama_url = "http://localhost:11434/api/generate"
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            payload = {
                "model": "qwen2.5:7b",
                "prompt": f"Enhance and correct this content chunk for clarity and accuracy: {chunk}. Include only the corrected text, do not include a summarization of the changes.",
                "stream": False
            }
            try:
                response = requests.post(ollama_url, json=payload, timeout=30)
                if response.status_code == 200:
                    enhanced_text = response.json()['response']
                    enhanced_chunks.append(enhanced_text)
                else:
                    enhanced_chunks.append(chunk)  # Fallback
            except requests.exceptions.RequestException as e:
                enhanced_chunks.append(chunk)
        processed_text = '\n\n'.join(enhanced_chunks)
    return processed_text, chunks

def run_file_ingestion(task_id, custom_name, file_path, use_ollama, tasks, completed_collections):
    print(f"Starting File ingestion task {task_id} for file: {file_path}")
    conn = sqlite3.connect('crawled.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS urls
                 (url TEXT PRIMARY KEY, timestamp DATETIME, cleaned_text TEXT)''')
    print("URLs table ensured.")
    c.execute('''CREATE TABLE IF NOT EXISTS chunks
                 (hash TEXT PRIMARY KEY, content TEXT, source TEXT, tag TEXT)''')
    print("Chunks table ensured.")
    try:
        c.execute("ALTER TABLE chunks ADD COLUMN tag TEXT")
        print("Added 'tag' column to chunks table.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e):
            raise e
        print("'tag' column already exists in chunks table.")
    conn.commit()
    try:
        response = ""
        raw_tag = custom_name if custom_name else os.path.basename(file_path)
        tag = sanitize_tag(raw_tag.replace(" ", "_"))  # Sanitize after replacing spaces
        print(f"Debug: Sanitized tag from '{raw_tag}' to '{tag}'")
        name = custom_name or os.path.basename(file_path)

        # Extract text
        text = extract_text_from_file(file_path)
        response += f"Extracted text from file: {len(text)} characters.\n"

        # Process content
        processed_text, chunks = process_file_content(text, use_ollama)
        response += f"Processed text: {len(processed_text)} characters, {len(chunks)} chunks.\n"

        # Save consolidated text (since single file, it's the processed_text)
        prefix = "ollama_" if use_ollama else ""
        safe_filename = prefix + quote(os.path.basename(file_path)[:100]) + ".txt"
        filepath = os.path.join(RAW_DIR, safe_filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(processed_text)
        response += f"Saved consolidated text to {filepath}\n"

        # HTML view
        html_safe_filename = safe_filename.replace(".txt", ".html")
        html_filepath = os.path.join(RAW_DIR, html_safe_filename)
        escaped_text = html.escape(processed_text)
        html_content = f"""<html><head><style>body {{ font-family: sans-serif; padding: 20px; line-height: 1.6; max-width: 800px; margin: auto; }} pre {{ white-space: pre-wrap; word-wrap: break-word; }}</style></head><body><h1>Consolidated Content for {os.path.basename(file_path)}</h1><pre>{escaped_text}</pre></body></html>"""
        with open(html_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Store and add to vectorstore
        store_content(conn, file_path, processed_text)  # Use file_path as 'url'
        new_docs_total = 0
        new_docs = []  # Initialize new_docs here
        for chunk in chunks:
            if add_chunk_if_new(conn, chunk, file_path, tag=tag):
                metadata = {"source": file_path, "tag": tag}
                new_docs.append(Document(page_content=chunk, metadata=metadata))
                new_docs_total += 1
        if new_docs_total > 0:
            with lock:
                vs = get_vectorstore(tag)
                vs.add_documents(new_docs)
                print(f"Debug: Added {new_docs_total} documents to vectorstore for tag {tag}. ntotal after add: {vs.index.ntotal}")
                # Save the vectorstore to disk after adding documents
                save_path = os.path.join(FAISS_PATH, tag)
                vs.save_local(save_path)
                print(f"Debug: Saved vectorstore for tag {tag} to {save_path}.")
        else:
            print(f"No new documents added for tag {tag}.")

        add_collection(conn, name, tag)  # Save to DB

        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['message'] = f"Ingestion completed. {new_docs_total} new chunks added. Please refresh sources in the Chat tab."
        tasks[task_id]['tag'] = tag
        completed_collections.append({'name': name, 'tag': tag})
        print(f"File ingestion task {task_id} completed.")
    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['message'] = str(e)
        print(f"File ingestion task {task_id} error: {e}")
    finally:
        conn.close()

def start_file_ingestion(custom_name, file_path, use_ollama, tasks, completed_collections):
    task_id = len(tasks)
    task = {'id': task_id, 'type': 'file', 'custom_name': custom_name, 'file_path': file_path, 'use_ollama': use_ollama, 'status': 'running', 'message': ''}
    tasks.append(task)
    threading.Thread(target=run_file_ingestion, args=(task_id, custom_name, file_path, use_ollama, tasks, completed_collections)).start()
    return "File ingestion started in background.", tasks, completed_collections