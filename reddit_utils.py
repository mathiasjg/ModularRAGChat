# reddit_utils.py
import threading
import requests
import sqlite3
from config import MAX_URLS, FAISS_PATH, RAW_DIR
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from web_utils import search_web
from db_utils import add_chunk_if_new, store_content, get_stored_content, add_collection
from vectorstore_manager import get_vectorstore
from utils import lock
from urllib.parse import quote
import os
import html
import re  # Added for sanitization

def sanitize_tag(name):
    # Replace invalid path characters with '_'
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', name)
    # Strip leading/trailing whitespace
    sanitized = sanitized.strip()
    # Collapse multiple '_' into single
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized

def run_reddit_collection(task_id, custom_name, query, timelimit, max_urls, use_ollama, max_comments, tasks, completed_collections):
    print(f"Starting Reddit collection task {task_id} for query: {query}")
    conn = sqlite3.connect('crawled.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS urls
                 (url TEXT PRIMARY KEY, timestamp DATETIME, cleaned_text TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS chunks
                 (hash TEXT PRIMARY KEY, content TEXT, source TEXT, tag TEXT)''')
    try:
        c.execute("ALTER TABLE chunks ADD COLUMN tag TEXT")
        print("Added 'tag' column to chunks table.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e):
            raise e
        print("'tag' column already exists in chunks table.")
    conn.commit()
    try:
        site = "reddit.com"
        timelimit_code = {'Day': 'd', 'Week': 'w', 'Month': 'm', 'Year': 'y'}.get(timelimit)
        urls = search_web(query, site=site, timelimit=timelimit_code)
        all_urls = list(set(urls))[:max_urls]

        raw_tag = f"reddit_{query}_{timelimit}"
        tag = sanitize_tag(raw_tag)  # Sanitize to prevent invalid path characters
        print(f"Debug: Sanitized tag from '{raw_tag}' to '{tag}'")
        name = custom_name or f"Reddit - {query} ({timelimit})"
        history = [{"role": "assistant", "content": ""}]  # Dummy
        message = query
        response = ""

        # For Reddit, fetch post + comments
        for url in all_urls:
            try:
                json_url = url + '.json'
                json_response = requests.get(json_url, headers={'User-Agent': 'Mozilla/5.0'})
                if json_response.status_code == 200:
                    data = json_response.json()
                    post_text = data[0]['data']['children'][0]['data']['selftext']
                    comments = data[1]['data']['children']
                    comment_texts = [comment['data']['body'] for comment in comments[:max_comments] if 'body' in comment['data']]
                    full_text = post_text + " ".join(comment_texts)
                    # Save to raw_contents
                    prefix = "ollama_" if use_ollama else ""
                    safe_filename = prefix + quote(url.replace("https://", "").replace("http://", "").replace("/", "_")[:100]) + ".txt"
                    filepath = os.path.join(RAW_DIR, safe_filename)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(full_text)

                    # HTML view
                    html_safe_filename = safe_filename.replace(".txt", ".html")
                    html_filepath = os.path.join(RAW_DIR, html_safe_filename)
                    escaped_text = html.escape(full_text)
                    html_content = f"""<html><head><style>body {{ font-family: sans-serif; padding: 20px; line-height: 1.6; max-width: 800px; margin: auto; }} pre {{ white-space: pre-wrap; word-wrap: break-word; }}</style></head><body><h1>Extracted Content for {url}</h1><pre>{escaped_text}</pre></body></html>"""
                    with open(html_filepath, "w", encoding="utf-8") as f:
                        f.write(html_content)

                    store_content(conn, url, full_text)
                    response += f"Fetched post and comments for {url}\n"
                else:
                    response += f"Failed to fetch for {url}\n"
            except Exception as e:
                response += f"Error for {url}: {e}\n"

        # Then process to chunks
        new_docs_total = 0
        for url in all_urls:
            content = get_stored_content(conn, url)
            if content:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                chunks = text_splitter.split_text(content)
                new_docs = []
                for chunk in chunks:
                    if add_chunk_if_new(conn, chunk, url, tag=tag):
                        metadata = {"source": url, "tag": tag}
                        new_docs.append(Document(page_content=chunk, metadata=metadata))
                if new_docs:
                    with lock:
                        vs = get_vectorstore(tag)
                        vs.add_documents(new_docs)
                        print(f"Debug: Added {len(new_docs)} documents to vectorstore for tag {tag}. ntotal after add: {vs.index.ntotal}")
                        # Save the vectorstore to disk after adding documents
                        save_path = os.path.join(FAISS_PATH, tag)
                        vs.save_local(save_path)
                        print(f"Debug: Saved vectorstore for tag {tag} to {save_path}.")
                    new_docs_total += len(new_docs)

        add_collection(conn, name, tag)  # Save to DB

        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['message'] = f"Collection completed. {new_docs_total} new chunks added."
        tasks[task_id]['tag'] = tag
        completed_collections.append({'name': name, 'tag': tag})
        print(f"Reddit collection task {task_id} completed.")
    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['message'] = str(e)
        print(f"Reddit collection task {task_id} error: {e}")
    finally:
        conn.close()

def start_reddit_collection(custom_name, query, timelimit, max_urls=10, use_ollama=False, max_comments=50, tasks=None, completed_collections=None):
    task_id = len(tasks)
    task = {'id': task_id, 'type': 'reddit', 'custom_name': custom_name, 'query': query, 'timelimit': timelimit, 'max_urls': max_urls, 'use_ollama': use_ollama, 'max_comments': max_comments, 'status': 'running', 'message': ''}
    tasks.append(task)
    threading.Thread(target=run_reddit_collection, args=(task_id, custom_name, query, timelimit, max_urls, use_ollama, max_comments, tasks, completed_collections)).start()
    return "Reddit collection started in background.", tasks, completed_collections