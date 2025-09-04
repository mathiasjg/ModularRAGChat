# web_utils.py
import os
import spacy
import requests
import threading
import sqlite3
from ddgs import DDGS
from bs4 import BeautifulSoup
import re
from config import MAX_URLS, FAISS_PATH, RAW_DIR
from process_utils import process_urls
from utils import lock
from vectorstore_manager import get_vectorstore
from db_utils import add_collection, get_stored_content  # Added get_stored_content
from urllib.parse import quote
import html
import re  # Added for sanitization
from datetime import datetime  # Added for timestamp in consolidated file
from augment_utils import augment_chunk  # Import for augmentation

nlp = spacy.load("en_core_web_sm")

def sanitize_tag(name):
    # Replace spaces with '_'
    name = re.sub(r'\s+', '_', name)
    # Replace invalid path characters with '_'
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', name)
    # Strip leading/trailing whitespace (though after sub, unlikely)
    sanitized = sanitized.strip()
    # Collapse multiple '_' into single
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized

def search_web(query, site=None, timelimit=None):
    print(f"Debug: Searching web for query: {query}" + (f" site:{site}" if site else "") + (f" timelimit:{timelimit}" if timelimit else ""))
    with DDGS() as ddgs:
        search_query = query
        if 'lyrics' in query.lower():
            search_query += " site:genius.com OR site:azlyrics.com"
        if site:
            search_query += f" site:{site}"
        results = list(ddgs.text(search_query, max_results=10, timelimit=timelimit))
        urls = [result['href'] for result in results if 'href' in result]
    print(f"Debug: Found {len(urls)} URLs: {urls}")
    return urls

def run_web_collection(task_id, custom_name, query, timelimit, max_urls, use_ollama, tasks, completed_collections):
    print(f"Debug: Starting Web collection task {task_id} for query: {query}")
    conn = sqlite3.connect('crawled.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS urls
                 (url TEXT PRIMARY KEY, timestamp DATETIME, cleaned_text TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS chunks
                 (hash TEXT PRIMARY KEY, content TEXT, source TEXT, tag TEXT)''')
    try:
        c.execute("ALTER TABLE chunks ADD COLUMN tag TEXT")
        print("Debug: Added 'tag' column to chunks table.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e):
            raise e
        print("Debug: 'tag' column already exists in chunks table.")
    conn.commit()
    try:
        print("Debug: Mapping timelimit to code...")
        timelimit_code = {'Day': 'd', 'Week': 'w', 'Month': 'm', 'Year': 'y'}.get(timelimit)
        urls = search_web(query, timelimit=timelimit_code)
        all_urls = list(set(urls))[:max_urls]
        print(f"Debug: Filtered to {len(all_urls)} unique URLs.")

        raw_tag = f"web_{query}_{timelimit}"
        tag = sanitize_tag(raw_tag)  # Sanitize to prevent invalid path characters
        print(f"Debug: Sanitized tag from '{raw_tag}' to '{tag}'")
        name = custom_name or f"Web - {query} ({timelimit})"
        history = [{"role": "assistant", "content": ""}]  # Dummy
        message = query
        response = ""

        print("Debug: Starting URL processing...")
        process_gen = process_urls(all_urls, response, history, message, conn=conn, source_tag=tag, use_ollama=use_ollama)
        try:
            while True:
                next(process_gen)
        except StopIteration as e:
            sources, response, history = e.value
        print("Debug: URL processing completed.")

        # Create consolidated file after processing
        print("Debug: Creating consolidated file...")
        current_time = datetime.now()
        timestamp_str = current_time.strftime("%H-%M-%d-%m-%Y")  # Use dashes to avoid invalid characters
        prefix = "augmented-" if use_ollama else "combined-"
        consolidated_filename = f"{prefix}{tag}-{timestamp_str}.txt"
        consolidated_filepath = os.path.join(RAW_DIR, consolidated_filename)
        consolidated_content = ""
        for url in all_urls:
            content = get_stored_content(conn, url)
            if content:
                consolidated_content += f"Content from {url}:\n{content}\n\n"
            else:
                print(f"Debug: No content found for {url} in consolidated file.")
        with open(consolidated_filepath, "w", encoding="utf-8") as f:
            f.write(consolidated_content)
        print(f"Debug: Created consolidated file: {consolidated_filepath}")

        add_collection(conn, name, tag)  # Save to DB

        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['message'] = "Collection completed. New content added to vectorstore if applicable."
        tasks[task_id]['tag'] = tag
        completed_collections.append({'name': name, 'tag': tag})
        print(f"Debug: Web collection task {task_id} completed.")
    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['message'] = str(e)
        print(f"Debug: Web collection task {task_id} error: {e}")
    finally:
        conn.close()

def start_web_collection(custom_name, query, timelimit, max_urls=10, use_ollama=False, tasks=None, completed_collections=None):
    print("Debug: Starting web collection in background.")
    task_id = len(tasks)
    task = {'id': task_id, 'type': 'web', 'custom_name': custom_name, 'query': query, 'timelimit': timelimit, 'max_urls': max_urls, 'use_ollama': use_ollama, 'status': 'running', 'message': ''}
    tasks.append(task)
    threading.Thread(target=run_web_collection, args=(task_id, custom_name, query, timelimit, max_urls, use_ollama, tasks, completed_collections)).start()
    return "Web collection started in background.", tasks, completed_collections