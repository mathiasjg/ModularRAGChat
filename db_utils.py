# db_utils.py
import os
import sqlite3
from datetime import datetime, timedelta
import hashlib
from utils import lock
import shutil
from config import FAISS_PATH

def init_db():
    print("Debug: Initializing database...")
    conn = sqlite3.connect('crawled.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS urls
                 (url TEXT PRIMARY KEY, timestamp DATETIME, cleaned_text TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS chunks
                 (hash TEXT PRIMARY KEY, content TEXT, source TEXT, tag TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS collections
                 (name TEXT PRIMARY KEY, tag TEXT)''')
    # Add tag column if not exists (for migration)
    try:
        c.execute("ALTER TABLE chunks ADD COLUMN tag TEXT")
        print("Debug: Added 'tag' column to chunks table.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e):
            raise e
        print("Debug: 'tag' column already exists in chunks table.")
    conn.commit()
    print("Debug: Database initialized.")
    return conn

def get_stored_content(conn, url):
    print(f"Debug: Checking stored content for URL: {url}")
    with lock:
        c = conn.cursor()
        c.execute("SELECT cleaned_text, timestamp FROM urls WHERE url = ?", (url,))
        result = c.fetchone()
        if result:
            text, ts_str = result
            ts = datetime.fromisoformat(ts_str)
            if datetime.now() - ts < timedelta(days=1):
                print(f"Debug: Found recent stored content for {url}")
                return text
            else:
                print(f"Debug: Stored content for {url} is outdated. Will fetch new content.")
        else:
            print(f"Debug: No stored content found for {url}. Will fetch and process new content.")
    return None

def store_content(conn, url, cleaned_text):
    print(f"Debug: Storing content for URL: {url}")
    ts = datetime.now().isoformat()
    with lock:
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO urls (url, timestamp, cleaned_text) VALUES (?, ?, ?) ",
                  (url, ts, cleaned_text))
        conn.commit()
    print(f"Debug: Content stored for {url}")

def add_chunk_if_new(conn, content, source, tag=None):
    print(f"Debug: Adding new chunk if not exists for source: {source}, tag: {tag}")
    chunk_hash = hashlib.sha256(content.encode()).hexdigest()
    with lock:
        c = conn.cursor()
        c.execute("SELECT hash FROM chunks WHERE hash = ?", (chunk_hash,))
        if not c.fetchone():
            c.execute("INSERT INTO chunks (hash, content, source, tag) VALUES (?, ?, ?, ?)",
                      (chunk_hash, content, source, tag))
            conn.commit()
            print("Debug: New chunk added.")
            return True
        else:
            print("Debug: Chunk already exists.")
    return False

def get_unique_tags(conn):
    with lock:
        c = conn.cursor()
        c.execute("SELECT DISTINCT tag FROM chunks WHERE tag IS NOT NULL")
        tags = [row[0] for row in c.fetchall()]
    return tags

def add_collection(conn, name, tag):
    with lock:
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO collections (name, tag) VALUES (?, ?)", (name, tag))
        conn.commit()

def get_collections(conn):
    with lock:
        c = conn.cursor()
        c.execute("SELECT name, tag FROM collections")
        collections = [{'name': row[0], 'tag': row[1]} for row in c.fetchall()]
    return collections

def rename_collection(conn, old_name, new_name):
    with lock:
        c = conn.cursor()
        c.execute("UPDATE collections SET name = ? WHERE name = ?", (new_name, old_name))
        conn.commit()
    print(f"Debug: Renamed collection from {old_name} to {new_name}")

def delete_collection(conn, name, tag):
    with lock:
        c = conn.cursor()
        c.execute("DELETE FROM collections WHERE name = ?", (name,))
        c.execute("DELETE FROM chunks WHERE tag = ?", (tag,))
        conn.commit()
    # Delete FAISS folder
    tag_path = os.path.join(FAISS_PATH, tag)
    if os.path.exists(tag_path):
        shutil.rmtree(tag_path)
    print(f"Debug: Deleted collection {name} with tag {tag}")