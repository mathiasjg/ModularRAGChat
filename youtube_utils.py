# youtube_utils.py
import time
import spacy
import requests
import threading
import sqlite3
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import MAX_URLS, FAISS_PATH, RAW_DIR
from web_utils import search_web
from db_utils import add_chunk_if_new, store_content, get_stored_content, add_collection
from vectorstore_manager import get_vectorstore
from utils import lock
from urllib.parse import quote
import html
import os
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

def fetch_youtube_transcript(url, use_ollama=False):
    yield ("status", f"Fetching YouTube transcript for {url} with Ollama: {use_ollama}")
    options = Options()
    options.add_argument('--disable-notifications')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_argument('--log-level=3')
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    wait = WebDriverWait(driver, 10)  # Reduced timeout for faster failure

    try:
        yield ("status", "Step 1/8: Navigating to URL...")
        driver.get(url)
        yield ("status", "Step 1/8 completed: Page loaded.")

        # Handle consent popup with wait instead of sleep
        try:
            yield ("status", "Step 2/8: Checking for consent popup...")
            wait.until(EC.frame_to_be_available_and_switch_to_it((By.CSS_SELECTOR, "iframe[src*='consent.youtube.com']")))
            accept_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Accept all') or contains(., 'I agree') or contains(@aria-label, 'Accept')]")))
            try:
                accept_button.click()
            except ElementClickInterceptedException:
                driver.execute_script("arguments[0].click();", accept_button)
            driver.switch_to.default_content()
            yield ("status", "Step 2/8 completed: Consent popup handled.")
        except TimeoutException:
            yield ("status", "Step 2/8: No consent popup found or already handled.")

        # Expand description
        try:
            yield ("status", "Step 3/8: Expanding description...")
            expander_xpath = "//ytd-text-inline-expander[@id='description-inline-expander']//tp-yt-paper-button[@id='expand']"
            description_expander = wait.until(EC.element_to_be_clickable((By.XPATH, expander_xpath)))
            try:
                description_expander.click()
            except ElementClickInterceptedException:
                driver.execute_script("arguments[0].click();", description_expander)
            yield ("status", "Step 3/8 completed: Description expanded.")
        except TimeoutException:
            yield ("status", "Step 3/8: Description may already be expanded or not found.")

        # Click 'Show transcript'
        try:
            yield ("status", "Step 4/8: Clicking 'Show transcript' button...")
            transcript_button_xpath = "//ytd-video-description-transcript-section-renderer//ytd-button-renderer/yt-button-shape/button"
            transcript_button = wait.until(EC.element_to_be_clickable((By.XPATH, transcript_button_xpath)))
            try:
                transcript_button.click()
            except ElementClickInterceptedException:
                driver.execute_script("arguments[0].click();", transcript_button)
            yield ("status", "Step 4/8 completed: 'Show transcript' clicked.")
        except TimeoutException:
            yield ("status", "Step 4/8: No 'Show transcript' button found in description. Trying fallback method...")
            # Fallback to more actions menu
            try:
                more_actions = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[@aria-label="More actions"]')))
                try:
                    more_actions.click()
                except ElementClickInterceptedException:
                    driver.execute_script("arguments[0].click();", more_actions)
                transcript_item = wait.until(EC.element_to_be_clickable((By.XPATH, '//tp-yt-paper-item[contains(text(), "Show transcript")]')))
                try:
                    transcript_item.click()
                except ElementClickInterceptedException:
                    driver.execute_script("arguments[0].click();", transcript_item)
                yield ("status", "Step 4/8 completed: 'Show transcript' clicked via fallback.")
            except TimeoutException:
                yield ("status", f"Error: Could not find 'Show transcript' for {url}")
                return

        # Wait for transcript panel
        try:
            yield ("status", "Step 5/8: Waiting for transcript panel to load...")
            transcript_title_xpath = "//ytd-engagement-panel-section-list-renderer[@target-id='engagement-panel-searchable-transcript']//yt-formatted-string[@id='title-text']"
            wait.until(EC.visibility_of_element_located((By.XPATH, transcript_title_xpath)))
            yield ("status", "Step 5/8 completed: Transcript panel loaded.")
        except TimeoutException:
            yield ("status", f"Error: Transcript panel did not load for {url}")
            return

        # Extract transcript
        try:
            yield ("status", "Step 6/8: Extracting transcript text...")
            transcript_elements_xpath = "//ytd-engagement-panel-section-list-renderer[@target-id='engagement-panel-searchable-transcript']//ytd-transcript-segment-renderer//yt-formatted-string"
            transcript_elements = driver.find_elements(By.XPATH, transcript_elements_xpath)
            if not transcript_elements:
                transcript_elements = driver.find_elements(By.CSS_SELECTOR, ".ytd-transcript-segment-renderer .ytd-transcript-segment-text")
            if not transcript_elements:
                transcript_elements = driver.find_elements(By.CSS_SELECTOR, ".cue.style-scope.ytd-transcript-body-renderer")
            if not transcript_elements:
                yield ("status", f"Error: No transcript elements found for {url}")
                return
            transcript_text = '\n'.join([elem.text.strip() for elem in transcript_elements if elem.text.strip()])
            yield ("status", f"Step 6/8 completed: Extracted {len(transcript_text.splitlines())} lines of transcript.")
        except NoSuchElementException:
            yield ("status", f"Error: Transcript elements not found for {url}")
            return

        # NLP Processing
        yield ("status", "Step 7/8: Processing transcript with NLP...")
        doc = nlp(transcript_text)
        processed_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]
        processed_text = ' '.join(processed_tokens)
        yield ("status", "Step 7/8 completed: NLP processing done.")

        # Chunking
        yield ("status", "Step 8/8: Chunking transcript...")
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
        yield ("status", f"Step 8 completed: Created {len(chunks)} chunks.")

        # Optional Ollama enhancement
        if use_ollama:
            yield ("status", "Enhancing chunks with Ollama...")
            ollama_url = "http://localhost:11434/api/generate"
            enhanced_chunks = []
            for i, chunk in enumerate(chunks):
                yield ("status", f"Enhancing chunk {i+1}/{len(chunks)}...")
                payload = {
                    "model": "qwen2.5:7b",
                    "prompt": f"Enhance and correct this transcript chunk for clarity and accuracy: {chunk}. Include only the corrected text, do not include a summarization of the changes.",
                    "stream": False
                }
                try:
                    response = requests.post(ollama_url, json=payload, timeout=30)
                    if response.status_code == 200:
                        enhanced_text = response.json()['response']
                        enhanced_chunks.append(enhanced_text)
                        yield ("status", f"Chunk {i+1} enhanced.")
                    else:
                        yield ("status", f"Error enhancing chunk {i+1} with Ollama: {response.text}")
                        enhanced_chunks.append(chunk)  # Fallback
                except requests.exceptions.RequestException as e:
                    yield ("status", f"Ollama request failed for chunk {i+1}: {e}. Falling back to original chunk.")
                    enhanced_chunks.append(chunk)
            processed_text = '\n\n'.join(enhanced_chunks)
            yield ("status", "Step 9 completed: Ollama enhancement done.")
        else:
            processed_text = ' '.join(processed_tokens)

        # Save processed text to raw_contents
        prefix = "ollama_" if use_ollama else ""
        safe_filename = prefix + quote(url.replace("https://", "").replace("http://", "").replace("/", "_")[:100]) + ".txt"
        filepath = os.path.join(RAW_DIR, safe_filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(processed_text)
        yield ("status", f"Saved processed transcript to {filepath}")

        # HTML view
        html_safe_filename = safe_filename.replace(".txt", ".html")
        html_filepath = os.path.join(RAW_DIR, html_safe_filename)
        escaped_text = html.escape(processed_text)
        html_content = f"""<html><head><style>body {{ font-family: sans-serif; padding: 20px; line-height: 1.6; max-width: 800px; margin: auto; }} pre {{ white-space: pre-wrap; word-wrap: break-word; }}</style></head><body><h1>Processed Transcript for {url}</h1><pre>{escaped_text}</pre></body></html>"""
        with open(html_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        yield ("transcript", processed_text)

    finally:
        yield ("status", "Final Step: Closing browser...")
        driver.quit()
        yield ("status", "Browser closed.")

def run_youtube_collection(task_id, custom_name, query, url_list, max_videos, use_ollama, tasks, completed_collections):
    print(f"Starting YouTube collection task {task_id} for query: {query} or URLs: {url_list}")
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
        response = ""
        history = [{"role": "assistant", "content": ""}]  # Dummy
        message = query or "custom_urls"
        raw_tag = "youtube_" + message.replace(" ", "_")
        tag = sanitize_tag(raw_tag)  # Sanitize to prevent invalid path characters
        print(f"Debug: Sanitized tag from '{raw_tag}' to '{tag}'")
        name = custom_name or f"YouTube - {message}"

        if query:
            youtube_urls = search_web(query, site="youtube.com")
            all_urls = list(set(youtube_urls))[:max_videos]  # Limit to max_videos
        else:
            all_urls = [url.strip() for url in url_list if url.strip()][:max_videos]  # Limit to max_videos

        transcripts = []
        new_docs_total = 0
        for i, url in enumerate(all_urls):
            print(f"Processing URL {i+1}/{len(all_urls)}: {url}")
            tasks[task_id]['message'] = f"Processing URL {i+1}/{len(all_urls)}: {url}"
            stored = get_stored_content(conn, url)
            if stored:
                response += f"Using stored transcript for {url}\n"
                transcript = stored
            else:
                gen = fetch_youtube_transcript(url, use_ollama=use_ollama)
                transcript = None
                for item in gen:
                    item_type, value = item
                    if item_type == "status":
                        response += value + "\n"
                    elif item_type == "transcript":
                        transcript = value
                if transcript:
                    store_content(conn, url, transcript)

            if transcript:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                chunks = text_splitter.split_text(transcript)
                new_docs = []
                for chunk in chunks:
                    if add_chunk_if_new(conn, chunk, url, tag=tag):
                        metadata = {"source": url, "tag": tag}
                        new_docs.append(Document(page_content=chunk, metadata=metadata))
                if new_docs:
                    with lock:
                        vs = get_vectorstore(tag)
                        vs.add_documents(new_docs)
                        print(f"Added {len(new_docs)} documents to vectorstore for tag {tag}.")
                    new_docs_total += len(new_docs)

        add_collection(conn, name, tag)  # Save to DB

        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['message'] = f"Collection completed. {new_docs_total} new chunks added. Please refresh sources in the Chat tab."
        tasks[task_id]['tag'] = tag
        completed_collections.append({'name': name, 'tag': tag})
        print(f"YouTube collection task {task_id} completed.")
    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['message'] = str(e)
        print(f"YouTube collection task {task_id} error: {e}")
    finally:
        conn.close()

def start_youtube_collection(custom_name, mode, query, url_list, max_videos=10, use_ollama=False, tasks=None, completed_collections=None):
    task_id = len(tasks)
    task = {'id': task_id, 'type': 'youtube', 'custom_name': custom_name, 'mode': mode, 'query': query, 'url_list': url_list, 'max_videos': max_videos, 'use_ollama': use_ollama, 'status': 'running', 'message': ''}
    tasks.append(task)
    threading.Thread(target=run_youtube_collection, args=(task_id, custom_name, query, url_list, max_videos, use_ollama, tasks, completed_collections)).start()
    return "YouTube collection started in background.", tasks, completed_collections