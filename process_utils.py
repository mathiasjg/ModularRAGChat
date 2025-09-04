# process_utils.py
import os
from config import RAW_DIR, MAX_DISPLAY_CHARS, FAISS_PATH
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from urllib.parse import quote
from db_utils import get_stored_content, store_content, add_chunk_if_new
from vectorstore_manager import get_vectorstore
from utils import lock
import html
import requests
from bs4 import BeautifulSoup
import re
from augment_utils import augment_chunk  # Import for augmentation

# Removed spaCy import and usage to avoid any potential modification during extraction

def clean_web_content(url, use_ollama=False):
    yield ("status", f"Debug: Fetching and cleaning URL: {url} with Ollama: {use_ollama}")
    try:
        yield ("status", "Debug: Step 1: Sending request to URL...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        html = response.text
        yield ("status", "Debug: Step 1 completed: Response received. Raw HTML length: {len(html)}")

        yield ("status", "Debug: Step 2: Parsing HTML with BeautifulSoup...")
        soup = BeautifulSoup(html, 'html.parser')
        for elem in soup.select('script, style, nav, header, footer, .ad, .advert, iframe, noscript'):
            elem.extract()

        # Special handling for lyrics sites to preserve full structure
        if 'genius.com' in url:
            yield ("status", "Debug: Detected Genius.com - extracting full lyrics with structure preserved.")
            lyrics_divs = soup.find_all('div', class_=re.compile(r'Lyrics__Container'))
            text = '\n'.join([div.get_text(separator='\n', strip=False) for div in lyrics_divs])
        elif 'azlyrics.com' in url:
            yield ("status", "Debug: Detected AZLyrics.com - extracting full lyrics with structure preserved.")
            lyrics_div = soup.find('div', class_='ringtone').find_next_sibling('div') if soup.find('div', class_='ringtone') else soup.find('div', id='lyrics-body-text')
            text = lyrics_div.get_text(separator='\n', strip=False) if lyrics_div else ''
        else:
            yield ("status", "Debug: General content - extracting full text with structure preserved.")
            main_content = soup.find('main') or soup.find('article') or soup
            text = main_content.get_text(separator='\n', strip=False)

        # Extremely minimal cleanup: remove URLs/emails only, preserve all whitespace and structure
        cleaned_text = re.sub(r'http\S+|www\S+|[\w\.-]+@[\w\.-]+', '', text)  # Remove URLs/emails
        cleaned_text = cleaned_text.strip()  # Trim leading/trailing whitespace only
        yield ("status", f"Debug: Step 2 completed: Cleaned text length: {len(cleaned_text)} characters. Preview: {cleaned_text[:200]}...")

        # Chunking (before augmentation) - use character-based splitter to preserve structure
        yield ("status", "Debug: Step 3: Chunking content...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""],  # Prioritize newlines for structure
            keep_separator=True
        )
        chunks = text_splitter.split_text(cleaned_text)
        if len(chunks) == 0:
            yield ("status", "Debug: Warning: No chunks created from cleaned text.")
        yield ("status", f"Debug: Step 3 completed: Created {len(chunks)} chunks. First chunk preview: {chunks[0][:100]}..." if chunks else "No chunks.")

        # Augment chunks if use_ollama is True
        if use_ollama:
            yield ("status", "Debug: Step 4: Augmenting chunks with Ollama correction...")
            augmented_chunks = []
            for i, chunk in enumerate(chunks):
                yield ("status", f"Debug: Augmenting chunk {i+1}/{len(chunks)}...")
                augmented_text = augment_chunk(chunk)
                augmented_chunks.append(augmented_text)
            processed_text = '\n\n'.join(augmented_chunks)
            yield ("status", f"Debug: Step 4 completed: Augmentation done. Processed text length: {len(processed_text)}. Preview: {processed_text[:200]}...")
        else:
            # If no augmentation, just join chunks with newlines to preserve structure
            processed_text = '\n\n'.join(chunks)
            yield ("status", f"Debug: Step 4 skipped: No augmentation requested. Processed text length: {len(processed_text)}. Preview: {processed_text[:200]}...")

        yield ("content", processed_text)
    except Exception as e:
        yield ("status", f"Debug: Error cleaning {url}: {e}")
        yield ("content", None)

def process_urls(all_urls, response, history, message, is_chat=True, conn=None, source_tag=None, use_ollama=False):
    print(f"Debug: Starting process_urls with {len(all_urls)} URLs. is_chat: {is_chat}, source_tag: {source_tag}, use_ollama: {use_ollama}")
    documents = []
    sources = []
    for url in all_urls:
        stored = get_stored_content(conn, url)
        if stored:
            response += f"Debug: Using stored content for {url}\n"
            if is_chat:
                history[-1]['content'] = response
                yield history, ""
            cleaned_text = stored
        else:
            response += f"Debug: Fetching and processing {url}...\n"
            if is_chat:
                history[-1]['content'] = response
                yield history, ""
            gen = clean_web_content(url, use_ollama=use_ollama)
            cleaned_text = None
            for item in gen:
                item_type, value = item
                if item_type == "status":
                    response += value + "\n"
                    if is_chat:
                        history[-1]['content'] = response
                        yield history, ""
                elif item_type == "content":
                    cleaned_text = value

        if cleaned_text:
            sources.append(url)
            store_content(conn, url, cleaned_text)

            prefix = "ollama_" if use_ollama else ""
            safe_filename = prefix + quote(url.replace("https://", "").replace("http://", "").replace("/", "_")[:100]) + ".txt"
            filepath = os.path.join(RAW_DIR, safe_filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            response += f"[Download full raw text for {url}](/file={safe_filename})\n\n"
            if is_chat:
                history[-1]['content'] = response
                yield history, ""

            # Save as HTML for viewing
            html_safe_filename = safe_filename.replace(".txt", ".html")
            html_filepath = os.path.join(RAW_DIR, html_safe_filename)
            escaped_text = html.escape(cleaned_text)
            html_content = f"""
<html>
<head>
    <style>
        body {{ font-family: sans-serif; padding: 20px; line-height: 1.6; max-width: 800px; margin: auto; }}
        pre {{ white-space: pre-wrap; word-wrap: break-word; }}
    </style>
</head>
<body>
    <h1>Processed Text for {url}</h1>
    <pre>{escaped_text}</pre>
</body>
</html>
"""
            with open(html_filepath, "w", encoding="utf-8") as f:
                f.write(html_content)

            response += f'<a href="/file={html_safe_filename}" target="_blank">View Processed Text for {url} in new tab</a>\n\n'
            if is_chat:
                history[-1]['content'] = response
                yield history, ""

            display_text = cleaned_text[:MAX_DISPLAY_CHARS] + "..." if len(cleaned_text) > MAX_DISPLAY_CHARS else cleaned_text
            response += f"Cleaned content from {url} (preview):\n{display_text}\n\n"
            if is_chat:
                history[-1]['content'] = response
                yield history, ""

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = text_splitter.split_text(cleaned_text)
            new_docs = []
            for chunk in chunks:
                if add_chunk_if_new(conn, chunk, url, tag=source_tag):
                    metadata = {"source": url, "tag": source_tag}
                    if 'lyrics' in message.lower():
                        metadata["source_type"] = "lyrics"
                    new_docs.append(Document(page_content=chunk, metadata=metadata))

            if new_docs:
                with lock:
                    vs = get_vectorstore(source_tag)
                    vs.add_documents(new_docs)
                    print(f"Debug: Added {len(new_docs)} documents to vectorstore for tag {source_tag}. ntotal after add: {vs.index.ntotal}")
                    # Save the vectorstore to disk after adding documents
                    save_path = os.path.join(FAISS_PATH, source_tag)
                    vs.save_local(save_path)
                    print(f"Debug: Saved vectorstore for tag {source_tag} to {save_path}.")
                documents.extend(new_docs)

    response += f"Number of new document chunks added: {len(documents)}\n\n"
    if is_chat:
        history[-1]['content'] = response
        yield history, ""

    print("Debug: process_urls completed.")
    return sources, response, history