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
import spacy
from augment_utils import augment_chunk  # Import for augmentation

nlp = spacy.load("en_core_web_sm")

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
        main_content = soup.find('main') or soup.find('article') or soup
        text = main_content.get_text(separator='\n', strip=True)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\n\t\r]+', ' ', text)
        text = re.sub(r'http\S+|www\S+|[\w\.-]+@[\w\.-]+', '', text)
        # Removed the len >20 filter to keep short lines, e.g., for lyrics
        lines = [line.strip() for line in text.split('.') if line.strip()]
        cleaned_text = '. '.join(lines)
        yield ("status", f"Debug: Step 2 completed: Cleaned text length: {len(cleaned_text)} characters.")

        # Chunking (before augmentation)
        yield ("status", "Debug: Step 3: Chunking content...")
        doc = nlp(cleaned_text)  # Use NLP for sentence splitting
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
        if len(chunks) == 0:
            yield ("status", "Debug: Warning: No chunks created from cleaned text.")
        yield ("status", f"Debug: Step 3 completed: Created {len(chunks)} chunks.")

        # Augment chunks if use_ollama is True
        if use_ollama:
            yield ("status", "Debug: Step 4: Augmenting chunks with NLP and Ollama...")
            augmented_chunks = []
            for i, chunk in enumerate(chunks):
                yield ("status", f"Debug: Augmenting chunk {i+1}/{len(chunks)}...")
                augmented_text = augment_chunk(chunk)
                augmented_chunks.append(augmented_text)
            processed_text = '\n\n'.join(augmented_chunks)
            yield ("status", "Debug: Step 4 completed: Augmentation done. Processed text length: {len(processed_text)}")
        else:
            # If no augmentation, just join chunks
            processed_text = '\n\n'.join(chunks)
            yield ("status", "Debug: Step 4 skipped: No augmentation requested. Processed text length: {len(processed_text)}")

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