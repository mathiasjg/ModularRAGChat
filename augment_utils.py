# augment_utils.py
import spacy
import requests
import re

nlp = spacy.load("en_core_web_sm")

def augment_chunk(chunk):
    """
    Enhance the chunk with Ollama for spelling and grammar correction only, without changing words, meaning, or structure.
    Returns the corrected text.
    """
    print("Debug: Starting Ollama correction...")
    ollama_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5:7b",
        "prompt": f"Correct spelling and grammar in this content chunk without changing any words, meaning, or structure: {chunk}. Include only the corrected text, do not add or remove anything else.",
        "stream": False
    }
    try:
        response = requests.post(ollama_url, json=payload, timeout=30)
        if response.status_code == 200:
            enhanced_text = response.json()['response'].strip()
            print(f"Debug: Ollama correction successful. Corrected text length: {len(enhanced_text)}")
            return enhanced_text
        else:
            print(f"Error correcting chunk with Ollama: {response.text}")
            return chunk  # Fallback to original
    except requests.exceptions.RequestException as e:
        print(f"Ollama request failed: {e}. Falling back to original text.")
        return chunk