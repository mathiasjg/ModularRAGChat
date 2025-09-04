# augment_utils.py
import spacy
import requests
import re

nlp = spacy.load("en_core_web_sm")

def augment_chunk(chunk):
    """
    Perform NLP on the chunk and enhance with Ollama.
    Returns the augmented text. Since augmentation is requested, Ollama is always used.
    """
    print("Debug: Starting NLP processing on chunk...")
    # NLP Processing: lemmatization, remove stop words and punctuation
    doc = nlp(chunk)
    processed_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]
    processed_text = ' '.join(processed_tokens)
    print(f"Debug: NLP processing completed. Processed text length: {len(processed_text)}")

    print("Debug: Starting Ollama enhancement...")
    ollama_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5:7b",
        "prompt": f"Enhance and correct this content chunk for clarity and accuracy: {processed_text}. Include only the corrected text, do not include a summarization of the changes or any additional text.",
        "stream": False
    }
    try:
        response = requests.post(ollama_url, json=payload, timeout=30)
        if response.status_code == 200:
            enhanced_text = response.json()['response'].strip()
            print(f"Debug: Ollama enhancement successful. Enhanced text length: {len(enhanced_text)}")
            return enhanced_text
        else:
            print(f"Error enhancing chunk with Ollama: {response.text}")
            return processed_text  # Fallback
    except requests.exceptions.RequestException as e:
        print(f"Ollama request failed: {e}. Falling back to NLP-processed text.")
        return processed_text