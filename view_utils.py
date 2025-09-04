# view_utils.py
import os
import pandas as pd
from utils import lock
from vectorstore_manager import get_vectorstore
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from db_utils import get_stored_content
from config import MODEL_NAME
import sqlite3
from sqlalchemy import create_engine
import faissqlite  # Assume installed; if not, comment out and use basic FAISS
from langchain_ollama import OllamaEmbeddings

def view_db(conn):
    print("Viewing database...")
    with lock:
        df_urls = pd.read_sql("SELECT * FROM urls", conn)
        df_chunks = pd.read_sql("SELECT * FROM chunks", conn)
    print("Database viewed.")
    return df_urls, df_chunks

def execute_sql_query(conn, sql_query):
    try:
        df = pd.read_sql(sql_query, conn)
        return df.to_markdown(), ""
    except Exception as e:
        return "", f"Error executing query: {str(e)}"

def view_vectorstore():
    print("Viewing vectorstore...")
    with lock:
        if get_vectorstore().index.ntotal == 0:
            return "No content in vectorstore."
        docs = get_vectorstore().similarity_search(" ", k=get_vectorstore().index.ntotal)
    out = ""
    seen_sources = set()
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        if source not in seen_sources:
            seen_sources.add(source)
            out += f"**Source: {source}**\n"
        out += f"Chunk: {doc.page_content[:200]}...\n\n"
    print("Vectorstore viewed.")
    return out

def perform_similarity_search(query_text):
    embeddings = OllamaEmbeddings(model=MODEL_NAME)  # Assume same as project
    query_emb = embeddings.embed_query(query_text)
    distances, indices = get_vectorstore().index.search(query_emb, k=5)
    results = ""
    for idx in indices[0]:
        doc = get_vectorstore().similarity_search_by_index(idx)
        results += f"Document: {doc.page_content[:200]}... (Score: {distances[0][idx]})\n\n"
    return results

def refresh_tasks(tasks):
    print("Refreshing tasks...")
    return pd.DataFrame(tasks)

def show_task_detail(task_id, tasks, conn):
    print(f"Showing detail for task ID: {task_id}")
    if task_id is None or task_id < 0 or task_id >= len(tasks):
        return "Invalid task ID", "", ""
    task = tasks[task_id]
    if 'urls' not in task or 'tag' not in task:
        return "No details available for this task", "", ""
    content_out = ""
    for url in task['urls']:
        cleaned = get_stored_content(conn, url)
        if cleaned:
            content_out += f"**{url}**\n{cleaned[:500]}...\n\n"
    
    llm = OllamaLLM(model=MODEL_NAME)
    retriever = get_vectorstore(task['tag']).as_retriever(search_kwargs={"k": 5, "filter": {"tag": task['tag']}})
    
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize the following retrieved content in a concise manner:\n\n{context}"
    )
    summary_chain = create_stuff_documents_chain(llm, summary_prompt)
    summary_chain_with_docs = create_retrieval_chain(retriever, summary_chain)
    summary_response = summary_chain_with_docs.invoke({"input": ""})
    summary = summary_response["answer"]
    
    qa_prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {input}\nIf the context doesn't contain relevant information, say 'I don't know'."
    )
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    qa_chain_with_docs = create_retrieval_chain(retriever, qa_chain)
    qa_response = qa_chain_with_docs.invoke({"input": task['query']})
    answer = qa_response["answer"]
    
    print("Task detail generated.")
    return content_out, summary, answer

def view_available_tags():
    conn = sqlite3.connect('crawled.db')
    c = conn.cursor()
    c.execute("SELECT DISTINCT tag FROM chunks WHERE tag IS NOT NULL")
    tags = [row[0] for row in c.fetchall()]
    conn.close()
    if not tags:
        return "No available data sources (tags) found."
    out = "**Available Data Sources (Tags):**\n"
    for tag in tags:
        out += f"- {tag}\n"
    return out