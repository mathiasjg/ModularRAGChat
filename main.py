# main.py
# main.py
import os
import gradio as gr
from db_utils import init_db, get_collections, rename_collection, delete_collection
from chat_utils import chat_bot
from web_utils import start_web_collection
from youtube_utils import start_youtube_collection
from reddit_utils import start_reddit_collection
from subreddit_utils import start_subreddit_collection
from file_utils import start_file_ingestion
from view_utils import view_db, execute_sql_query, view_vectorstore, perform_similarity_search, refresh_tasks, show_task_detail, view_available_tags
from config import MODEL_NAME
import pandas as pd

conn = init_db()

def load_completed_collections():
    return get_collections(conn)

def update_dropdown():
    completed_collections = load_completed_collections()
    print("Debug: Loaded collections:", completed_collections)  # Debug
    return gr.update(choices=["No RAG"] + [c['name'] for c in completed_collections], value="No RAG"), completed_collections

def submit_chat(m, h, s, completed_collections):
    tag = next((c['tag'] for c in completed_collections if c['name'] == s), None) if s != "No RAG" else None
    print(f"Debug: Submitting chat with source: {s}, tag: {tag}")  # Debug
    gen = chat_bot(m, h, conn=conn, selected_source=s, selected_tag=tag)
    for chat_out, msg_out in gen:
        yield chat_out, msg_out

def toggle_youtube_inputs(mode):
    if mode == "Search Query":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

def load_data_sources():
    collections = load_completed_collections()
    df = pd.DataFrame(collections)
    return df

def select_data_source(selected_row, collections):
    if selected_row is None:
        return "", pd.DataFrame()
    name = selected_row['name']
    tag = selected_row['tag']
    chunks_df = pd.read_sql(f"SELECT * FROM chunks WHERE tag = '{tag}'", conn)
    return name, chunks_df

def rename_data_source(selected_row, new_name, collections):
    if selected_row is None:
        return "No source selected"
    old_name = selected_row['name']
    tag = selected_row['tag']
    rename_collection(conn, old_name, new_name)
    # Update state
    for c in collections:
        if c['name'] == old_name:
            c['name'] = new_name
    return "Renamed successfully. Refresh to see changes."

def confirm_delete_data_source(selected_row, collections):
    if selected_row is None:
        return "No source selected"
    name = selected_row['name']
    tag = selected_row['tag']
    delete_collection(conn, name, tag)
    # Update state
    collections = [c for c in collections if c['name'] != name]
    return "Deleted successfully. Refresh to see changes."

with gr.Blocks(title="Enhanced RAG Chatbot with Qwen 2.5:7B", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# Enhanced RAG Chatbot\nCurrent Model: {MODEL_NAME}")
    
    collection_tasks_state = gr.State([])
    completed_collections_state = gr.State([])
    
    with gr.Tabs():
        with gr.Tab("Chat"):
            gr.Markdown("""**Instructions:** Select a RAG source below to augment your query with pre-collected data.""")
            source_dropdown = gr.Dropdown(label="Select RAG Source (optional)", choices=[], value=None, interactive=True)
            refresh_sources_btn = gr.Button("Refresh Sources")
            chatbot = gr.Chatbot(height=500, type="messages")
            msg = gr.Textbox(placeholder="Enter your prompt here...", show_label=False)
            with gr.Row():
                submit_btn = gr.Button("Submit")
                clear = gr.Button("Clear")
            demo.load(update_dropdown, outputs=[source_dropdown, completed_collections_state])
            refresh_sources_btn.click(update_dropdown, outputs=[source_dropdown, completed_collections_state])
            submit_btn.click(submit_chat, [msg, chatbot, source_dropdown, completed_collections_state], [chatbot, msg])
            clear.click(lambda: None, None, chatbot, queue=False)
            clear.click(lambda: "", None, msg, queue=False)
        
        with gr.Tab("Web Collection"):
            name_input_web = gr.Textbox(label="Data Source Name (optional)")
            query_input_web = gr.Textbox(label="Search Query")
            timelimit_input_web = gr.Dropdown(["Day", "Week", "Month", "Year"], label="Time Limit")
            max_urls_input_web = gr.Number(label="Max URLs", value=10)
            use_ollama_web = gr.Checkbox(label="Use Ollama Augmentation", value=False)
            collect_btn_web = gr.Button("Start Collection")
            status_web = gr.Textbox(label="Status")
            collect_btn_web.click(lambda n, q, tl, mu, u, ts, cs: start_web_collection(n, q, tl, mu, u, ts, cs), [name_input_web, query_input_web, timelimit_input_web, max_urls_input_web, use_ollama_web, collection_tasks_state, completed_collections_state], [status_web, collection_tasks_state, completed_collections_state])
        
        with gr.Tab("YouTube Collection"):
            name_input_yt = gr.Textbox(label="Data Source Name (optional)")
            mode_yt = gr.Radio(["Search Query", "List of URLs"], label="Collection Mode", value="Search Query")
            query_input_yt = gr.Textbox(label="Search Query", visible=True)
            urls_input_yt = gr.TextArea(label="List of URLs (one per line)", visible=False)
            max_videos_input_yt = gr.Number(label="Max Videos", value=10)
            use_ollama_yt = gr.Checkbox(label="Use Ollama Augmentation", value=False)
            collect_btn_yt = gr.Button("Start Collection")
            status_yt = gr.Textbox(label="Status")
            mode_yt.change(toggle_youtube_inputs, mode_yt, [query_input_yt, urls_input_yt])
            collect_btn_yt.click(lambda n, m, q, urls, mv, u, ts, cs: start_youtube_collection(n, m, q if m == "Search Query" else None, urls.splitlines() if m == "List of URLs" else None, mv, u, ts, cs), [name_input_yt, mode_yt, query_input_yt, urls_input_yt, max_videos_input_yt, use_ollama_yt, collection_tasks_state, completed_collections_state], [status_yt, collection_tasks_state, completed_collections_state])
        
        with gr.Tab("Reddit Collection"):
            name_input_reddit = gr.Textbox(label="Data Source Name (optional)")
            query_input_reddit = gr.Textbox(label="Search Query")
            timelimit_input_reddit = gr.Dropdown(["Day", "Week", "Month", "Year"], label="Time Limit")
            max_urls_input_reddit = gr.Number(label="Max URLs", value=10)
            use_ollama_reddit = gr.Checkbox(label="Use Ollama Augmentation", value=False)
            max_comments_reddit = gr.Number(label="Max Comments per Thread", value=50)
            collect_btn_reddit = gr.Button("Start Collection")
            status_reddit = gr.Textbox(label="Status")
            collect_btn_reddit.click(lambda n, q, tl, mu, u, mc, ts, cs: start_reddit_collection(n, q, tl, mu, u, mc, ts, cs), [name_input_reddit, query_input_reddit, timelimit_input_reddit, max_urls_input_reddit, use_ollama_reddit, max_comments_reddit, collection_tasks_state, completed_collections_state], [status_reddit, collection_tasks_state, completed_collections_state])
        
        with gr.Tab("Subreddit Collection"):
            name_input_sub = gr.Textbox(label="Data Source Name (optional)")
            subreddit_input = gr.Textbox(label="Subreddit Name (e.g., wallstreetbets)")
            timelimit_input_sub = gr.Dropdown(["Day", "Week", "Month", "Year"], label="Time Limit")
            query_input_sub = gr.Textbox(label="Search Query (e.g., best stocks to buy)")
            max_urls_input_sub = gr.Number(label="Max URLs", value=10)
            use_ollama_sub = gr.Checkbox(label="Use Ollama Augmentation", value=False)
            max_comments_sub = gr.Number(label="Max Comments per Thread", value=50)
            collect_btn_sub = gr.Button("Start Collection")
            status_sub = gr.Textbox(label="Status")
            collect_btn_sub.click(lambda n, s, tl, q, mu, u, mc, ts, cs: start_subreddit_collection(n, s, tl, q, mu, u, mc, ts, cs), [name_input_sub, subreddit_input, timelimit_input_sub, query_input_sub, max_urls_input_sub, use_ollama_sub, max_comments_sub, collection_tasks_state, completed_collections_state], [status_sub, collection_tasks_state, completed_collections_state])
        
        with gr.Tab("File Ingestion"):
            name_input_file = gr.Textbox(label="Data Source Name (optional)")
            file_upload = gr.File(label="Upload TXT or PDF file", file_types=['.txt', '.pdf'], type="filepath")
            use_ollama_file = gr.Checkbox(label="Use Ollama Augmentation", value=False)
            ingest_btn = gr.Button("Start Ingestion")
            status_file = gr.Textbox(label="Status")
            ingest_btn.click(lambda n, fp, u, ts, cs: start_file_ingestion(n, fp, u, ts, cs), [name_input_file, file_upload, use_ollama_file, collection_tasks_state, completed_collections_state], [status_file, collection_tasks_state, completed_collections_state])
        
        with gr.Tab("Tasks"):
            refresh_btn = gr.Button("Refresh Tasks")
            tasks_df = gr.Dataframe(label="Tasks")
            with gr.Accordion("Task Detail", open=False):
                task_id_input = gr.Number(label="Task ID")
                view_detail_btn = gr.Button("View Detail")
                detail_content = gr.Markdown(label="Scraped Content")
                detail_summary = gr.Markdown(label="LLM Summarization")
                detail_answer = gr.Markdown(label="Answer to Search Query")
            refresh_btn.click(refresh_tasks, collection_tasks_state, tasks_df)
            view_detail_btn.click(lambda tid, ts: show_task_detail(tid, ts, conn), [task_id_input, collection_tasks_state], [detail_content, detail_summary, detail_answer])
        
        with gr.Tab("View Database"):
            sources_df = gr.Dataframe(label="Available Data Sources", interactive=True)
            load_sources_btn = gr.Button("Load Data Sources")
            selected_source = gr.Textbox(label="Selected Data Source", interactive=False)
            source_contents_df = gr.Dataframe(label="Data Source Contents")
            new_name_input = gr.Textbox(label="New Name for Selected Source")
            rename_btn = gr.Button("Rename Selected Source")
            rename_status = gr.Textbox(label="Rename Status")
            delete_btn = gr.Button("Delete Selected Source")
            delete_confirm = gr.Checkbox(label="Confirm Deletion")
            delete_status = gr.Textbox(label="Delete Status")
            load_sources_btn.click(load_data_sources, outputs=sources_df)
            sources_df.change(select_data_source, [sources_df, completed_collections_state], [selected_source, source_contents_df])
            rename_btn.click(lambda idx, new_name, cs: rename_data_source(idx, new_name, cs.value), [sources_df, new_name_input, completed_collections_state], rename_status)
            delete_btn.click(lambda idx, confirm, cs: confirm_delete_data_source(idx, confirm, cs.value) if confirm else "Please confirm deletion.", [sources_df, delete_confirm, completed_collections_state], delete_status)
        
        with gr.Tab("Admin"):
            # Moved advanced features here
            load_db_btn = gr.Button("Load Database Contents")
            urls_df = gr.Dataframe(label="URLs Table")
            chunks_df = gr.Dataframe(label="Chunks Table")
            load_db_btn.click(lambda: view_db(conn), outputs=[urls_df, chunks_df])
            sql_query_input = gr.Textbox(label="Custom SQL Query (e.g., SELECT * FROM urls LIMIT 5)")
            execute_query_btn = gr.Button("Execute Query")
            query_output = gr.Markdown(label="Query Results")
            query_error = gr.Textbox(label="Error")
            execute_query_btn.click(lambda q: execute_sql_query(conn, q), sql_query_input, [query_output, query_error])
            show_vs_btn = gr.Button("Show Vector Store Contents")
            vs_output = gr.Markdown(label="Vector Store Entries")
            show_vs_btn.click(view_vectorstore, outputs=vs_output)
            similarity_query_input = gr.Textbox(label="Similarity Search Query")
            similarity_search_btn = gr.Button("Perform Similarity Search")
            similarity_results = gr.Markdown(label="Similarity Search Results")
            similarity_search_btn.click(perform_similarity_search, similarity_query_input, similarity_results)

demo.queue(default_concurrency_limit=5).launch()