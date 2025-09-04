# chat_utils.py
import os
from config import MODEL_NAME, FAISS_PATH
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from vectorstore_manager import get_vectorstore
import time
import spacy

nlp = spacy.load("en_core_web_sm")

def chat_bot(message, history, conn=None, selected_source=None, selected_tag=None):
    print(f"Starting chat_bot with message: {message}, selected_source: {selected_source}, selected_tag: {selected_tag}")
    history.append({"role": "user", "content": message})
    yield history, ""

    response = ""
    history.append({"role": "assistant", "content": response})
    yield history, ""

    # NLP processing for intent and NER
    doc = nlp(message)
    entities = [ent.text for ent in doc.ents]
    print(f"Debug: Extracted entities: {entities}")

    chat_history = []
    for h in history[:-1]:
        if h["role"] == "user":
            chat_history.append(HumanMessage(content=h["content"]))
        elif h["role"] == "assistant":
            chat_history.append(AIMessage(content=h["content"]))

    llm = OllamaLLM(model=MODEL_NAME)

    retriever = None
    if selected_source != "No RAG":
        response += "**Processing Status:**\n"
        vs = get_vectorstore(selected_tag)
        print(f"Debug: Vector store for tag {selected_tag} loaded with ntotal: {vs.index.ntotal}")
        if vs.index.ntotal == 0:
            response += "No relevant content in vectorstore.\n\n**Specific Answer:**\nSorry, I couldn't find any information."
            history[-1]["content"] = response
            yield history, ""
            return
        search_kwargs = {"k": 5, "filter": {"tag": selected_tag}}
        if 'lyrics' in message.lower():
            search_kwargs["filter"]["source_type"] = "lyrics"
        dense_retriever = vs.as_retriever(search_kwargs=search_kwargs)
        bm_docs = vs.similarity_search(" ", k=min(vs.index.ntotal, 100))  # Limit to avoid large empty stores
        bm25_retriever = BM25Retriever.from_documents(bm_docs)
        retriever = EnsembleRetriever(retrievers=[dense_retriever, bm25_retriever], weights=[0.7, 0.3])
        print(f"Debug: Created ensemble retriever for tag {selected_tag}. BM25 docs loaded: {len(bm_docs)}")

    if retriever is None:
        qa_prompt = ChatPromptTemplate.from_template(
            "Answer the question:\n\nQuestion: {input}"
        )
        qa_chain = qa_prompt | llm
        answer = qa_chain.invoke({"input": message})
        response = f"**Specific Answer:**\n{answer}\n\n"
        history[-1]["content"] = response
        yield history, ""
    else:
        response += "Vector store ready.\n\n"
        history[-1]["content"] = response
        yield history, ""

        response += "Generating summarization of the found content...\n"
        history[-1]["content"] = response
        yield history, ""

        summary_prompt = ChatPromptTemplate.from_template(
            "Summarize the following retrieved content related to the question '{input}' in a concise manner:\n\n{context}"
        )
        summary_chain = create_stuff_documents_chain(llm, summary_prompt)
        summary_chain_with_docs = create_retrieval_chain(retriever, summary_chain)
        summary_response = summary_chain_with_docs.invoke({"input": message})
        summary_docs = summary_response["context"]  # Log retrieved docs
        print("Retrieved docs for summary:", [doc.metadata for doc in summary_docs])
        summary = summary_response["answer"]

        response += f"**Summarization of Found Content:**\n{summary}\n\n"
        history[-1]["content"] = response
        yield history, ""

        response += "Generating specific answer to the prompt...\n"
        history[-1]["content"] = response
        yield history, ""

        rephrase_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Focus on the current question and ignore unrelated history."),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, rephrase_prompt)

        qa_prompt = ChatPromptTemplate.from_template(
            "Use the context to answer the question as accurately as possible. If lyrics are present, extract and format them clearly with verses, chorus, etc., and ignore non-lyric content like discussions. If uncertain or incomplete, note limitations but provide what's available:\n\n{context}\n\nQuestion: {input}"
        )
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        qa_chain_with_docs = create_retrieval_chain(history_aware_retriever, qa_chain)
        qa_response = qa_chain_with_docs.invoke({"input": message, "chat_history": chat_history})
        qa_docs = qa_response["context"]  # Log retrieved docs
        print("Retrieved docs for QA:", [doc.metadata for doc in qa_docs])
        answer = qa_response["answer"]

        response += f"**Specific Answer:**\n{answer}\n\n"
        history[-1]["content"] = response
        yield history, ""

        # Collect unique sources from retrieved documents
        all_docs = set()
        for doc in summary_docs + qa_docs:
            if 'source' in doc.metadata:
                all_docs.add(doc.metadata['source'])

        if all_docs:
            response += "**Referenced Sources:**\n"
            for source in all_docs:
                response += f"- {source}\n"
            history[-1]["content"] = response
            yield history, ""

    print("chat_bot completed.")