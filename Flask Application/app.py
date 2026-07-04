# -*- coding: utf-8 -*-
"""
Flask application for a RAG-based Q&A system.
"""

import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
import csv
import time

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate



dir_path = "C:/Users/jjkwiee/Documents/myPython/Master/FYP/Flask Application/db/"
db_list = "db_list.csv"

template = """
You are an chatbot assistant specialized in analyzing research papers from the arXiv dataset.
Your task is to generate a direct and short answer to the user's question by synthesizing
the key insights from the retrieved documents provided below.

Guidelines:
1. Base your answer only on the retrieved documents; do not hallucinate or add unsupported claims.
2. Summarize the relevant findings, arguments, and conclusions from the documents.
3. If documents contain conflicting views, present both perspectives clearly.
4. You are not necessary to use all the references given.
5. Write in an academic and objective tone, keep it concise and direct (in 2-3 sentences).

Retrieved Documents:
{references}

User Question:
{question}
"""

model = OllamaLLM(model="llama3.2")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
retriever = None


app = Flask(__name__)

if not os.path.exists(db_list):
    with open(db_list, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["embeddings", "db"])

@app.route("/")
def index():  
    return render_template("index.html")

@app.route("/data")
def data():
    df = pd.read_csv(db_list, dtype={"embeddings": str, "db": str})
    return jsonify(df.to_dict(orient="records"))

@app.route("/switch", methods=["POST"])
def switch(): 
    global retriever
    db_location = dir_path + request.json["db"]
    embeddings = OllamaEmbeddings(model= request.json["embeddings"])

    vector_store = Chroma(
        collection_name="arxiv",
        persist_directory=db_location,
        embedding_function=embeddings
    )       

    retriever = vector_store.as_retriever(search_kwargs={"k": int(request.json["k_value"])})
    print("vector store size:", vector_store._collection.count())
    print("Retriever k value:", retriever.search_kwargs.get("k"))

    return jsonify({"ok":  "Switch successful"})

@app.route("/new", methods = ["POST"])
def new():
    form_db = request.form.get("db")
    form_embeddings = request.form.get("embeddings")

    db_location = dir_path + form_db
    embeddings = OllamaEmbeddings(model= form_embeddings)
    df = pd.read_csv(request.files["csv_file"])

    vector_store = Chroma(
    collection_name="arxiv",
    persist_directory=db_location,
    embedding_function=embeddings
    )

    documents = []
    ids = []

    for i,row in df.iterrows():
        document = Document(
            page_content = f"""Tittle: {row['title']}.
            Published_date: {row['published_date']}.
            Summary: {row['summary'].replace("\n"," ")}""",
            id = str(i)
        )
        documents.append(document)
        ids.append(str(i))
    
    vector_store.add_documents(documents=documents, ids=ids)
    with open(db_list, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([form_embeddings, form_db])

    return jsonify({"ok": "ok"})



@app.route("/ask", methods=["POST"])
def ask():
    if not request.json or "question" not in request.json:
        return jsonify({"error": "Invalid request format"}), 400

    question = request.json["question"]
    print(f"Received question: {question}")

    try:
        start_retrieve = time.time()
        reference_docs = retriever.invoke(question)
        end_retrieve = time.time()
        retrieve_time = end_retrieve - start_retrieve

        start_gen = time.time()
        result = chain.invoke({"references": reference_docs, "question": question})
        end_gen = time.time()
        gen_time = end_gen - start_gen
        print(f"retriever_time {retrieve_time:.2f} s")
        print(f"generation_time {gen_time:.2f} s")

        return jsonify({
            "answer": result
        })

    except Exception as e:
        print(f"An error occurred while processing the question: {e}")
        return jsonify({"error": "An error occurred on the server."}), 500







if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    