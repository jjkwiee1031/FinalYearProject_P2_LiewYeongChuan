# -*- coding: utf-8 -*-
"""
Redesigned Flask application for optimized local RAG-based Q&A system.
"""

import os
import sys
import pandas as pd
import csv
import time
from flask import Flask, render_template, request, jsonify

# Add CORE sibling directory to path to import ArxivRAGSystem
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CORE.core import ArxivRAGSystem

# Initialize paths
current_dir = os.path.dirname(os.path.abspath(__file__))
dir_path = "C:/Users/jjkwiee/Documents/myPython/Master/FYP/CORE/db/"
db_list_path = os.path.join(current_dir, "db_list.csv")

# Create Flask app
app = Flask(__name__)

# Initialize local RAG system
rag = ArxivRAGSystem(dir_path=dir_path, db_list_file=db_list_path)

# Check if "demo" database exists in db_list
has_demo = False
if os.path.exists(db_list_path) and os.path.getsize(db_list_path) >= 30:
    try:
        df = pd.read_csv(db_list_path)
        if "db" in df.columns and "demo" in df["db"].values:
            has_demo = True
    except Exception:
        pass

if not has_demo:
    print("\n--- [Startup Setup] ---")
    print("Database 'demo' not registered. Automatically indexing demo dataset 'demo.csv'...")
    demo_csv_path = "C:/Users/jjkwiee/Documents/myPython/Master/FYP/dataset/old/demo.csv"
    if os.path.exists(demo_csv_path):
        try:
            rag.create_new_db(demo_csv_path, "demo", "granite-embedding:30m")
            
            # Save generator field in db_list for demo DB
            df = pd.read_csv(db_list_path)
            if "generator" not in df.columns:
                df["generator"] = "qwen2.5:3b"
            df.loc[df.index[-1], "generator"] = "qwen2.5-3b-ragas"
            df.to_csv(db_list_path, index=False)
            print("Demo dataset successfully indexed with granite-embedding:30m!")
        except Exception as e:
            print(f"Error indexing demo dataset: {e}")
    else:
        print(f"Demo CSV not found at: {demo_csv_path}")
    print("------------------------\n")


@app.route("/")
def index():  
    return render_template("index.html")


@app.route("/data")
def data():
    try:
        if not os.path.exists(db_list_path):
            return jsonify([])
        df = pd.read_csv(db_list_path, dtype={"embeddings": str, "db": str})
        if "generator" not in df.columns:
            df["generator"] = "qwen2.5:3b"
        df["generator"] = df["generator"].fillna("qwen2.5:3b")
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        print(f"Error reading database list: {e}")
        return jsonify([])


@app.route("/switch", methods=["POST"])
def switch(): 
    try:
        db_name = request.json["db"]
        embeddings_model = request.json["embeddings"]
        k_value = int(request.json.get("k_value", 10))
        n_value = int(request.json.get("n_value", 6))
        generator_model = request.json.get("generator", "qwen2.5:3b")

        print(f"\n[Switch Request]")
        print(f"  Database: {db_name}")
        print(f"  Embeddings: {embeddings_model}")
        print(f"  Dense K: {k_value}")
        print(f"  Rerank N: {n_value}")
        print(f"  Generator: {generator_model}")

        # Switch the active database and configure the hybrid retriever pipeline
        rag.switch_db(db_name, embeddings_model, k_value=k_value, n_value=n_value)
        
        # Switch the dynamic generator model
        rag.switch_generator(generator_model)

        return jsonify({"ok": "Switch successful"})
    except Exception as e:
        print(f"Error switching RAG configuration: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/new", methods=["POST"])
def new():
    try:
        form_db = request.form.get("db")
        form_embeddings = request.form.get("embeddings")
        form_generator = request.form.get("generator", "qwen2.5:3b")
        csv_file = request.files["csv_file"]

        # Create temporary file to feed RAG system loader
        temp_dir = os.path.join(current_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, csv_file.filename)
        csv_file.save(temp_file_path)

        print(f"\n[Indexing Request]")
        print(f"  New Database Name: {form_db}")
        print(f"  Embeddings: {form_embeddings}")
        print(f"  Generator: {form_generator}")

        # Process indexing using optimized core loader
        rag.create_new_db(temp_file_path, form_db, form_embeddings)
        
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        # Update metadata CSV to save the generator mapping
        df = pd.read_csv(db_list_path)
        if "generator" not in df.columns:
            df["generator"] = "qwen2.5:3b"
        df.loc[df.index[-1], "generator"] = form_generator
        df.to_csv(db_list_path, index=False)

        return jsonify({"ok": "Database indexed successfully"})
    except Exception as e:
        print(f"Error creating new database: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    if not request.json or "question" not in request.json:
        return jsonify({"error": "Invalid request format"}), 400

    question = request.json["question"]
    
    try:
        start_time = time.time()
        # Execute question processing through optimized pipeline
        answer, doc_ids, doc_contents = rag.ask(question, return_doc_texts=True)
        elapsed_time = time.time() - start_time

        if answer is None:
            return jsonify({"answer": "Error: RAG pipeline failed to respond."})

        # Map document IDs and contents to display references in the accordion
        sources = []
        for i, (d_id, content) in enumerate(zip(doc_ids, doc_contents)):
            title = "Reference Document"
            # Clean title line extraction
            if content.startswith("Title:"):
                lines = content.split("\n")
                title_line = lines[0].replace("Title:", "").strip()
                if title_line.endswith("."):
                    title_line = title_line[:-1]
                title = title_line
            
            sources.append({
                "id": d_id,
                "title": title,
                "content": content
            })

        return jsonify({
            "answer": answer,
            "sources": sources,
            "time_taken": f"{elapsed_time:.2f}s"
        })

    except Exception as e:
        print(f"An error occurred while answering the question: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)