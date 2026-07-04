import os
import sys
import json
import pandas as pd
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core import ArxivRAGSystem

QUESTION_SET_DIR = r"C:/Users/jjkwiee/Documents/myPython/Master/FYP/questionSet"
EVAL_OUT_BASE_DIR = r"C:/Users/jjkwiee/Documents/myPython/Master/FYP/CORE/RAGAS/questionset"

EMBEDDING_MODEL = "nomic-embed-text:latest"

def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["before", "after"], default="after", 
                        help="Run mode: 'before' (baseline: qwen2.5:3b, k=10, n=5) or 'after' (optimized: qwen2.5-3b-ragas, k=10, n=6)")
    args_cli = parser.parse_args()

    if args_cli.mode == "before":
        generator_model = "qwen2.5:3b"
        k_val = 10
        n_val = 5
        eval_out_dir = os.path.join(EVAL_OUT_BASE_DIR, "before")
    else:
        generator_model = "qwen2.5-3b-ragas"
        k_val = 10
        n_val = 6
        eval_out_dir = os.path.join(EVAL_OUT_BASE_DIR, "after")

    print("=" * 60)
    print(f"RAGAS PREDICTION RUN: {args_cli.mode.upper()} MODE")
    print(f"Generator Model: {generator_model}")
    print(f"Retriever k: {k_val}")
    print(f"Reranker top_n (top N): {n_val}")
    print(f"Output Directory: {eval_out_dir}")
    print("=" * 60)

    print("Initializing RAG system...")
    rag = ArxivRAGSystem()
    rag.switch_generator(generator_model)
    
    os.makedirs(eval_out_dir, exist_ok=True)
    
    for filename in os.listdir(QUESTION_SET_DIR):
        if not filename.endswith(".csv"):
            continue
            
        csv_path = os.path.join(QUESTION_SET_DIR, filename)
        
        db_name = filename.replace(".csv", "")
        
        print(f"\n" + "="*50)
        print(f"Evaluating Question Set: {db_name}")
        print("="*50)
        
        try:
            rag.switch_db(
                db_name=db_name,
                embeddings_model_name=EMBEDDING_MODEL,
                k_value=k_val,
                n_value=n_val
            )
        except Exception as e:
            print(f"Error: Skipped running eval on '{db_name}' because the Chroma Database could not be loaded. Details: {e}")
            continue
        
        eval_out_path = os.path.join(eval_out_dir, filename)
        
        if os.path.exists(eval_out_path):
            df = pd.read_csv(eval_out_path)
            print(f"Loaded existing output file from: {eval_out_path}")
        else:
            df = pd.read_csv(csv_path)
            print(f"Loaded source file from: {csv_path}")
        
        if 'Question' not in df.columns:
            print(f"Skipping {filename}: Could not find a column named 'Question'. Available columns are: {list(df.columns)}")
            continue
            
        predictions = []
        document_ids = []
        retrieved_contexts_list = []
        
        for i, (idx, row) in enumerate(df.iterrows(), 1):
            question = row['Question']
            print(f"\n--- Question {i}/{len(df)} ---")
            
            result, retrieved_ids, retrieved_contexts = rag.ask(question, return_doc_texts=True)
            
            predictions.append(result)
            document_ids.append(", ".join(retrieved_ids) if retrieved_ids else "")
            
            retrieved_contexts_list.append(json.dumps(retrieved_contexts, ensure_ascii=False) if retrieved_contexts else "[]")
            
        if generator_model != "":
            df[generator_model] = predictions
        df['documentIDs'] = document_ids
        df['context_retrieved'] = retrieved_contexts_list
        
        df.to_csv(eval_out_path, index=False)
        print(f"\n[Success] Saved predictions & contexts for {filename} to {eval_out_path}")

if __name__ == "__main__":
    run_evaluation()
