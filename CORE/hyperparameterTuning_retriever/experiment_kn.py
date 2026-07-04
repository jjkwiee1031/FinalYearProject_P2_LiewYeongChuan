import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from core import ArxivRAGSystem

QUESTION_SET_DIR = r"C:/Users/jjkwiee/Documents/myPython/Master/FYP/questionSet"
EVAL_BASE_DIR = r"C:/Users/jjkwiee/Documents/myPython/Master/FYP/CORE/hyperparameterTuning_retriever/pred/"

EMBEDDING_MODEL = "granite-embedding:30m"
GENERATOR_MODEL = "llama3.2:latest"

K_VALUES = [5, 10, 15]
N_VALUES = list(range(1, 11, 1))

def run_experiment():
    print("Initializing RAG system for experiments...")
    rag = ArxivRAGSystem()
    rag.switch_generator(GENERATOR_MODEL)
    
    for k in K_VALUES:
        for n in N_VALUES:
            if n > k:
                print(f"Skipping k={k}, n={n} as n cannot be greater than k.")
                continue

            print(f"\n==================================================")
            print(f"Running Experiment with K={k}, N={n}")
            print(f"==================================================")

            safe_model_name = EMBEDDING_MODEL.replace(":", "-").replace("/", "-")
            experiment_name = f"{safe_model_name}_k{k}_n{n}"
            pred_output_dir = os.path.join(EVAL_BASE_DIR, experiment_name)
            os.makedirs(pred_output_dir, exist_ok=True)
            
            for filename in os.listdir(QUESTION_SET_DIR):
                if not filename.endswith(".csv"):
                    continue
                    
                csv_path = os.path.join(QUESTION_SET_DIR, filename)
                db_name = filename.replace(".csv", "")
                
                print(f"\nEvaluating Question Set: {db_name} (K={k}, N={n})")
                
                try:
                    rag.switch_db(
                        db_name=db_name,
                        embeddings_model_name=EMBEDDING_MODEL,
                        k_value=k,
                        n_value=n
                    )
                except Exception as e:
                    print(f"Error: Skipped running eval on '{db_name}'. Details: {e}")
                    continue
                
                df = pd.read_csv(csv_path)
                
                if 'Question' not in df.columns:
                    print(f"Skipping {filename}: Could not find a column named 'Question'.")
                    continue
                    
                predictions = []
                document_ids = []
                
                for idx, row in df.iterrows():
                    question = row['Question']
                    print(f"\n--- Question {idx + 1}/{len(df)} ---")
                    
                    result, retrieved_ids = rag.ask(question, return_docs=True)
                    
                    predictions.append(result)
                    document_ids.append(", ".join(retrieved_ids) if retrieved_ids else "")
                    
                df['prediction'] = predictions
                df['documentIDs'] = document_ids
                
                out_path = os.path.join(pred_output_dir, filename)
                df.to_csv(out_path, index=False)
                print(f"\n[Success] Saved predictions for {filename} to {out_path}")

if __name__ == "__main__":
    run_experiment()
