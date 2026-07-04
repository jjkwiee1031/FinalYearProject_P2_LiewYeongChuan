import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core import ArxivRAGSystem

QUESTION_SET_DIR = r"C:/Users/jjkwiee/Documents/myPython/Master/FYP/questionSet"
EVAL_BASE_DIR = r"C:/Users/jjkwiee/Documents/myPython/Master/FYP/CORE/generator_evaluation"

EMBEDDING_MODEL = "granite-embedding:30m"
GENERATOR_MODEL = "qwen2.5:3b"

def run_evaluation():
    print("Initializing RAG system...")
    rag = ArxivRAGSystem()
    rag.switch_generator(GENERATOR_MODEL)
    
    eval_output_dir = os.path.join(EVAL_BASE_DIR, EMBEDDING_MODEL.replace(":", "-").replace("/", "-"))
    os.makedirs(eval_output_dir, exist_ok=True)
    
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
                k_value=10
            )
        except Exception as e:
            print(f"Error: Skipped running eval on '{db_name}' because the Chroma Database could not be loaded. Please ensure it was created. Details: {e}")
            continue
        
        eval_out_path = os.path.join(eval_output_dir, filename)
        
        if os.path.exists(eval_out_path):
            df = pd.read_csv(eval_out_path)
        else:
            df = pd.read_csv(csv_path)
        
        if 'Question' not in df.columns:
            print(f"Skipping {filename}: Could not find a column named 'question'. Available columns are: {list(df.columns)}")
            continue
            
        predictions = []
        document_ids = []
        
        for i, (idx, row) in enumerate(df.iterrows(), 1):
            question = row['Question']
            print(f"\n--- Question {i}/{len(df)} ---")
            
            result, retrieved_ids = rag.ask(question, return_docs=True)
            
            predictions.append(result)
            document_ids.append(", ".join(retrieved_ids) if retrieved_ids else "")
            
        if GENERATOR_MODEL != "":
            df[GENERATOR_MODEL] = predictions
        df['documentIDs'] = document_ids
        
        df.to_csv(eval_out_path, index=False)
        print(f"\n[Success] Saved predictions for {filename} to {eval_out_path}")

if __name__ == "__main__":
    run_evaluation()
