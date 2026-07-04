import os
import glob
from core import ArxivRAGSystem

def build_all_databases():
    dataset_dir = r"C:/Users/jjkwiee/Documents/myPython/Master/FYP/dataset"

    embedding_model = "qwen3-embedding:0.6b"  
    
    print("Initializing Arxiv RAG System...")
    rag = ArxivRAGSystem()

    search_pattern = os.path.join(dataset_dir, "*.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"No CSV files found in {dataset_dir}")
        return

    print(f"Found {len(csv_files)} CSV files. Starting bulk database creation...")
    
    for idx, file_path in enumerate(csv_files, start=1):
        file_name = os.path.basename(file_path)
        db_name = os.path.splitext(file_name)[0]
        
        print("\n" + "="*50)
        print(f"[{idx}/{len(csv_files)}] Processing file: '{file_name}'")
        print(f"Target DB Name: '{db_name}'")
        print("="*50)
        
        try:
            rag.create_new_db(
                csv_file_path=file_path, 
                db_name=db_name, 
                embeddings_model_name=embedding_model
            )
            print(f"✓ '{db_name}' successfully built!")
        except Exception as e:
            print(f"❌ Failed to process '{file_name}': {e}")
            
    print("\nBulk database creation process finished!")

if __name__ == "__main__":
    build_all_databases()
