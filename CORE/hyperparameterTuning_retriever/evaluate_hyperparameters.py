import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def calculate_recall_at_k(true_ids_str, pred_ids, k):
    """
    Calculate the proportion of true_ids that are among the top k predicted ids.
    Handles multiple true IDs if they are comma-separated.
    """
    if pd.isna(true_ids_str):
        return 0
    
    true_ids = [str(t).strip() for t in str(true_ids_str).split(',') if str(t).strip()]
    if not true_ids:
        return 0
        
    top_k_preds = pred_ids[:k]
    hits = sum(1 for true_id in true_ids if true_id in top_k_preds)
    return hits / len(true_ids)

def evaluate_hyperparameters(pred_dir):
    """
    Reads predictions from subdirectories in pred_dir, extracts K and N, 
    and calculates Recall@k for each hyperparameter combination.
    """
    if not os.path.exists(pred_dir):
        print(f"Directory {pred_dir} does not exist yet. Please run the experiment first.")
        return

    results = []
    
    pattern = re.compile(r"(.+)_k(\d+)_n(\d+)$")
    
    for folder_name in os.listdir(pred_dir):
        model_path = os.path.join(pred_dir, folder_name)
        if not os.path.isdir(model_path):
            continue
            
        match = pattern.match(folder_name)
        if not match:
            print(f"Skipping folder {folder_name}: Doesn't match expected pattern (Model_kX_nY).")
            continue
            
        model_name, k_val, n_val = match.groups()
        k_val = int(k_val)
        n_val = int(n_val)
        
        print(f"Evaluating K={k_val}, N={n_val} for model {model_name}...")
        
        total_questions = 0
        recall_at_n = 0
        
        for filename in os.listdir(model_path):
            if not filename.endswith(".csv"):
                continue
                
            csv_path = os.path.join(model_path, filename)
            df = pd.read_csv(csv_path)
            
            if 'Source Article Index' not in df.columns or 'documentIDs' not in df.columns:
                continue
            
            q_count = len(df)
            if q_count == 0:
                continue
                
            total_questions += q_count
            
            for _, row in df.iterrows():
                true_id = row['Source Article Index']
                
                raw_doc_ids = str(row['documentIDs'])
                if raw_doc_ids.strip().lower() == 'nan' or not raw_doc_ids.strip():
                    pred_ids = []
                else:
                    raw_preds = [doc_id.strip() for doc_id in raw_doc_ids.split(',')]
                    pred_ids = []
                    for pid in raw_preds:
                        if pid and pid not in pred_ids:
                            pred_ids.append(pid)
                    
                recall_at_n += calculate_recall_at_k(true_id, pred_ids, n_val)
                
        if total_questions > 0:
            results.append({
                'Model': model_name,
                'K': k_val,
                'N': n_val,
                'Total Questions': total_questions,
                'Recall@N': recall_at_n / total_questions
            })
            
    if not results:
        print("No valid evaluation data found to calculate metrics.")
        return
        
    results_df = pd.DataFrame(results)
    
    results_df = results_df.sort_values(by=['Model', 'K', 'N']).reset_index(drop=True)
    
    output_str = "=== Hyperparameter Tuning Results (Varying K and N) ===\n"
    output_str += results_df.to_string(index=False)
    
    print("\n" + output_str + "\n")
    
    result_dir = r"C:/Users/jjkwiee/Documents/myPython/Master/FYP/CORE/hyperparameterTuning_retriever/result/"
    os.makedirs(result_dir, exist_ok=True)
    
    csv_path = os.path.join(result_dir, "hyperparameter_tuning_metrics.csv")
    txt_path = os.path.join(result_dir, "hyperparameter_tuning_metrics.txt")
    
    results_df.to_csv(csv_path, index=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(output_str + "\n")
        
    plt.figure(figsize=(10, 6))
    
    for k_val in sorted(results_df['K'].unique()):
        subset = results_df[results_df['K'] == k_val]
        subset = subset.sort_values(by='N')
        plt.plot(subset['N'], subset['Recall@N'], marker='o', label=f'K={k_val}')
        
    plt.title('Recall@N vs Reranker N for different Retriever K values')
    plt.xlabel('N (Reranker Top N)')
    plt.ylabel('Recall@N')
    plt.xticks(sorted(results_df['N'].unique()))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Retriever K')
    
    chart_path = os.path.join(result_dir, "hyperparameter_tuning_chart.png")
    plt.savefig(chart_path)
    plt.close()
        
    print(f"Saved metrics summary to {csv_path} and {txt_path}")
    print(f"Saved line chart to {chart_path}")

if __name__ == "__main__":
    BASE_EVAL_DIR = r"C:/Users/jjkwiee/Documents/myPython/Master/FYP/CORE/hyperparameterTuning_retriever/pred/"
    evaluate_hyperparameters(BASE_EVAL_DIR)
