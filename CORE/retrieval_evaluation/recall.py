import os
import pandas as pd
import matplotlib.pyplot as plt

def calculate_recall_at_n(true_ids_str, pred_ids, n):
    """
    Calculate the proportion of true_ids that are among the top n predicted ids.
    Handles multiple true IDs if they are comma-separated.
    """
    if pd.isna(true_ids_str):
        return 0
    
    true_ids = [str(t).strip() for t in str(true_ids_str).split(',') if str(t).strip()]
    if not true_ids:
        return 0
        
    top_n_preds = pred_ids[:n]
    hits = sum(1 for true_id in true_ids if true_id in top_n_preds)
    return hits / len(true_ids)

def extract_year_group(filename):
    if filename.startswith('df_2015_2024'): return 'df_2015_2024'
    if filename.startswith('df_2020_2024'): return 'df_2020_2024'
    if filename.startswith('df_2023_2024'): return 'df_2023_2024'
    if filename.startswith('df_2024'): return 'df_2024'
    return 'Other'

def evaluate_models(pred_dir="pred"):
    """
    Reads predictions from subdirectories in pred_dir and calculates Recall@N.
    """
    results = []
    
    for model_name in os.listdir(pred_dir):
        if model_name in ("__pycache__", "result"):
            continue
        model_path = os.path.join(pred_dir, model_name)
        if not os.path.isdir(model_path):
            continue
            
        print(f"Evaluating model predictions: {model_name}")
        
        for filename in os.listdir(model_path):
            if not filename.endswith(".csv"):
                continue
                
            csv_path = os.path.join(model_path, filename)
            df = pd.read_csv(csv_path)
            
            if 'Source Article Index' not in df.columns or 'documentIDs' not in df.columns:
                print(f"Skipping {filename}: Missing required columns.")
                continue
            
            total_questions = len(df)
            if total_questions == 0:
                continue
                
            recall_at_n = 0
            
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
                    
                recall_at_n += calculate_recall_at_n(true_id, pred_ids, 5)
                
            results.append({
                'Model': model_name,
                'Year Group': extract_year_group(filename),
                'Dataset': filename,
                'Total Questions': total_questions,
                'Recall@N': recall_at_n / total_questions,
            })
            
    if not results:
        print("No valid evaluation data found.")
        return
        
    results_df = pd.DataFrame(results)
    
    avg_output = "--- Average Recall across all datasets ---\n"
    agg_df = results_df.groupby('Model')[['Recall@N']].mean().reset_index()
    avg_output += agg_df.to_string(index=False)
    
    avg_output += "\n\n--- Average Recall Sub-grouped by Years ---\n"
    year_agg_df = results_df.groupby(['Model', 'Year Group'])[['Recall@N']].mean().reset_index()
    avg_output += year_agg_df.to_string(index=False)
    
    detail_output = "\n\n--- Detailed Results ---\n"
    detail_output += results_df.to_string(index=False)
    
    print(avg_output)
    print(detail_output)
    
    # Ensure result directory exists
    result_dir = r"C:/Users/jjkwiee/Documents/myPython/Master/FYP/CORE/retrieval_evaluation/result"
    os.makedirs(result_dir, exist_ok=True)
    
    models_str = "_".join(results_df['Model'].unique())
    txt_path = os.path.join(result_dir, f"{models_str}_recall_metrics.txt")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(avg_output + detail_output + "\n")
        
    plt.figure(figsize=(10, 6))
    models = agg_df['Model'].tolist()
    recalls = agg_df['Recall@N'].tolist()
    plt.bar(models, recalls, color='skyblue', edgecolor='black')
    plt.title('Average Recall@N across all datasets by Model')
    plt.ylabel('Recall@N')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    
    for i, v in enumerate(recalls):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
    plt.tight_layout()
    chart1_path = os.path.join(result_dir, f"{models_str}_avg_recall.png")
    plt.savefig(chart1_path)
    plt.close()

    pivot_df = year_agg_df.pivot(index='Year Group', columns='Model', values='Recall@N')
    ax = pivot_df.plot(kind='bar', figsize=(12, 6), rot=45, edgecolor='black')
    plt.title('Recall@N by Year Group and Model')
    plt.ylabel('Recall@N')
    plt.ylim(0, 1.2)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontweight='bold', fontsize=8, rotation=90, padding=3)
        
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    chart2_path = os.path.join(result_dir, f"{models_str}_year_recall.png")
    plt.savefig(chart2_path)
    plt.close()
        
    print(f"\nSaved summary and detailed metrics to {txt_path}")
    print(f"Saved Average Model Chart to: {chart1_path}")
    print(f"Saved Grouped Year Chart to: {chart2_path}")

if __name__ == "__main__":
    base_pred_dir = r"C:/Users/jjkwiee/Documents/myPython/Master/FYP/CORE/retrieval_evaluation"
    evaluate_models(base_pred_dir)
