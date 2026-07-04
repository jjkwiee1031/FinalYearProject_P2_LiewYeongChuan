import os
import json
import pandas as pd
import numpy as np

# Configurable paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NLI_RESULTS_PATH = os.path.join(BASE_DIR, "CORE", "generator_evaluation", "Entailment Score", "result", "nli_raw_results.csv")
RAGAS_QS_DIR = os.path.join(BASE_DIR, "CORE", "RAGAS", "questionset")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(CURRENT_DIR, "filtered_real_dataset.json")

def parse_domain(dataset_name):
    """Extracts the domain from the dataset filename."""
    name = dataset_name.replace(".csv", "")
    # Remove prefix like df_2015_2024_
    for prefix in ["df_2015_2024_", "df_2020_2024_", "df_2023_2024_", "df_2024_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name

def main():
    print("=" * 60)
    print("PHASE 1: Dataset Filtering and Stratified Sampling")
    print("=" * 60)

    # 1. Load NLI results
    if not os.path.exists(NLI_RESULTS_PATH):
        raise FileNotFoundError(f"NLI results not found at {NLI_RESULTS_PATH}. Please run the NLI evaluation first.")
    
    print(f"Loading NLI raw results from: {NLI_RESULTS_PATH}")
    df_nli = pd.read_csv(NLI_RESULTS_PATH)
    print(f"Total NLI evaluations: {len(df_nli)}")

    # 2. Filter for entailment majority label
    df_ent = df_nli[df_nli["majority_label"] == "entailment"].copy()
    print(f"Filtered to entailed responses: {len(df_ent)}")

    # Calculate average NLI score as a proxy for entailment confidence
    df_ent["avg_nli_score"] = df_ent[["bart_score", "roberta_score", "deberta_score"]].mean(axis=1)

    # 3. Read source datasets and extract actual question, context, and responses
    candidates = []
    missing_files = set()
    cache_dfs = {}

    for idx, row in df_ent.iterrows():
        ds_name = row["Dataset"]
        row_idx = int(row["Row_Index"])
        llm_model = row["LLM_Model"]
        avg_score = row["avg_nli_score"]

        csv_path = os.path.join(RAGAS_QS_DIR, ds_name)
        if not os.path.exists(csv_path):
            missing_files.add(ds_name)
            continue

        if csv_path not in cache_dfs:
            cache_dfs[csv_path] = pd.read_csv(csv_path)

        src_df = cache_dfs[csv_path]
        
        if row_idx >= len(src_df):
            print(f"Warning: Row index {row_idx} out of bounds for {ds_name}")
            continue

        src_row = src_df.iloc[row_idx]
        question = src_row.get("Question")
        context = src_row.get("Summary")
        response = src_row.get(llm_model)
        q_type = src_row.get("Question Type", "Unknown")

        # Skip if missing key data
        if pd.isna(question) or pd.isna(context) or pd.isna(response):
            continue

        # Basic text cleaning
        question = str(question).strip()
        context = str(context).strip()
        response = str(response).strip()
        q_type = str(q_type).strip()
        domain = parse_domain(ds_name)

        candidates.append({
            "question": question,
            "context": context,
            "response": response,
            "question_type": q_type,
            "domain": domain,
            "source_dataset": ds_name,
            "source_row_index": row_idx,
            "llm_model": llm_model,
            "avg_nli_score": avg_score
        })

    if missing_files:
        print(f"Warning: Could not find {len(missing_files)} source files in {RAGAS_QS_DIR}: {missing_files}")

    print(f"Successfully compiled {len(candidates)} candidates with full text data.")

    # Deduplicate: if the same question has multiple entailed responses, keep the highest scoring one
    df_cand = pd.DataFrame(candidates)
    df_cand = df_cand.sort_values(by="avg_nli_score", ascending=False)
    # Deduplicate by question (keeps the first/highest scoring one)
    df_cand_dedup = df_cand.drop_duplicates(subset=["question"]).copy()
    print(f"Unique questions with entailed responses: {len(df_cand_dedup)}")

    # 4. Stratified Sampling for ~200 diverse samples
    target_size = min(200, len(df_cand_dedup))
    
    # Define stratum based on (domain, question_type)
    df_cand_dedup["stratum"] = df_cand_dedup["domain"] + " | " + df_cand_dedup["question_type"]
    
    # Calculate stratum proportions
    stratum_counts = df_cand_dedup["stratum"].value_counts()
    stratum_proportions = stratum_counts / len(df_cand_dedup)
    
    # Allocate samples to each stratum
    allocated_sizes = (stratum_proportions * target_size).round().astype(int)
    
    # Ensure we don't allocate more than available in any stratum
    allocated_sizes = np.minimum(allocated_sizes, stratum_counts)
    
    selected_indices = []
    
    # Sample from each stratum
    for stratum, size in allocated_sizes.items():
        if size == 0:
            continue
        stratum_data = df_cand_dedup[df_cand_dedup["stratum"] == stratum]
        # Since it is sorted by avg_nli_score descending, taking the head(size) gives the highest quality ones
        selected_indices.extend(stratum_data.head(size).index.tolist())
        
    # If we are slightly under target due to rounding, fill with highest scoring remaining
    remaining_pool = df_cand_dedup.drop(selected_indices)
    fill_needed = target_size - len(selected_indices)
    if fill_needed > 0 and len(remaining_pool) > 0:
        selected_indices.extend(remaining_pool.head(fill_needed).index.tolist())
        
    selected_df = df_cand_dedup.loc[selected_indices]
    
    # 5. Format and Save output
    output_samples = []
    for _, row in selected_df.iterrows():
        output_samples.append({
            "question": row["question"],
            "context": row["context"],
            "response": row["response"],
            "metadata": {
                "question_type": row["question_type"],
                "domain": row["domain"],
                "source_dataset": row["source_dataset"],
                "source_row_index": row["source_row_index"],
                "llm_model": row["llm_model"],
                "avg_nli_score": float(row["avg_nli_score"])
            }
        })
        
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_samples, f, indent=4, ensure_ascii=False)
        
    print(f"\n[Success] Selected {len(output_samples)} highly diverse, entailed samples.")
    print(f"Output saved to: {OUTPUT_PATH}")
    
    # Print some stats
    print("\nSelected Samples Domain Distribution:")
    for dom, count in selected_df["domain"].value_counts().items():
        print(f"  - {dom}: {count}")
        
    print("\nSelected Samples Question Type Distribution:")
    for qt, count in selected_df["question_type"].value_counts().items():
        print(f"  - {qt}: {count}")

if __name__ == "__main__":
    main()
