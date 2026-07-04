import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from collections import Counter


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
RESULT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, 'result'))

NLI_MODELS = {
    "bart": "facebook/bart-large-mnli",
    "deberta_large": "cross-encoder/nli-deberta-v3-large",
    "deberta_ml": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
}

EXCLUDE_COLUMNS = {
    'Question', 'Question Type', 'Source Article Index', 'Summary', 
    'Unnamed: 3', 'Unnamed: 4', 'documentIDs'
}

def split_sentences(text):
    """Split text into sentences using nltk.sent_tokenize if possible, otherwise regex."""
    if not text or not isinstance(text, str):
        return []
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        sents = nltk.sent_tokenize(text)
        if sents:
            return sents
    except Exception:
        pass
    import re
    # Fallback basic sentence splitter
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def extract_year_group(filename):
    """Extract year group from filename for categorization."""
    if filename.startswith('df_2015_2024'): return 'df_2015_2024'
    if filename.startswith('df_2020_2024'): return 'df_2020_2024'
    if filename.startswith('df_2023_2024'): return 'df_2023_2024'
    if filename.startswith('df_2024'): return 'df_2024'
    return 'Other'

def get_entailment_index(model, model_name):
    """Dynamically find the index of the entailment label."""
    label2id = getattr(model.config, 'label2id', None)
    if label2id:
        for k, v in label2id.items():
            if 'entail' in k.lower():
                return v
    # Fallbacks based on known mappings
    if 'deberta' in model_name.lower():
        return 1
    return 2

def get_standard_label(pred_idx, model_config):
    """Map raw model label ID to standardized lower case label."""
    if not hasattr(model_config, 'id2label') or pred_idx not in model_config.id2label:
        return str(pred_idx)
    raw_label = model_config.id2label[pred_idx]
    raw_label_lower = raw_label.lower()
    if 'entail' in raw_label_lower:
        return 'entailment'
    elif 'contradict' in raw_label_lower:
        return 'contradiction'
    elif 'neutr' in raw_label_lower:
        return 'neutral'
    return raw_label_lower

def run_nli_evaluation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    eval_dirs = []
    for entry in os.listdir(EVAL_BASE_DIR):
        full_path = os.path.join(EVAL_BASE_DIR, entry)
        if os.path.isdir(full_path) and entry not in ["__pycache__", "Entailment Score", "RefusalRate", "result"]:
            eval_dirs.append((entry, full_path))

    if not eval_dirs:
        print("No evaluation subdirectories found under generator_evaluation.")
        return

    print(f"Found embedding directories to evaluate: {[d[0] for d in eval_dirs]}")

    loaded_models = {}
    print("\n--- Loading NLI Models and Tokenizers ---")
    for key, model_path in NLI_MODELS.items():
        print(f"Loading {key} from {model_path}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
            model.eval()
            entail_idx = get_entailment_index(model, model_path)
            loaded_models[key] = {
                "tokenizer": tokenizer,
                "model": model,
                "entail_idx": entail_idx
            }
            print(f"Successfully loaded {key} (Entailment label index: {entail_idx})")
        except Exception as e:
            print(f"Error loading {key} model: {e}")

    if not loaded_models:
        print("Fatal: No NLI models could be loaded. Exiting.")
        return

    all_row_evaluations = []

    for embed_name, embed_path in eval_dirs:
        print(f"\n==================================================")
        print(f"Processing Embedding Model: {embed_name}")
        print(f"==================================================")

        details_dir = os.path.join(embed_path, "nli_details")
        os.makedirs(details_dir, exist_ok=True)

        for filename in os.listdir(embed_path):
            if not filename.endswith(".csv"):
                continue

            csv_path = os.path.join(embed_path, filename)
            df = pd.read_csv(csv_path)

            # Ensure necessary columns are present
            if 'Summary' not in df.columns:
                print(f"Skipping {filename}: Missing 'Summary' reference column.")
                continue

            llm_columns = [col for col in df.columns if col not in EXCLUDE_COLUMNS and not col.startswith('Unnamed:')]
            if not llm_columns:
                print(f"Skipping {filename}: No LLM generator columns found.")
                continue

            print(f"\nEvaluating dataset: {filename}")
            print(f"Found LLM columns: {llm_columns}")

            df_details = df.copy()

            for llm_col in llm_columns:
                sent_pairs = []
                valid_indices = []
                row_sentences = {}

                for idx, row in df.iterrows():
                    q_type = row.get('Question Type')
                    if pd.notna(q_type) and ('negative' in str(q_type).lower() or 'unanswerable' in str(q_type).lower()):
                        continue

                    premise = row['Summary']
                    hypothesis = row[llm_col]
                    
                    if pd.isna(premise) or pd.isna(hypothesis):
                        continue
                        
                    premise_str = str(premise).strip()
                    hypothesis_str = str(hypothesis).strip()
                    
                    if not premise_str or not hypothesis_str:
                        continue
                        
                    sents = split_sentences(hypothesis_str)
                    if not sents:
                        sents = [hypothesis_str]

                    valid_indices.append(idx)
                    row_sentences[idx] = []
                    for sent in sents:
                        row_sentences[idx].append(len(sent_pairs))
                        sent_pairs.append((premise_str, sent))

                if not sent_pairs:
                    print(f"  No valid answers found for model {llm_col} in {filename}")
                    continue

                nli_results_dict = {}

                for nli_key, nli_obj in loaded_models.items():
                    tokenizer = nli_obj["tokenizer"]
                    model = nli_obj["model"]
                    entail_idx = nli_obj["entail_idx"]

                    flat_scores = []
                    flat_labels = []
                    batch_size = 16

                    print(f"  Running NLI Model: {nli_key} on generator: {llm_col} ({len(sent_pairs)} sentence pairs)...")
                    
                    with torch.no_grad():
                        for b_idx in range(0, len(sent_pairs), batch_size):
                            batch = sent_pairs[b_idx:b_idx+batch_size]
                            batch_premises = [p for p, h in batch]
                            batch_hypotheses = [h for p, h in batch]

                            inputs = tokenizer(
                                batch_premises,
                                batch_hypotheses,
                                padding=True,
                                truncation=True,
                                max_length=512,
                                return_tensors="pt"
                            ).to(device)

                            outputs = model(**inputs)
                            logits = outputs.logits
                            probs = torch.softmax(logits, dim=-1)
                            
                            batch_scores = probs[:, entail_idx].cpu().numpy().tolist()
                            flat_scores.extend(batch_scores)

                            batch_pred_ids = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
                            for pred_id in batch_pred_ids:
                                flat_labels.append(get_standard_label(pred_id, model.config))

                    row_scores = []
                    row_labels = []
                    for idx in valid_indices:
                        sent_idxs = row_sentences[idx]
                        row_sent_labels = [flat_labels[j] for j in sent_idxs]
                        
                        entailed_count = sum(1 for l in row_sent_labels if l == 'entailment')
                        row_score = entailed_count / len(row_sent_labels)
                        
                        counts = Counter(row_sent_labels)
                        row_label = counts.most_common(1)[0][0]
                        
                        row_scores.append(row_score)
                        row_labels.append(row_label)

                    nli_results_dict[nli_key] = {
                        "scores": row_scores,
                        "labels": row_labels
                    }

                    df_details.loc[valid_indices, f"{llm_col}_score_{nli_key}"] = row_scores
                    df_details.loc[valid_indices, f"{llm_col}_label_{nli_key}"] = row_labels

                majority_labels = []
                maj_threshold = (len(loaded_models) // 2) + 1
                for i in range(len(valid_indices)):
                    row_labels = [nli_results_dict[k]["labels"][i] for k in loaded_models.keys()]
                    counts = Counter(row_labels)
                    most_common = counts.most_common()
                    if most_common[0][1] >= maj_threshold:
                        maj_label = most_common[0][0]
                    else:
                        maj_label = 'Ambiguous'
                    majority_labels.append(maj_label)

                    row_eval_item = {
                        'Embedding_Model': embed_name,
                        'LLM_Model': llm_col,
                        'Year_Group': extract_year_group(filename),
                        'Dataset': filename,
                        'Row_Index': valid_indices[i],
                        'majority_label': maj_label
                    }
                    for k in loaded_models.keys():
                        row_eval_item[f"{k}_score"] = nli_results_dict[k]["scores"][i]
                        row_eval_item[f"{k}_label"] = nli_results_dict[k]["labels"][i]
                    all_row_evaluations.append(row_eval_item)

                df_details.loc[valid_indices, f"{llm_col}_majority_label"] = majority_labels

            detail_csv_path = os.path.join(details_dir, filename.replace(".csv", "_nli.csv"))
            df_details.to_csv(detail_csv_path, index=False)
            print(f"Saved detailed question-level scores to {detail_csv_path}")

    if not all_row_evaluations:
        print("No evaluation results were computed.")
        return

    eval_df = pd.DataFrame(all_row_evaluations)

    os.makedirs(RESULT_DIR, exist_ok=True)
    raw_results_path = os.path.join(RESULT_DIR, "nli_raw_results.csv")
    eval_df.to_csv(raw_results_path, index=False)
    print(f"Saved raw NLI evaluation results to {raw_results_path}")
    print("\n[Success] NLI Evaluation complete! Run 'nli_graph.py' to generate metrics and plots.")

if __name__ == "__main__":
    run_nli_evaluation()
