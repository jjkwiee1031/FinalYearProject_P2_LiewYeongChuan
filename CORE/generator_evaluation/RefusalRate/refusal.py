import os
import sys
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

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

REFUSAL_PHRASES = [
    "no answer",
    "not provided",
    "not mention",
    "does not mention",
    "do not mention",
    "not explicitly mentioned",
    "does not contain",
    "do not contain",
    "no information",
    "cannot answer",
    "unable to answer",
    "cannot be answered",
    "insufficient information",
    "not discuss",
    "does not discuss",
    "not state",
    "does not state",
    "not address",
    "does not address",
    "no answer is possible",
    "not possible to answer",
]

def get_entailment_index(model, model_name):
    """Dynamically find the index of the entailment label."""
    label2id = getattr(model.config, 'label2id', None)
    if label2id:
        for k, v in label2id.items():
            if 'entail' in k.lower():
                return v
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


def extract_year_group(filename):
    """Extract year group from filename for categorization."""
    if filename.startswith('df_2015_2024'): return 'df_2015_2024'
    if filename.startswith('df_2020_2024'): return 'df_2020_2024'
    if filename.startswith('df_2023_2024'): return 'df_2023_2024'
    if filename.startswith('df_2024'): return 'df_2024'
    return 'Other'

def is_refusal(text):
    """Check if the response contains any common refusal phrases or is empty/NaN."""
    if pd.isna(text):
        return True
    text_str = str(text).strip().lower()
    if not text_str:
        return True
        
    for phrase in REFUSAL_PHRASES:
        if phrase in text_str:
            return True
            
    return False

def run_refusal_evaluation():
    print("="*60)
    print("STARTING REFUSAL RATE EVALUATION")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    loaded_models = {}
    print("\n--- Loading NLI Models and Tokenizers for Refusal Evaluation ---")
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
        print("Warning: No NLI models could be loaded. Refusal calculation will fallback to keyword-only.")
    
    eval_dirs = []
    for entry in os.listdir(EVAL_BASE_DIR):
        full_path = os.path.join(EVAL_BASE_DIR, entry)
        # Exclude subfolders and result
        if os.path.isdir(full_path) and entry not in ["__pycache__", "result", "nli_details", "Entailment Score", "RefusalRate"]:

            eval_dirs.append((entry, full_path))

    if not eval_dirs:
        print("No evaluation subdirectories found under generator_evaluation.")
        return

    print(f"Found embedding directories to evaluate: {[d[0] for d in eval_dirs]}")

    # We will accumulate all unanswerable question-level evaluations here
    all_refusal_evaluations = []

    # Process each embedding directory
    for embed_name, embed_path in eval_dirs:
        print(f"\nProcessing Embedding Model: {embed_name}")
        
        for filename in os.listdir(embed_path):
            if not filename.endswith(".csv"):
                continue

            csv_path = os.path.join(embed_path, filename)
            df = pd.read_csv(csv_path)

            if 'Question Type' not in df.columns or 'Question' not in df.columns:
                print(f"Skipping {filename}: Missing 'Question Type' or 'Question' column.")
                continue

            llm_columns = [col for col in df.columns if col not in EXCLUDE_COLUMNS and not col.startswith('Unnamed:')]
            if not llm_columns:
                print(f"Skipping {filename}: No LLM generator columns found.")
                continue

            unanswerable_mask = df['Question Type'].fillna('').astype(str).str.lower().str.contains('negative|unanswerable')
            unanswerable_df = df[unanswerable_mask]

            if unanswerable_df.empty:
                continue

            print(f"  Evaluating {filename} ({len(unanswerable_df)} unanswerable questions)...")

            for idx, row in unanswerable_df.iterrows():
                question = row['Question']
                q_type = row['Question Type']
                year_group = extract_year_group(filename)

                for llm_col in llm_columns:
                    response = row[llm_col]
                    
                    refused = is_refusal(response)

                    all_refusal_evaluations.append({
                        'Embedding_Model': embed_name,
                        'LLM_Model': llm_col,
                        'Year_Group': year_group,
                        'Dataset': filename,
                        'Question_Set': filename.replace('.csv', ''),
                        'Row_Index': idx,
                        'Question': question,
                        'Question_Type': q_type,
                        'LLM_Response': response if pd.notna(response) else '',
                        'Is_Refusal_Keyword': refused
                    })

    if not all_refusal_evaluations:
        print("No unanswerable questions evaluated.")
        return

    if loaded_models:
        print(f"\n--- Running NLI Refusal Inference on {len(all_refusal_evaluations)} responses ---")
        
        sent_pairs = []
        response_sentences = {}

        for idx, item in enumerate(all_refusal_evaluations):
            resp = str(item['LLM_Response']).strip()
            if not resp:
                sents = ["I cannot answer this question."]
            else:
                sents = split_sentences(resp)
                if not sents:
                    sents = [resp]
            
            response_sentences[idx] = []
            for sent in sents:
                response_sentences[idx].append(len(sent_pairs))
                sent_pairs.append((sent, "The response do not have answer for the question."))

        for item in all_refusal_evaluations:
            for k in loaded_models.keys():
                item[f"{k}_score"] = 0.0
                item[f"{k}_label"] = "neutral"
            item["majority_label"] = "neutral"
            item["Is_Refusal_NLI"] = False

        for nli_key, nli_obj in loaded_models.items():
            tokenizer = nli_obj["tokenizer"]
            model = nli_obj["model"]
            entail_idx = nli_obj["entail_idx"]
            
            flat_scores = []
            flat_labels = []
            batch_size = 16
            
            print(f"Running NLI Model: {nli_key} on {len(sent_pairs)} sentence pairs...")
            with torch.no_grad():
                for b_idx in tqdm(range(0, len(sent_pairs), batch_size)):
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
            
            for idx, item in enumerate(all_refusal_evaluations):
                sent_idxs = response_sentences[idx]
                row_sent_scores = [flat_scores[j] for j in sent_idxs]
                row_sent_labels = [flat_labels[j] for j in sent_idxs]
                
                # If any sentence entails refusal, classify response as refusal (entailment)
                if 'entailment' in row_sent_labels:
                    row_label = 'entailment'
                elif 'contradiction' in row_sent_labels:
                    row_label = 'contradiction' if all(l == 'contradiction' for l in row_sent_labels) else 'neutral'
                else:
                    row_label = 'neutral'
                
                row_score = max(row_sent_scores) if row_sent_scores else 0.0
                
                item[f"{nli_key}_score"] = row_score
                item[f"{nli_key}_label"] = row_label

        maj_threshold = (len(loaded_models) // 2) + 1
        for item in all_refusal_evaluations:
            # If response is empty/nan, it is always a refusal
            if not str(item['LLM_Response']).strip():
                item["majority_label"] = "entailment"
                item["Is_Refusal_NLI"] = True
                continue
                
            row_labels = [item[f"{k}_label"] for k in loaded_models.keys()]
            counts = Counter(row_labels)
            most_common = counts.most_common()
            if most_common[0][1] >= maj_threshold:
                maj_label = most_common[0][0]
            else:
                maj_label = 'Ambiguous'
            
            item["majority_label"] = maj_label
            item["Is_Refusal_NLI"] = (maj_label == "entailment")
    else:
        # Fallback if NLI model is not loaded
        for item in all_refusal_evaluations:
            item["Is_Refusal_NLI"] = item["Is_Refusal_Keyword"]

    eval_df = pd.DataFrame(all_refusal_evaluations)

    os.makedirs(RESULT_DIR, exist_ok=True)
    raw_results_path = os.path.join(RESULT_DIR, "refusal_raw_results.csv")
    eval_df.to_csv(raw_results_path, index=False)
    print(f"\nSaved raw refusal evaluation results to: {raw_results_path}")


    try:
        from refusal_graph import generate_refusal_graphs
        generate_refusal_graphs()
    except ImportError as e:
        print(f"\nCould not run refusal_graph.py automatic graphing: {e}")
        print("Please run 'refusal_graph.py' manually to generate metrics and plots.")

if __name__ == "__main__":
    run_refusal_evaluation()
