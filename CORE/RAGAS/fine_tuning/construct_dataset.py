import os
import json
import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# Configurable paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_PATH = os.path.join(CURRENT_DIR, "filtered_real_dataset.json")
SYNTHETIC_PATH = os.path.join(CURRENT_DIR, "synthetic_dataset.json")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "final_training_dataset.json")

# SFT System Prompt
SYSTEM_PROMPT = "You are an chatbot assistant specialized in analyzing research papers from the arXiv dataset."

# SFT User Prompt Template matching core.py
USER_TEMPLATE = """Your task is to generate a direct and short answer to the user's question by synthesizing
the key insights from the retrieved documents provided below.

Guidelines:
1. Base your answer only on the retrieved documents; do not hallucinate or add unsupported claims.
2. Summarize the relevant findings, arguments, and conclusions from the documents.
3. If documents contain conflicting views, present both perspectives clearly.
4. You are not necessary to use all the references given.
5. Write in an academic and objective tone, keep it concise and direct (in 2-3 sentences).
6. If the documents do not contain the information needed to answer the question, respond with "No Answer" only.

Retrieved Documents:
{context}

User Question:
{question}
"""

def get_entailment_index(model, model_name):
    """Finds the index of the entailment label."""
    label2id = getattr(model.config, 'label2id', None)
    if label2id:
        for k, v in label2id.items():
            if 'entail' in k.lower():
                return v
    if 'deberta' in model_name.lower():
        return 1
    return 2

def validate_synthetic_data(samples, model_path="cross-encoder/nli-deberta-v3-base", batch_size=16, threshold=0.5):
    """Validates if generated answers are entailed by the context using a local NLI model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading NLI validation model '{model_path}' on {device}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        model.eval()
        entail_idx = get_entailment_index(model, model_path)
    except Exception as e:
        print(f"Error loading NLI model: {e}. Skipping NLI validation.")
        return samples

    print("Running NLI validation on synthetic samples...")
    valid_samples = []
    
    with torch.no_grad():
        for b_idx in tqdm(range(0, len(samples), batch_size), desc="NLI Validation"):
            batch = samples[b_idx:b_idx+batch_size]
            premises = [s["context"] for s in batch]
            hypotheses = [s["response"] for s in batch]
            
            inputs = tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            scores = probs[:, entail_idx].cpu().numpy().tolist()
            
            for i, score in enumerate(scores):
                # Keep sample if the entailment probability exceeds the threshold
                if score >= threshold:
                    batch[i]["metadata"]["nli_validation_score"] = float(score)
                    valid_samples.append(batch[i])
                    
    print(f"NLI validation filtered out {len(samples) - len(valid_samples)} samples. {len(valid_samples)} passed.")
    return valid_samples

def format_to_chatml(question, context, response):
    """Formats sample into SFT-compatible ChatML messages list."""
    user_prompt = USER_TEMPLATE.format(context=context, question=question)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response}
        ]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-nli", action="store_true", help="Run local NLI validation on synthetic data")
    parser.add_argument("--nli-threshold", type=float, default=0.5, help="NLI entailment probability threshold")
    parser.add_argument("--nli-model", type=str, default="cross-encoder/nli-deberta-v3-base", help="Hugging Face NLI model name")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 3: Dataset Construction & Formatting")
    print("=" * 60)

    # 1. Load datasets
    real_samples = []
    synthetic_samples = []

    if os.path.exists(REAL_PATH):
        with open(REAL_PATH, "r", encoding="utf-8") as f:
            real_samples = json.load(f)
        print(f"Loaded {len(real_samples)} real samples.")
    else:
        print(f"Warning: Real dataset not found at {REAL_PATH}.")

    if os.path.exists(SYNTHETIC_PATH):
        with open(SYNTHETIC_PATH, "r", encoding="utf-8") as f:
            synthetic_samples = json.load(f)
        print(f"Loaded {len(synthetic_samples)} synthetic samples.")
    else:
        print(f"Warning: Synthetic dataset not found at {SYNTHETIC_PATH}.")

    # 2. Run NLI validation if requested
    if args.validate_nli and len(synthetic_samples) > 0:
        synthetic_samples = validate_synthetic_data(
            synthetic_samples, 
            model_path=args.nli_model, 
            threshold=args.nli_threshold
        )

    # 3. Combine and Deduplicate by question text
    combined_raw = []
    seen_questions = set()

    # Add real samples first (to prioritize original data over synthetic)
    for s in real_samples:
        q_norm = s["question"].strip().lower()
        if q_norm not in seen_questions:
            seen_questions.add(q_norm)
            combined_raw.append(s)

    # Add synthetic samples next
    for s in synthetic_samples:
        q_norm = s["question"].strip().lower()
        if q_norm not in seen_questions:
            seen_questions.add(q_norm)
            combined_raw.append(s)

    print(f"\nDeduplicated unique dataset size: {len(combined_raw)} samples.")

    # 4. Format into SFT ChatML format
    final_dataset = []
    for s in combined_raw:
        chat_format = format_to_chatml(s["question"], s["context"], s["response"])
        final_dataset.append(chat_format)

    # 5. Save final training set
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=4, ensure_ascii=False)

    print(f"\n[Success] Dataset compilation complete.")
    print(f"Saved {len(final_dataset)} training samples to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
