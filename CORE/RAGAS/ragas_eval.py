import time
import asyncio
import json
import argparse
import glob
import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ragas.metrics._context_precision import IDBasedContextPrecision
from ragas.metrics._context_recall import IDBasedContextRecall
from ragas import SingleTurnSample, EvaluationDataset



NLI_MODELS = {
    "bart": "facebook/bart-large-mnli",
    "deberta_large": "cross-encoder/nli-deberta-v3-large",
    "deberta_ml": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
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



def load_dataset_from_dir(directory, model_column):
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    samples = []
    summaries_list = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            user_input_raw = row.get("Question")
            response_raw = row.get(model_column)
            
            # Skip rows with missing question or response
            if pd.isna(user_input_raw) or pd.isna(response_raw):
                continue
                
            user_input = str(user_input_raw).strip()
            response = str(response_raw).strip()
            
            if not user_input or not response:
                continue
                
            # retrieved_contexts -> context_retrieved (stored as a JSON string of a list of strings)
            retrieved_contexts_raw = row.get("context_retrieved", "[]")
            if pd.isna(retrieved_contexts_raw) or not str(retrieved_contexts_raw).strip():
                retrieved_contexts = []
            else:
                try:
                    retrieved_contexts = json.loads(str(retrieved_contexts_raw))
                    if not isinstance(retrieved_contexts, list):
                        retrieved_contexts = [str(retrieved_contexts)]
                except Exception:
                    retrieved_contexts = [str(retrieved_contexts_raw)]
            
            # retrieved_context_ids -> documentIDs (split by comma and strip)
            retrieved_context_ids_raw = row.get("documentIDs", "")
            if pd.isna(retrieved_context_ids_raw):
                retrieved_context_ids = []
            else:
                retrieved_context_ids = [x.strip() for x in str(retrieved_context_ids_raw).split(",") if x.strip()]
                
            # reference_context_ids -> Source Article Index (split by comma and strip)
            reference_context_ids_raw = row.get("Source Article Index", "")
            if pd.isna(reference_context_ids_raw):
                reference_context_ids = []
            else:
                reference_context_ids = [x.strip() for x in str(reference_context_ids_raw).split(",") if x.strip()]
                
            # Gold source article summary
            summary = str(row.get("Summary", "")).strip()
            
            sample = SingleTurnSample(
                user_input=user_input,
                retrieved_contexts=retrieved_contexts,
                response=response,
                retrieved_context_ids=retrieved_context_ids,
                reference_context_ids=reference_context_ids
            )
            samples.append(sample)
            summaries_list.append(summary)
            
    return EvaluationDataset(samples=samples), summaries_list

async def evaluate_dataset(
    dataset, summaries_list, mode_name, 
    rel_tokenizer, rel_model, 
    nli_tokenizer, nli_model, entail_idx, 
    device, batch_size=16
):
    metric_precision = IDBasedContextPrecision()
    metric_recall = IDBasedContextRecall()

    print(f"Evaluating RAGAS metrics on {mode_name} dataset with {len(dataset.samples)} samples...")
    
    tasks = []
    for sample in dataset.samples:
        async def evaluate_deterministic_metrics(s):
            try:
                prec = await metric_precision.single_turn_ascore(s)
            except Exception:
                try:
                    prec = await metric_precision.ascore(s)
                except Exception:
                    prec = 0.0
            
            try:
                rec = await metric_recall.single_turn_ascore(s)
            except Exception:
                try:
                    rec = await metric_recall.ascore(s)
                except Exception:
                    rec = 0.0
            
            # Extract numerical values
            p_val = prec.value if hasattr(prec, 'value') else prec
            r_val = rec.value if hasattr(rec, 'value') else rec
            
            return p_val if p_val is not None else 0.0, r_val if r_val is not None else 0.0
            
        tasks.append(evaluate_deterministic_metrics(sample))
        
    deterministic_scores = await asyncio.gather(*tasks)
    
    from tqdm import tqdm
    relevancy_scores = []
    
    questions = [s.user_input for s in dataset.samples]
    responses = [s.response for s in dataset.samples]
    
    with torch.no_grad():
        for b_idx in tqdm(range(0, len(dataset.samples), batch_size), desc="  Computing QA Relevancy"):
            batch_questions = questions[b_idx:b_idx+batch_size]
            batch_responses = responses[b_idx:b_idx+batch_size]
            
            # Tokenize pairs together: (Question, Answer)
            encoded = rel_tokenizer(
                batch_questions, 
                batch_responses, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(device)
            
            outputs = rel_model(**encoded)
            logits = outputs.logits
            
            # Apply sigmoid to convert logits to [0, 1] relevance probabilities
            probs = torch.sigmoid(logits).cpu().numpy().flatten().tolist()
            relevancy_scores.extend(probs)
            
    faithfulness_scores = []
    sent_pairs = []
    row_mapping = {}
    
    for idx, sample in enumerate(dataset.samples):
        premise_str = summaries_list[idx]
        
        claims = split_sentences(sample.response)
        if not claims:
            claims = [sample.response]
            
        row_mapping[idx] = []
        for claim in claims:
            row_mapping[idx].append(len(sent_pairs))
            sent_pairs.append((premise_str, claim))
            
    if sent_pairs:
        flat_labels = []
        with torch.no_grad():
            for b_idx in tqdm(range(0, len(sent_pairs), batch_size), desc="  Computing Faithfulness NLI"):
                batch = sent_pairs[b_idx:b_idx+batch_size]
                batch_premises = [p for p, h in batch]
                batch_hypotheses = [h for p, h in batch]
                
                inputs = nli_tokenizer(
                    batch_premises,
                    batch_hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)
                
                outputs = nli_model(**inputs)
                logits = outputs.logits
                
                batch_pred_ids = torch.argmax(logits, dim=-1).cpu().numpy().flatten().tolist()
                for pred_id in batch_pred_ids:
                    flat_labels.append(get_standard_label(pred_id, nli_model.config))
                    
        # Map sentence predictions back to sample faithfulness scores
        for idx in range(len(dataset.samples)):
            sent_idxs = row_mapping[idx]
            if not sent_idxs:
                faithfulness_scores.append(0.0)
                continue
            row_sent_labels = [flat_labels[j] for j in sent_idxs]
            entailed_count = sum(1 for l in row_sent_labels if l == 'entailment')
            score = entailed_count / len(row_sent_labels)
            faithfulness_scores.append(score)
    else:
        faithfulness_scores = [0.0] * len(dataset.samples)
        
    total_relevancy = sum(relevancy_scores)
    total_faithfulness = sum(faithfulness_scores)
    total_precision = sum(p for p, r in deterministic_scores)
    total_recall = sum(r for p, r in deterministic_scores)
    count = len(dataset.samples)
    
    if count == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    avg_relevancy = total_relevancy / count
    avg_faithfulness = total_faithfulness / count
    avg_precision = total_precision / count
    avg_recall = total_recall / count
    avg_overall = (avg_relevancy + avg_faithfulness + avg_precision + avg_recall) / 4
    
    return avg_relevancy, avg_faithfulness, avg_precision, avg_recall, avg_overall

async def main_async():
    parser = argparse.ArgumentParser(description="Evaluate RAGAS dataset metrics (Answer Relevancy, Faithfulness, Context Precision/Recall) locally.")
    parser.add_argument("--relevancy-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        help="HuggingFace cross-encoder model for Answer Relevancy (default: cross-encoder/ms-marco-MiniLM-L-6-v2)")
    parser.add_argument("--nli-model", type=str, default="deberta_ml", choices=list(NLI_MODELS.keys()),
                        help="NLI model key to use for Faithfulness (default: deberta_ml)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for model inference (default: 32)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"\nLoading local relevancy model '{args.relevancy_model}'...")
    try:
        rel_tokenizer = AutoTokenizer.from_pretrained(args.relevancy_model)
        rel_model = AutoModelForSequenceClassification.from_pretrained(args.relevancy_model).to(device)
        rel_model.eval()
        print("Successfully loaded Relevancy model.")
    except Exception as e:
        print(f"Error loading Relevancy model: {e}")
        return

    nli_model_name = NLI_MODELS[args.nli_model]
    print(f"\nLoading local NLI model '{args.nli_model}' from '{nli_model_name}'...")
    try:
        nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
        nli_model.eval()
        entail_idx = get_entailment_index(nli_model, nli_model_name)
        print(f"Successfully loaded NLI model (Entailment index: {entail_idx}).")
    except Exception as e:
        print(f"Error loading NLI model: {e}")
        return

    current_dir = os.path.dirname(os.path.abspath(__file__))
    questionset_dir = os.path.join(current_dir, "questionset")
    before_dir = os.path.join(questionset_dir, "before")
    after_dir = os.path.join(questionset_dir, "after")

    results_to_print = []
    output_lines = []

    # Evaluate baseline (Before)
    if os.path.exists(before_dir) and glob.glob(os.path.join(before_dir, "*.csv")):
        dataset_before, summaries_before = load_dataset_from_dir(before_dir, "qwen2.5:3b")
        avg_r_before, avg_f_before, avg_p_before, avg_rec_before, avg_o_before = await evaluate_dataset(
            dataset_before, summaries_before, "Baseline (Before)", 
            rel_tokenizer, rel_model, 
            nli_tokenizer, nli_model, entail_idx, 
            device, batch_size=args.batch_size
        )
        results_to_print.append({
            "mode": "Before (Baseline)",
            "generator": "qwen2.5:3b",
            "scores": (avg_r_before, avg_f_before, avg_p_before, avg_rec_before, avg_o_before)
        })
    else:
        pass

    # Evaluate optimized (After)
    if os.path.exists(after_dir) and glob.glob(os.path.join(after_dir, "*.csv")):
        dataset_after, summaries_after = load_dataset_from_dir(after_dir, "qwen2.5-3b-ragas")
        avg_r_after, avg_f_after, avg_p_after, avg_rec_after, avg_o_after = await evaluate_dataset(
            dataset_after, summaries_after, "Optimized (After)", 
            rel_tokenizer, rel_model, 
            nli_tokenizer, nli_model, entail_idx, 
            device, batch_size=args.batch_size
        )
        results_to_print.append({
            "mode": "After (Optimized)",
            "generator": "qwen2.5-3b-ragas",
            "scores": (avg_r_after, avg_f_after, avg_p_after, avg_rec_after, avg_o_after)
        })
    else:
        pass

    # Print comparative matrix
    if len(results_to_print) == 2:
        before = results_to_print[0]
        after = results_to_print[1]
        
        r_b, f_b, p_b, rec_b, o_b = before["scores"]
        r_a, f_a, p_a, rec_a, o_a = after["scores"]
        
        gain_r = r_a - r_b
        gain_f = f_a - f_b
        gain_p = p_a - p_b
        gain_rec = rec_a - rec_b
        gain_o = o_a - o_b
        
        pct_r = (gain_r / r_b) * 100 if r_b > 0 else 0
        pct_f = (gain_f / f_b) * 100 if f_b > 0 else 0
        pct_p = (gain_p / p_b) * 100 if p_b > 0 else 0
        pct_rec = (gain_rec / rec_b) * 100 if rec_b > 0 else 0
        pct_o = (gain_o / o_b) * 100 if o_b > 0 else 0
        
        # Calculate Group Averages
        gen_b = (r_b + f_b) / 2
        gen_a = (r_a + f_a) / 2
        gain_gen = gen_a - gen_b
        pct_gen = (gain_gen / gen_b) * 100 if gen_b > 0 else 0

        ret_b = (p_b + rec_b) / 2
        ret_a = (p_a + rec_a) / 2
        gain_ret = ret_a - ret_b
        pct_ret = (gain_ret / ret_b) * 100 if ret_b > 0 else 0

        output_lines = [
            "================ RAGAS Local Metric Evaluation Comparison ================",
            f"{'Metric':<30} | {'Before (Baseline)':<18} | {'After (Optimized)':<18} | {'Improvement':<12}",
            "-" * 87,
            f"{'Generator Model':<30} | {before['generator']:<18} | {after['generator']:<18} |",
            f"{'Relevancy Model':<30} | {args.relevancy_model.split('/')[-1]:<18} | {args.relevancy_model.split('/')[-1]:<18} |",
            f"{'NLI Model':<30} | {args.nli_model:<18} | {args.nli_model:<18} |",
            "-" * 87,
            " [Generator Metrics]",
            f"  {'Answer Relevancy Score':<28} | {r_b:<18.4f} | {r_a:<18.4f} | {gain_r:<+11.4f} ({pct_r:+.2f}%)",
            f"  {'Faithfulness Score':<28} | {f_b:<18.4f} | {f_a:<18.4f} | {gain_f:<+11.4f} ({pct_f:+.2f}%)",
            "- " * 44,
            " [Retriever Metrics]",
            f"  {'Context Precision Score':<28} | {p_b:<18.4f} | {p_a:<18.4f} | {gain_p:<+11.4f} ({pct_p:+.2f}%)",
            f"  {'Context Recall Score':<28} | {rec_b:<18.4f} | {rec_a:<18.4f} | {gain_rec:<+11.4f} ({pct_rec:+.2f}%)",
            "-" * 87,
            f"{'Generator Metrics Average':<30} | {gen_b:<18.4f} | {gen_a:<18.4f} | {gain_gen:<+11.4f} ({pct_gen:+.2f}%)",
            f"{'Retriever Metrics Average':<30} | {ret_b:<18.4f} | {ret_a:<18.4f} | {gain_ret:<+11.4f} ({pct_ret:+.2f}%)",
            "-" * 87,
            f"{'Overall RAGAS Index':<30} | {o_b:<18.4f} | {o_a:<18.4f} | {gain_o:<+11.4f} ({pct_o:+.2f}%)",
            "=========================================================================="
        ]
    elif len(results_to_print) == 1:
        res = results_to_print[0]
        r, f, p, rec, o = res["scores"]
        gen_avg = (r + f) / 2
        ret_avg = (p + rec) / 2
        output_lines = [
            f"================ RAGAS Local Metric Evaluation: {res['mode']} ================",
            f"Generator Model: {res['generator']}",
            f"Relevancy Model: {args.relevancy_model}",
            f"NLI Model: {args.nli_model}",
            "-" * 60,
            " [Generator Metrics]",
            f"  Answer Relevancy:   {r:.4f}",
            f"  Faithfulness:       {f:.4f}",
            "-" * 60,
            " [Retriever Metrics]",
            f"  Context Precision:  {p:.4f}",
            f"  Context Recall:     {rec:.4f}",
            "-" * 60,
            f"Generator Metrics Average: {gen_avg:.4f}",
            f"Retriever Metrics Average: {ret_avg:.4f}",
            "-" * 60,
            f"Overall RAGAS Index:       {o:.4f}",
            "=========================================================================="
        ]
    else:
        output_lines = ["No evaluation results generated."]

    print()
    for line in output_lines:
        print(line)
    print()

    # Save to file
    filename = "ragas_results.txt"
    results_path = os.path.join(current_dir, filename)
    with open(results_path, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")
    print(f"Results recorded in: {results_path}")

    # Save to CSV for graphing
    try:
        csv_path = os.path.join(current_dir, "ragas_results.csv")
        if len(results_to_print) == 2:
            before = results_to_print[0]
            after = results_to_print[1]
            r_b, f_b, p_b, rec_b, o_b = before["scores"]
            r_a, f_a, p_a, rec_a, o_a = after["scores"]
            
            gen_b = (r_b + f_b) / 2
            gen_a = (r_a + f_a) / 2
            ret_b = (p_b + rec_b) / 2
            ret_a = (p_a + rec_a) / 2
            
            csv_data = [
                {"Metric": "Answer Relevancy Score", "Before": r_b, "After": r_a},
                {"Metric": "Faithfulness Score", "Before": f_b, "After": f_a},
                {"Metric": "Context Precision Score", "Before": p_b, "After": p_a},
                {"Metric": "Context Recall Score", "Before": rec_b, "After": rec_a},
                {"Metric": "Generator Metrics Average", "Before": gen_b, "After": gen_a},
                {"Metric": "Retriever Metrics Average", "Before": ret_b, "After": ret_a},
                {"Metric": "Overall RAGAS Index", "Before": o_b, "After": o_a}
            ]
            df_csv = pd.DataFrame(csv_data)
            df_csv.to_csv(csv_path, index=False)
            print(f"Metrics CSV saved to: {csv_path}\n")
            
        elif len(results_to_print) == 1:
            res = results_to_print[0]
            r, f, p, rec, o = res["scores"]
            gen_avg = (r + f) / 2
            ret_avg = (p + rec) / 2
            
            csv_data = [
                {"Metric": "Answer Relevancy Score", "Before": r, "After": 0.0},
                {"Metric": "Faithfulness Score", "Before": f, "After": 0.0},
                {"Metric": "Context Precision Score", "Before": p, "After": 0.0},
                {"Metric": "Context Recall Score", "Before": rec, "After": 0.0},
                {"Metric": "Generator Metrics Average", "Before": gen_avg, "After": 0.0},
                {"Metric": "Retriever Metrics Average", "Before": ret_avg, "After": 0.0},
                {"Metric": "Overall RAGAS Index", "Before": o, "After": 0.0}
            ]
            df_csv = pd.DataFrame(csv_data)
            df_csv.to_csv(csv_path, index=False)
            print(f"Metrics CSV saved to: {csv_path}\n")
    except Exception as e:
        print(f"Error saving results CSV: {e}\n")

def main():
    start_time = time.time()
    asyncio.run(main_async())
    end_time = time.time()
    print(f"Time taken for evaluation: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
