# Retrieval-Augmented Academic Assistant with Local RAG Implementation

A comprehensive local RAG system designed for querying and analyzing academic papers from the arXiv dataset. This repository contains the RAG engine, web application, hyperparameter tuning utilities, evaluation suites (NLI, Refusal, and RAGAS), and LLM fine-tuning pipelines.

## Project Structure

```directory
├── CORE/
│   ├── core.py                          # Core RAG engine (Hybrid Retriever + Cross-Encoder Reranker)
│   ├── createdb.py                      # Bulk database creation utility
│   ├── generator_evaluation/            # Suite for evaluating generated answers
│   │   ├── pred.py                      # Answer generation driver
│   │   ├── EntailmentScore/             # NLI-based Faithfulness (BART/DeBERTa)
│   │   └── RefusalRate/                 # Refusal rate evaluation (Keyword + NLI)
│   ├── retrieval_evaluation/            # Suite for evaluating document retrieval
│   │   ├── pred.py                      # Retrieval prediction driver
│   │   └── recall.py                    # Recall@N evaluator and visualizer
│   ├── hyperparameterTuning_retriever/  # Grid search tuning for retriever parameters (K & N)
│   │   ├── experiment_kn.py
│   │   └── evaluate_hyperparameters.py
│   └── RAGAS/                           # Local RAGAS evaluation & model fine-tuning
│       ├── pred.py                      # Prediction driver with serialized context
│       ├── ragas_eval.py                # Local Answer Relevancy & Faithfulness evaluator
│       ├── ragas_graph.py               # Comparative chart generator
│       └── fine_tuning/                 # Synthesizing dataset & LoRA fine-tuning pipelines
├── Flask Application/                   # Demo web interface
│   ├── app.py                           # Flask server
│   ├── templates/                       # Frontend HTML
│   └── static/                          # Styling and frontend assets
├── dataset/                             # Raw and category-grouped article metadata
│   └── old/                             # Archived raw datasets (ignored by git)
└── questionSet/                         # Standardized evaluation questions
```

---

## Key System Features

1. **Hybrid Retrieval**: Integrates LangChain Dense Vector Search (Chroma) and BM25 sparse keyword-based search with custom weights to construct a candidate document pool.
2. **Cross-Encoder Re-Ranking**: Reranks candidate document chunks using a MiniLM Cross-Encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to compress the final context and retain the most relevant top-$N$ snippets.
3. **Query Expansion & Enhancement**: Uses a local model (e.g., `llama3.2:1b`) to transform conversational user queries into optimized search keywords for index retrieval.
4. **Local Evaluation Suites**:
   - **Recall@N**: Benchmarks retrieval performance over varying parameters.
   - **NLI Faithfulness**: Evaluates response correctness by splitting answers into individual sentences and testing entailment against gold summaries using models like BART and DeBERTa.
   - **Refusal Rates**: Validates how robustly the LLM refuses to answer unanswerable or negative questions using keyword rules and NLI evaluation.
   - **Local RAGAS Metric Suite**: Computes Context Precision, Context Recall, Answer Relevancy, and Faithfulness locally without needing OpenAI API tokens.

---

## Research & Experimental Results

The technical framework and pipeline design are based on the Master's in Data Science Research Report: **"Retrieval-Augmented Academic Assistant with Local RAG Implementation"**. Below are the key findings and empirical results from the three main stages of experimentation:

### 1. Experiment 1: Retriever Selection
- **Metric**: Recall@5
- **Goal**: Identify the best-performing open-source embedding model for dense retrieval.
- **Results**: 
  - **`granite-embedding:30m`** (IBM) achieved the highest overall average Recall@5 of **89.90%**, peaking at **93.60%** on the 2024 dataset.
  - Notably, retrieval performance did not directly correlate with model size: the larger `qwen3-embedding:0.6b` (1024 dimensions) was outperformed by the smaller `granite-embedding:30m` (384 dimensions), highlighting that training methodology is more critical than parameter scale alone.

### 2. Experiment 2: Generator Selection
- **Metrics**: NLI Semantic Consistency & Refusal Rate
- **Goal**: Identify the most suitable lightweight language model (under 8B parameters) for generating grounded responses and handling unanswerable queries.
- **Results**:
  - **`Qwen2.5-3B`** was selected as the optimal generator. It achieved the highest combined average NLI entailment score of **0.626** and the highest majority-vote semantic consistency rate of **68.10%** across three NLI models.
  - It also displayed a strong refusal capability (**72.50%** keyword-based and **85.00%** NLI-based) on unanswerable questions.

### 3. Experiment 3: Pipeline Optimization & RAGAS Benchmarking
- **Retriever Parameter Tuning**: Grid search identified **$K=10$** (top-ranked chunks from Dense/Sparse) and **$N=6$** (retained chunks after Cross-Encoder reranking) as the optimal elbow point of the Recall@n curve.
- **Generator Fine-Tuning**: Supervised fine-tuning using **QLoRA** on an NLI-filtered synthetic dataset (augmented via `llama3:8b`) produced the optimized `qwen2.5-3b-ragas` model.
- **Local RAGAS Metric Improvements**:
  - **Faithfulness**: **+19.3%** relative improvement (from `0.5519` to `0.6586`), ensuring generated answers are highly grounded in the retrieved context.
  - **Context Recall**: **+3.0%** improvement (from `0.7475` to `0.7700`).
  - **Context Precision**: Decreased by **-17.5%** (from `0.2592` to `0.2140`) as a direct trade-off for higher recall, prioritizing broader context coverage.
  - **Overall RAGAS Index**: **+3.1%** overall improvement (from `0.6165` to `0.6355`).

---

## Setup and Installation

### 1. Prerequisites
- Python 3.10 or higher
- PyTorch (with CUDA support if running on GPU)
- [Ollama](https://ollama.com/) (running locally)

### 2. Local Models (via Ollama)
Ensure you have downloaded the required models:
```bash
ollama pull granite-embedding:30m
ollama pull nomic-embed-text:latest
ollama pull llama3.2:latest
ollama pull qwen2.5:3b
```

### 3. Install Dependencies
Install Python libraries:
```bash
pip install torch transformers pandas numpy scikit-learn langchain langchain-ollama langchain-chroma langchain-community sentence-transformers matplotlib flask tqdm ragas nltk
```

---

## How to Run

### Building the Vector Database
Place your raw `.csv` articles dataset in `dataset/` and run:
```bash
python CORE/createdb.py
```

### Running the Web Application
Start the Flask interface to index documents, configure vector configurations dynamically, and ask questions through a chat UI:
```bash
cd "Flask Application"
python app.py
```
Open your browser and navigate to `http://localhost:5000`.

### Retriever Hyperparameter Tuning
To perform grid-search tuning on retriever parameters ($K$) and reranker parameters ($N$):
```bash
# Generate predictions for combinations of K and N
python CORE/hyperparameterTuning_retriever/experiment_kn.py

# Calculate Recall@N metrics and plot optimization curves
python CORE/hyperparameterTuning_retriever/evaluate_hyperparameters.py
```

### NLI-Based Faithfulness Evaluation
```bash
# Generate LLM responses on the question set
python CORE/generator_evaluation/pred.py

# Run local NLI models to evaluate entailment scores
python CORE/generator_evaluation/EntailmentScore/nli.py

# Create summary metrics and distribution charts
python CORE/generator_evaluation/EntailmentScore/nli_graph.py
```

### Local RAGAS Evaluation
To compare a baseline generator model versus an optimized generator model:
```bash
# Generate predictions for both baseline and optimized models
python CORE/RAGAS/pred.py --mode before
python CORE/RAGAS/pred.py --mode after

# Evaluate local RAGAS metrics
python CORE/RAGAS/ragas_eval.py

# Generate comparison charts
python CORE/RAGAS/ragas_graph.py
```

> [!NOTE]
> The optimized model runs on a custom local Ollama model named `qwen2.5-3b-ragas`. Since the final GGUF file is too large for GitHub (>3 GB), it is ignored by `.gitignore`. However, the **LoRA adapter weights (57 MB)** are fully committed inside `CORE/RAGAS/fine_tuning/qwen2.5-3b-ragas-lora/`.
> 
> To recreate the custom model locally on a new machine:
> 1. **Merge the LoRA weights** with the base model:
>    ```bash
>    python CORE/RAGAS/fine_tuning/merge_and_export.py --base-model Qwen/Qwen2.5-3B-Instruct
>    ```
> 2. **Convert the merged weights to GGUF** using `llama.cpp`:
>    ```bash
>    python path/to/llama.cpp/convert_hf_to_gguf.py CORE/RAGAS/fine_tuning/qwen2.5-3b-ragas-merged --outfile CORE/RAGAS/fine_tuning/finetuned_gguf/qwen2.5-3b-rages.gguf
>    ```
> 3. **Create the Ollama model** from the Modelfile:
>    ```bash
>    cd CORE/RAGAS/fine_tuning/finetuned_gguf
>    ollama create qwen2.5-3b-ragas -f Modelfile
>    ```
