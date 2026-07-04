import os
import json
import asyncio
import argparse
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# Configurable paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CURRENT_DIR, "filtered_real_dataset.json")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "synthetic_dataset.json")

SYSTEM_PROMPT = """You are a dataset generator assistant. Your task is to generate high-quality synthetic training data for a RAG-based question answering model.
You will be provided with a research paper's Title & Abstract (Context), an original Question, and an original Answer.

Generate exactly 4 variations of the question along with their corresponding answers based ONLY on the provided Context.
The 4 variations must be:
1. "paraphrased": A paraphrased question with similar meaning but different phrasing.
2. "alternative": An alternative phrasing (e.g. conversational style, or search query style).
3. "simplified": A simplified, more direct version of the question.
4. "complex": A more complex version of the question (e.g. combining concepts or requiring multi-step reasoning from the context).

Each generated answer MUST follow these strict guidelines:
- Base the answer only on the provided context; do not hallucinate or add unsupported claims.
- Write in an academic and objective tone, keep it concise and direct (in 2-3 sentences).
- If the context does not contain the information needed, the answer must be 'No Answer'.

You must respond ONLY with a JSON object in this format:
{
  "variations": [
    {
      "type": "paraphrased",
      "question": "...",
      "response": "..."
    },
    {
      "type": "alternative",
      "question": "...",
      "response": "..."
    },
    {
      "type": "simplified",
      "question": "...",
      "response": "..."
    },
    {
      "type": "complex",
      "question": "...",
      "response": "..."
    }
  ]
}
Do not include any other text, markdown blocks, or explanations. Ensure the output is valid JSON.
"""

USER_TEMPLATE = """Context:
{context}

Original Question:
{question}

Original Answer:
{response}
"""

async def generate_sample_variations(client, model, sample, semaphore):
    async with semaphore:
        prompt = USER_TEMPLATE.format(
            context=sample["context"],
            question=sample["question"],
            response=sample["response"]
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
                
                raw_content = response.choices[0].message.content.strip()
                
                # Parse JSON
                parsed = json.loads(raw_content)
                if "variations" in parsed and len(parsed["variations"]) == 4:
                    # Enrich variations with original context and metadata
                    variations = []
                    for var in parsed["variations"]:
                        variations.append({
                            "question": var["question"].strip(),
                            "context": sample["context"],
                            "response": var["response"].strip(),
                            "metadata": {
                                "variation_type": var["type"],
                                "original_question": sample["question"],
                                "original_metadata": sample["metadata"]
                            }
                        })
                    return variations
                else:
                    print(f"Warning: Response for question '{sample['question'][:40]}...' did not contain 4 variations. Retrying...")
            except Exception as e:
                print(f"Error on attempt {attempt+1} for '{sample['question'][:40]}...': {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
        return []

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3:8b", help="Model name in Ollama")
    parser.add_argument("--concurrency", type=int, default=3, help="Max parallel requests to Ollama")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of input samples (for testing)")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 2: Synthetic Data Generation")
    print(f"Using model: {args.model}")
    print(f"Concurrency: {args.concurrency}")
    print("=" * 60)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found at {INPUT_PATH}. Run prepare_data.py first.")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if args.limit > 0:
        samples = samples[:args.limit]
        print(f"Limiting to first {args.limit} samples.")

    print(f"Loaded {len(samples)} real samples. Generating 4 variations per sample...")

    # Initialize Async OpenAI Client pointing to local Ollama
    client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    
    tasks = [
        generate_sample_variations(client, args.model, sample, semaphore)
        for sample in samples
    ]
    
    all_results = []
    
    # Run with async progress bar
    results_list = await tqdm.gather(*tasks, desc="Generating synthetic variations")
    
    for r in results_list:
        all_results.extend(r)
        
    # Save output
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
        
    print(f"\n[Success] Generated {len(all_results)} synthetic samples.")
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
