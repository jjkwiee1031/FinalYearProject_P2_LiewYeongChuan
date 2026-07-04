import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configurable paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ADAPTER_DIR = os.path.join(CURRENT_DIR, "qwen2.5-3b-ragas-lora")
DEFAULT_MERGED_DIR = os.path.join(CURRENT_DIR, "qwen2.5-3b-ragas-merged")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Hugging Face base model ID")
    parser.add_argument("--adapter-dir", type=str, default=DEFAULT_ADAPTER_DIR, help="Path to LoRA adapter weights directory")
    parser.add_argument("--merged-dir", type=str, default=DEFAULT_MERGED_DIR, help="Path to save merged model")
    args = parser.parse_args()

    print("=" * 60)
    print("PEFT LoRA Adapter Fusing & Merging Script")
    print(f"Base Model: {args.base_model}")
    print(f"Adapter Dir: {args.adapter_dir}")
    print(f"Merged Output Dir: {args.merged_dir}")
    print("=" * 60)

    # 1. Load tokenizer
    print(f"Loading tokenizer from {args.adapter_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)

    # 2. Load Base Model in Float16 / BFloat16 (Do not quantize!)
    print(f"Loading base model '{args.base_model}' in Float16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu", # Perform merge on CPU to avoid running out of GPU memory
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # 3. Load Peft Model with adapter weights
    print(f"Loading adapter weights from '{args.adapter_dir}'...")
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_dir
    )

    # 4. Merge weights
    print("Merging adapter weights into base model...")
    merged_model = model.merge_and_unload()

    # Free up memory before saving to avoid CPU OOM in Google Colab CPU runtime
    import gc
    del base_model
    del model
    gc.collect()

    # 5. Save merged model and tokenizer
    print(f"Saving merged model to '{args.merged_dir}'...")
    os.makedirs(args.merged_dir, exist_ok=True)
    merged_model.save_pretrained(args.merged_dir, max_shard_size="2GB")
    tokenizer.save_pretrained(args.merged_dir)

    print("\n[Success] Fused model saved successfully!")
    print(f"Merged model path: {args.merged_dir}")
    print("\nNext steps:")
    print("1. Convert to GGUF using llama.cpp:")
    print("   python llama.cpp/convert_hf_to_gguf.py qwen2.5-3b-ragas-merged --outfile qwen2.5-3b-ragas.gguf")
    print("2. Create Ollama custom model using a Modelfile:")
    print("   ollama create qwen2.5-3b-ragas -f Modelfile")

if __name__ == "__main__":
    main()
