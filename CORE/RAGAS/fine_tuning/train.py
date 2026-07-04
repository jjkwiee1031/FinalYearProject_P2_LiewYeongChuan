import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Configurable paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(CURRENT_DIR, "final_training_dataset.json")
DEFAULT_OUTPUT_DIR = os.path.join(CURRENT_DIR, "qwen2.5-3b-ragas-lora")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--dataset-path", type=str, default=DATASET_PATH, help="Path to final_training_dataset.json")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save output adapter")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank r")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen 2.5 3B QLoRA Fine-Tuning Script")
    print(f"Model ID: {args.model_id}")
    print(f"Dataset Path: {args.dataset_path}")
    print(f"Output Dir: {args.output_dir}")
    print("=" * 60)

    # 1. Check GPU availability and bfloat16 support
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    use_bf16 = False
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            use_bf16 = True
            print("GPU supports bfloat16. Using BF16 training.")
        else:
            print("GPU does not support bfloat16. Using FP16 training.")
    else:
        print("WARNING: CUDA is not available. Fine-tuning on CPU will be extremely slow.")

    # 2. Load dataset
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Training dataset not found at {args.dataset_path}")
    
    dataset = load_dataset("json", data_files=args.dataset_path)
    print(f"Loaded training dataset containing {len(dataset['train'])} samples.")

    # 3. Setup QLoRA configuration (4-bit quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else (torch.float16 if device == "cuda" else torch.float32),
        bnb_4bit_use_double_quant=True
    )

    # 4. Load Base Model and Tokenizer
    print(f"Loading tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading quantized base model {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config if device == "cuda" else None,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    if device == "cuda":
        model = prepare_model_for_kbit_training(model)

    # 5. Configure LoRA PEFT
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    print("LoRA Adapter Model Summary:")
    model.print_trainable_parameters()

    # 6. Configure Training Arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        fp16=not use_bf16 if device == "cuda" else False,
        bf16=use_bf16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        report_to="none",
        max_length=1024,
        dataset_kwargs={
            "skip_prepare_dataset": False # Enables SFTTrainer to process messages list automatically
        }
    )

    # 7. SFTTrainer Initialization
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
        args=training_args
    )

    # 8. Start Training
    print("Starting training...")
    trainer.train()

    # 9. Save the trained Adapter and training state
    print(f"Saving fine-tuned adapter and state to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    trainer.save_state()
    print("Training finished successfully!")

if __name__ == "__main__":
    main()
