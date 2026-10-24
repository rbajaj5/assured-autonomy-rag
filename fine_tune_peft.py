"""LoRA/QLoRA fine-tuning entrypoint for Assured Autonomy instruction data."""
from __future__ import annotations

import argparse

from .paths import DEFAULT_ADAPTER_DIR, DEFAULT_DATASET_PATH


def train() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--out", default=str(DEFAULT_ADAPTER_DIR))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable bitsandbytes QLoRA quantization.")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant,
        device_map="auto",
    )
    if quant is not None:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    )
    model = get_peft_model(model, lora)

    ds = load_dataset("json", data_files=args.dataset, split="train")

    def format_row(row):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a JHU Assured Autonomy RAG assistant. Distinguish "
                    "module concepts precisely and cite local readings in the "
                    "required format."
                ),
            },
            {
                "role": "user",
                "content": f"{row['instruction']}\n\n{row['input']}",
            },
            {"role": "assistant", "content": row["output"]},
        ]
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            prompt = (
                "System: " + messages[0]["content"] + "\n\n"
                "User: " + messages[1]["content"] + "\n\n"
                "Assistant: " + messages[2]["content"]
            )
        return tokenizer(prompt, truncation=True, max_length=args.max_length)

    tokenized = ds.map(format_row, remove_columns=ds.column_names)
    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)


if __name__ == "__main__":
    train()
