#!/usr/bin/env python3
"""Train a LoRA adapter from IDE accept events (FIM objective).

Takes the accept/reject telemetry produced by generate_synthetic_events.py
(or a real export in the same schema), keeps accepted completions, and trains
a LoRA on Qwen's FIM format with loss on the completion tokens only.

    python training/train_lora.py \
        --base-model Qwen/Qwen2.5-Coder-3B \
        --events training/data/events.jsonl \
        --out adapters/customer-a

The adapter is then hot-loaded into vLLM without a restart (serving/hotswap.py)
and must pass the eval gate (evals/eval_gate.py) before receiving traffic.
"""

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

FIM_TEMPLATE = "<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"


class FimDataset(Dataset):
    """FIM prompt + accepted completion; prompt tokens masked out of the loss."""

    def __init__(self, events, tokenizer, max_len=1024):
        self.samples = []
        for e in events:
            prompt = FIM_TEMPLATE.format(prefix=e["prefix"], suffix=e["suffix"])
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(
                e["completion"] + tokenizer.eos_token, add_special_tokens=False
            )["input_ids"]
            input_ids = (prompt_ids + completion_ids)[:max_len]
            labels = ([-100] * len(prompt_ids) + completion_ids)[:max_len]
            self.samples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch, pad_token_id):
    max_len = max(len(s["input_ids"]) for s in batch)
    input_ids, labels, attention = [], [], []
    for s in batch:
        pad = max_len - len(s["input_ids"])
        input_ids.append(s["input_ids"] + [pad_token_id] * pad)
        labels.append(s["labels"] + [-100] * pad)
        attention.append([1] * len(s["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-3B")
    parser.add_argument("--events", default="training/data/events.jsonl")
    parser.add_argument("--out", default="adapters/customer-a")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    events = [json.loads(line) for line in Path(args.events).read_text().splitlines()]
    accepted = [e for e in events if e["accepted"]]
    print(f"{len(events)} events -> {len(accepted)} accepted used for training")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(Path(args.out) / "_checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=2,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=5,
            save_strategy="no",
            bf16=use_cuda,
            report_to=[],
        ),
        train_dataset=FimDataset(accepted, tokenizer),
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    trainer.train()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"adapter saved -> {out.resolve()}")
    print("next: gate it (evals/eval_gate.py), then hot-load it (serving/hotswap.py)")


if __name__ == "__main__":
    main()
