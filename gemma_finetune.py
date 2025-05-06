#!/usr/bin/env python3
import os
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,    # bitsandbytes quant config :contentReference[oaicite:4]{index=4}
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def main():
    # 1. Load & split the CSV
    ds = load_dataset("csv", data_files="generated_programs.csv")
    full_ds = ds["train"].rename_column("Fault", "label")
    splits = full_ds.train_test_split(test_size=0.2, seed=42)
    train_ds, eval_ds = splits["train"], splits["test"]

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess(ex):
        prompt = (
            "### Human:\nHere is a program:\n"
            f"{ex['Program']}\n\nWhich line contains the bug?\n\n### Assistant:\n"
        )
        inp = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
        lbl = tokenizer(str(ex["label"]), truncation=True, padding="max_length", max_length=16)
        input_ids = inp["input_ids"] + lbl["input_ids"]
        attention_mask = inp["attention_mask"] + [1] * len(lbl["input_ids"])
        labels = [-100] * len(inp["input_ids"]) + lbl["input_ids"]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    eval_ds  = eval_ds.map(preprocess,  remove_columns=eval_ds.column_names)

    # 3. Quantize & load Gemma-3-12B in 4-bit NF4
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-12b-it",
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(base_model)  # ready for 4-bit training

    # 4. Attach LoRA adapter
    lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, lora_cfg)

    # 5. Disable KV cache for training
    model.config.use_cache = False

    # 6. Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 7. Metrics: exact-match accuracy
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        exact = [p.strip()==l.strip() for p,l in zip(decoded_preds, decoded_labels)]
        return {"accuracy": float(sum(exact))/len(exact)}

    # 8. Training arguments
    training_args = TrainingArguments(
        output_dir="gemma3-fault-loc",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=3e-4,
        num_train_epochs=3,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=20,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        label_names=["labels"]
    )

    # 9. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # 10. Save LoRA adapter only
    model.save_pretrained("gemma3-fault-loc-lora")
    tokenizer.save_pretrained("gemma3-fault-loc-lora")

if __name__ == "__main__":
    main()
