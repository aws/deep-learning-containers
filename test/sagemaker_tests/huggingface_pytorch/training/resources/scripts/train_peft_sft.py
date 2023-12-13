import os
import logging
import sys
import argparse
import evaluate
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
from peft import LoraConfig, PeftModel


def formatting_prompts_func(example):
    text = f"### Question: {example['question']}\n ### Answer: {example['answer']}"
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--output_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=2)
    parser.add_argument("--r", type=int, default=16)  # the dimension of the low-rank matrices
    parser.add_argument(
        "--lora_alpha", type=int, default=32
    )  # the scaling factor for the low-rank matrices
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05
    )  # the dropout probability of the LoRA layers

    args = parser.parse_args()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # load dataset
    dummy_dataset = Dataset.from_dict(
        {
            "question": [
                "Does llamas know how to code?",
                "Does llamas know how to fly?",
                "Does llamas know how to talk?",
                "Does llamas know how to code?",
                "Does llamas know how to fly?",
                "Does llamas know how to talk?",
                "Does llamas know how to swim?",
            ],
            "answer": [
                "Yes, llamas are very good at coding.",
                "No, llamas can't fly.",
                "Yes, llamas are very good at talking.",
                "Yes, llamas are very good at coding.",
                "No, llamas can't fly.",
                "Yes, llamas are very good at talking.",
                "No, llamas can't swim.",
            ],
            "text": [
                "### Question: Does llamas know how to code?\n ### Answer: Yes, llamas are very good at coding.",
                "### Question: Does llamas know how to fly?\n ### Answer: No, llamas can't fly.",
                "### Question: Does llamas know how to talk?\n ### Answer: Yes, llamas are very good at talking.",
                "### Question: Does llamas know how to code?\n ### Answer: Yes, llamas are very good at coding.",
                "### Question: Does llamas know how to fly?\n ### Answer: No, llamas can't fly.",
                "### Question: Does llamas know how to talk?\n ### Answer: Yes, llamas are very good at talking.",
                "### Question: Does llamas know how to swim?\n ### Answer: No, llamas can't swim.",
            ],
        }
    )
    train_dataset = ConstantLengthDataset(
        tokenizer,
        dummy_dataset,
        dataset_text_field=None,
        formatting_func=formatting_prompts_func,
        seq_length=16,
        num_of_sequences=16,
    )
    eval_dataset = ConstantLengthDataset(
        tokenizer,
        dummy_dataset,
        dataset_text_field=None,
        formatting_func=formatting_prompts_func,
        seq_length=16,
        num_of_sequences=16,
    )
    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded eval_dataset length is: {len(eval_dataset)}")

    # build peft config
    peft_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # define training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        per_device_train_batch_size=args.train_batch_size,
    )

    # create Trainer instance
    trainer = SFTTrainer(
        model=args.model_id,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=True,
    )
    self.assertTrue(isinstance(trainer.model, PeftModel))

    # train model
    trainer.train()
