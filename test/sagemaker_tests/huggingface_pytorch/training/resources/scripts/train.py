import os
import logging
import sys
import argparse
import evaluate
import numpy as np
import signal
import psutil
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

def signal_handler(signum, frame):
    logger.warning(f"Received signal {signum}. Stopping training.")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def log_resource_usage():
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    logger.info(f"CPU usage: {cpu_percent}%, Memory usage: {memory_percent}%")

class CustomTrainer(Trainer):
    def training_step(self, model, inputs):
        outputs = super().training_step(model, inputs)
        logger.info(f"Step {self.state.global_step}")
        if self.state.global_step % 10 == 0:  # Log resource usage every 10 steps
            log_resource_usage()
        return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Arguments: {args}")

    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    # download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # load dataset
    dataset = load_dataset("imdb")
    metric = evaluate.load("accuracy")

    # tokenizer helper function
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)

    # load dataset
    train_dataset, test_dataset = load_dataset("imdb", split=["train", "test"])
    test_dataset = test_dataset.shuffle().select(
        range(100)
    )  # smaller the size for test dataset to 10k

    # tokenize dataset
    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

    # set format for pytorch
    logger.info("set train format")
    train_dataset = train_dataset.rename_column("label", "labels")
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    logger.info("set test format")
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # compute metrics function for binary classification
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    # define training args
    logger.info("define training args")
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        dataloader_drop_last=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="steps",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
        logging_steps=10,  # Log every 10 steps
        save_steps=100,    # Save model every 100 steps
    )

    logger.info(f"Training arguments: {training_args}")

    logger.info("training args defined")

    # create Trainer instance
    logger.info("create trainer")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    logger.info("trainer defined")

    # train model
    logger.info("starting to train")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise
    
    logger.info("Training completed")

    # evaluate model
    logger.info("evaluating the model")
    eval_result = trainer.evaluate(eval_dataset=test_dataset)
    logger.info("evaluation ended")

    # writes eval result to file which can be accessed later in s3 ouput
    logger.info("writes eval result to file")
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print("***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    logger.info("file written")

    # Saves the model to s3
    logger.info("saving model")
    trainer.save_model(args.model_dir)
    logger.info("model saved")
