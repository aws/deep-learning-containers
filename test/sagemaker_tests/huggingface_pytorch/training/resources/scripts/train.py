import os
import logging
import sys
import argparse
import evaluate
import numpy as np
import signal
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

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
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout), logging.StreamHandler(sys.stderr)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # download model from model hub
    logger.info(f"args are {args}")
    logger.info("downloading model")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    logger.info("downloading tokenizer")
    # download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    logger.info("loading dataset")
    # load dataset
    train_dataset, test_dataset = load_dataset("imdb", split=["train", "test"])
    logger.info("evaluate load")
    metric = evaluate.load("accuracy")

    # tokenizer helper function
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)

    # load dataset
    logger.info("split train and test dataset")
    test_dataset = test_dataset.shuffle().select(
        range(100)
    )  # smaller the size for test dataset to 10k

    # tokenize dataset
    logger.info("tokenize")

    def timeout_handler(signum, frame):
        raise TimeoutError("Operation timed out")

    def run_with_timeout(func, timeout=5):
        # Set the signal handler and a 5-second alarm
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            result = func()
            signal.alarm(0)  # Clear the alarm
            return result
        except TimeoutError:
            print("Operation took too long, sending interrupt...")
            # Send interrupt to the main thread
            signal.raise_signal(signal.SIGINT)
        finally:
            signal.alarm(0)  # Clear the alarm

    train_dataset = run_with_timeout(
        train_dataset.map(tokenize, num_proc=1, batched=True, batch_size=len(train_dataset)),
        timeout=10,
    )
    test_dataset = run_with_timeout(
        test_dataset.map(tokenize, num_proc=1, batched=True, batch_size=len(test_dataset)),
        timeout=10,
    )

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
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
    )

    # create Trainer instance
    logger.info("create trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # train model
    logger.info("train model")
    trainer.train()

    # evaluate model
    logger.info("evaluate model")
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print("***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Saves the model to s3
    trainer.save_model(args.model_dir)
