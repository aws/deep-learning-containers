# Adapted from https://www.kaggle.com/code/hassanamin/bert-pytorch-cola-classification/notebook

import argparse
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import time
import torch
import torch.distributed as dist

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from smart_sifting.data_model.data_model_interface import SiftingBatch
from smart_sifting.dataloader.sift_dataloader import SiftingDataloader
from smart_sifting.loss.abstract_sift_loss_module import Loss
from smart_sifting.metrics.lightning import TrainingMetricsRecorder
from smart_sifting.sift_config.sift_configs import (
    RelativeProbabilisticSiftConfig,
    LossConfig,
    SiftingBaseConfig,
)


RANDOM_SEED = 7
logger = logging.getLogger(__name__)


class BertLoss(Loss):
    """
    This is an implementation of the Loss interface for the BERT model
    required for Smart Sifting. Use Cross-Entropy loss with 2 classes
    """

    def __init__(self):
        self.celoss = torch.nn.CrossEntropyLoss(reduction="none")

    def loss(
        self,
        model: torch.nn.Module,
        transformed_batch: SiftingBatch,
        original_batch: Any = None,
    ) -> torch.Tensor:
        # get original batch onto model device. Note that we are assuming the model is on the right device here already
        # Pytorch lightning takes care of this under the hood with the model thats passed in.
        # TODO: ensure batch and model are on the same device in SiftDataloader so that the customer
        #  doesn't have to implement this
        device = next(model.parameters()).device
        batch = [t.to(device) for t in original_batch]

        # compute loss
        outputs = model(batch)
        return self.celoss(outputs.logits, batch[2])


class ColaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sifting_enabled: bool,
        batch_size: int,
        model: torch.nn.Module,
        beta: float,
        history_size: int,
        sift_delay: int,
    ):
        super().__init__()
        self.sifting_enabled = sifting_enabled
        self.batch_size = batch_size
        self.model = model
        self.beta = beta
        self.history_size = history_size
        self.sift_delay = sift_delay

    def setup(self, stage: str) -> None:
        """
        Loads the data from s3, splits it into multi-batches.
        This logic is dataset specific.
        """

        logger.info(f"Preprocessing CoLA dataset")
        dataset = load_dataset("linxinyuan/cola")["train"]
        dataset = dataset.rename_column("text", "sentence")
        dataframe = dataset.to_pandas()

        # Split dataframes (Note: we use scikitlearn here because pytorch random_split doesn't work as intended - theres
        # a bug when we pass in proportions to random_split (https://stackoverflow.com/questions/74327447/how-to-use-random-split-with-percentage-split-sum-of-input-lengths-does-not-equ)
        # and we get a KeyError when iterating through the resulting split datasets)
        logger.info(f"Splitting dataframes into train, val, and test")
        train_df, test_df = train_test_split(dataframe, train_size=0.9, random_state=RANDOM_SEED)
        train_df, val_df = train_test_split(train_df, train_size=0.9, random_state=RANDOM_SEED)

        # Finally, transform the dataframes into PyTorch datasets
        logger.info(f"Transforming dataframes into datasets")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        max_sentence_length = 64

        self.train = self._transform_to_dataset(train_df, tokenizer, max_sentence_length)
        self.val = self._transform_to_dataset(val_df, tokenizer, max_sentence_length)
        self.test = self._transform_to_dataset(test_df, tokenizer, max_sentence_length)

        logger.info("Done preprocessing CoLA dataset")

    def train_dataloader(self):
        original_dataloader = DataLoader(self.train, self.batch_size, shuffle=True)
        if self.sifting_enabled:
            sift_config = RelativeProbabilisticSiftConfig(
                beta_value=self.beta,
                loss_history_length=self.history_size,
                loss_based_sift_config=LossConfig(
                    sift_config=SiftingBaseConfig(sift_delay=self.sift_delay)
                ),
            )
            return SiftingDataloader(
                sift_config=sift_config,
                orig_dataloader=original_dataloader,
                loss_impl=BertLoss(),
                model=self.model,
            )
        else:
            return original_dataloader

    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test, self.batch_size)

    def _transform_to_dataset(self, dataframe: pd.DataFrame, tokenizer, max_sentence_length):
        sentences = dataframe.sentence.values
        labels = dataframe.label.values

        input_ids = []
        for sent in sentences:
            encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
            input_ids.append(encoded_sent)

        # pad shorter sentences
        input_ids_padded = []
        for i in input_ids:
            while len(i) < max_sentence_length:
                i.append(0)
            input_ids_padded.append(i)
        input_ids = input_ids_padded

        # mask; 0: added, 1: otherwise
        attention_masks = []
        # For each sentence...
        for sent in input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)

        if torch.cuda.is_available():
            device = f"cuda:{(dist.get_rank() % torch.cuda.device_count()) if dist.is_initialized() else '0'}"
        else:
            device = "cpu"

        # convert to PyTorch data types.
        inputs = torch.tensor(input_ids, device=device)
        labels = torch.tensor(labels, device=device)
        masks = torch.tensor(attention_masks, device=device)

        return TensorDataset(inputs, masks, labels)


class BertLitModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._create_model()
        self.celoss = torch.nn.CrossEntropyLoss()

    def forward(self, batch):
        return self.model(batch[0], token_type_ids=None, attention_mask=batch[1])

    def training_step(self, batch, batch_idx):
        # Forward Pass
        outputs = self(batch)
        loss = self.celoss(outputs.logits, batch[2])
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        outputs = self(batch)

        # Move tensors to CPU
        logits = outputs.logits.detach().cpu().numpy()
        label_ids = batch[2].to("cpu").numpy()

        # compute accuracy
        acc = self._flat_accuracy(logits, label_ids)

        if stage:
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        logger.info("Initializing AdamW optimizer")
        optimizer = AdamW(
            self.parameters(),
            lr=2e-5,  # args.learning_rate - default is 5e-5, this script has 2e-5
            eps=1e-8,  # args.adam_epsilon - default is 1e-8.
        )

        # Create the learning rate scheduler.
        logger.info("Initializing learning rate scheduler")
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,  # Default value in run_glue.py
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return [optimizer], [scheduler]

    def _create_model(self):
        logger.info("Creating BertForSequenceClassification")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=2,  # The number of output labels--2 for binary classification.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )

        return model

    # Function to calculate the accuracy of our predictions vs labels
    def _flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)


def main(args):
    # Setting up logger config
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d PID:%(process)d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=args.log_level,
        # force=True
    )
    # logging.getLogger("smart_sifting.dataloader.sift_dataloader").setLevel(logging.WARNING)
    # logging.getLogger("smart_sifting.dataloader.sift_dataloader").setLevel(logging.WARNING)
    pl.seed_everything(RANDOM_SEED)

    model = BertLitModule()
    # For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32.
    data = ColaDataModule(
        sifting_enabled=True,
        batch_size=args.batch_size,
        model=model,
        beta=args.beta,
        history_size=args.history_size,
        sift_delay=args.sift_delay,
    )

    trainer = pl.Trainer(
        # Authors recommend 2 - 4
        max_epochs=args.epochs,
        strategy="ddp",
        num_nodes=args.num_nodes,
        # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
        gradient_clip_val=1,
        callbacks=[TrainingMetricsRecorder()],
    )

    logger.info(
        f"Starting training with sifting (beta={args.beta}, delay={args.sift_delay}, and history size={args.history_size})."
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        metavar="N",
        help="number of training nodes (default: 1)",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, metavar="N", help="number of training epochs (default: 2)"
    )
    parser.add_argument(
        "--log_level", type=int, default=30, metavar="N", help="log level (default: 1)"
    )
    parser.add_argument(
        "--beta", type=float, default=1, help="beta value for probabilistic sifting"
    )
    parser.add_argument(
        "--history_size", type=int, default=500, help="size of sifting loss history queue"
    )
    parser.add_argument("--sift_delay", type=int, default=0, help="delay in steps to start sifting")
    parser.add_argument("--batch_size", type=int, default=32, help="training batch size")

    args = parser.parse_args()

    main(args)
    logger.info(f"Total time: {time.perf_counter() - start_time}")
