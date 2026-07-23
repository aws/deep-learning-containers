"""FSDP fine-tune on GLUE/CoLA via Ray Train + Lightning.

Data is pre-staged to /fsx/hf_cache (HF_HUB_OFFLINE=1, no runtime downloads).
"""

import os

os.environ["HF_HOME"] = "/fsx/hf_cache"
os.environ["HF_HUB_OFFLINE"] = "1"

import pytorch_lightning as pl
import ray
import ray.train
import torch
import torch.nn.functional as F
from datasets import load_dataset
from evaluate import load as load_metric
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import (
    RayFSDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer
from torch.distributed.fsdp import BackwardPrefetch, ShardingStrategy
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "bert-base-cased"
NUM_WORKERS = 8
GLOBAL_BATCH_SIZE = 256
MAX_EPOCHS = 3

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_sentence(batch):
    outputs = tokenizer(
        list(batch["sentence"]),
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )
    outputs["labels"] = batch["label"]
    return outputs


class SentimentModel(pl.LightningModule):
    def __init__(self, lr=2e-5, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        self.metric = load_metric("glue", "cola")
        self.predictions = []
        self.references = []

    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        ).logits

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = F.cross_entropy(logits, batch["labels"])
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        self.predictions.append(torch.argmax(logits, dim=1))
        self.references.append(batch["labels"])

    def on_validation_epoch_end(self):
        preds = torch.cat(self.predictions).cpu().numpy()
        refs = torch.cat(self.references).cpu().numpy()
        score = self.metric.compute(predictions=preds, references=refs)
        self.log_dict(score, sync_dist=True)
        self.predictions.clear()
        self.references.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, eps=self.eps)


def train_func(config):
    world_size = ray.train.get_context().get_world_size()
    per_gpu_batch_size = config["batch_size"] // world_size

    train_shard = ray.train.get_dataset_shard("train")
    val_shard = ray.train.get_dataset_shard("validation")

    train_loader = train_shard.iter_torch_batches(batch_size=per_gpu_batch_size)
    val_loader = val_shard.iter_torch_batches(batch_size=per_gpu_batch_size)

    model = SentimentModel(lr=config["lr"], eps=config["eps"])

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="gpu",
        devices="auto",
        strategy=config["strategy"],
        plugins=[RayLightningEnvironment()],
        callbacks=[RayTrainReportCallback()],
        enable_progress_bar=False,
        enable_checkpointing=True,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Report final loss for the driver to assert on
    final_loss = trainer.callback_metrics.get("train_loss")
    if final_loss is not None:
        ray.train.report({"final_train_loss": final_loss.item(), "world_size": world_size})


def main():
    hf = load_dataset("nyu-mll/glue", "cola")
    train_ds = ray.data.from_items(
        [{"sentence": r["sentence"], "label": r["label"]} for r in hf["train"]]
    ).map_batches(tokenize_sentence, batch_format="numpy")
    val_ds = ray.data.from_items(
        [{"sentence": r["sentence"], "label": r["label"]} for r in hf["validation"]]
    ).map_batches(tokenize_sentence, batch_format="numpy")

    fsdp_strategy = RayFSDPStrategy(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        limit_all_gathers=True,
        cpu_offload=True,
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "lr": 1e-5,
            "eps": 1e-8,
            "batch_size": GLOBAL_BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "strategy": fsdp_strategy,
        },
        scaling_config=ScalingConfig(
            num_workers=NUM_WORKERS, use_gpu=True, resources_per_worker={"GPU": 1}
        ),
        run_config=RunConfig(
            name="bert-cola-fsdp-ci",
            storage_path="/fsx/ray_results",
            checkpoint_config=CheckpointConfig(num_to_keep=1),
        ),
        datasets={"train": train_ds, "validation": val_ds},
    )
    result = trainer.fit()
    metrics = result.metrics

    # --- Assertions ---
    # (a) All 8 GPUs participated
    assert metrics.get("world_size") == NUM_WORKERS, (
        f"Expected world_size={NUM_WORKERS}, got {metrics.get('world_size')}"
    )

    # (b) EFA provider was used for NCCL
    import subprocess

    nccl_log = subprocess.run(
        ["grep", "-r", "Selected provider is efa", "/tmp/ray"],
        capture_output=True,
        text=True,
    )
    assert nccl_log.returncode == 0, "EFA provider not found in NCCL logs (/tmp/ray)"

    # (c) Loss decreased (first epoch vs last epoch)
    first_loss = result.metrics_dataframe["train_loss"].iloc[0]
    last_loss = result.metrics_dataframe["train_loss"].iloc[-1]
    assert last_loss < first_loss, f"Loss did not decrease: {first_loss} -> {last_loss}"

    print(f"EKS_TEST_RESULT: world_size={metrics.get('world_size')}")
    print(f"EKS_TEST_RESULT: train_loss={last_loss:.4f} (from {first_loss:.4f})")
    print("EKS_TEST_RESULT: SUCCESS")


if __name__ == "__main__":
    main()
