"""Train one RoBERTa-base adaptation method on GLUE RTE.

Example:
    python train.py --method lora --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from methods import METHODS, configure_model, trainable_summary


METHOD_LEARNING_RATES = {
    "full_ft": 2e-5,
    "bitfit": 1e-3,
    "adapter": 1e-3,
    "lora": 5e-4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "directml"),
        default="auto",
        help="'auto' uses CUDA if present, otherwise CPU. DirectML is optional on AMD Windows.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available.")
        return torch.device("cuda")
    if requested == "directml":
        try:
            import torch_directml
        except ImportError as error:
            raise RuntimeError(
                "DirectML needs the optional package: pip install torch-directml"
            ) from error
        return torch_directml.device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: Path, args: argparse.Namespace) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    for arg_name, config_name in (
        ("learning_rate", "learning_rate"),
        ("epochs", "epochs"),
        ("train_batch_size", "train_batch_size"),
        ("max_length", "max_length"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            config[config_name] = value
    config.setdefault("learning_rate", METHOD_LEARNING_RATES[args.method])
    return config


def make_optimizer(model: torch.nn.Module, weight_decay: float, learning_rate: float) -> AdamW:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".bias") or "LayerNorm.weight" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for batch in loader:
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        outputs = model(**batch)
        loss_sum += outputs.loss.item() * batch["labels"].size(0)
        predictions = outputs.logits.argmax(dim=-1)
        correct += (predictions == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    return {"accuracy": correct / total, "loss": loss_sum / total}


def snapshot_trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy the state needed to restore the current best validation model."""
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }


def write_model_card(artifact_dir: Path, record: dict) -> None:
    method = record["method"]
    score = record["best_validation"]["accuracy"]
    if method in {"full_ft", "bitfit"}:
        usage = "```python\nfrom transformers import pipeline\nclassifier = pipeline(\"text-classification\", model=\"YOUR_REPOSITORY\")\n```"
    elif method == "lora":
        usage = "```python\nfrom peft import PeftModel\nfrom transformers import AutoModelForSequenceClassification\nbase = AutoModelForSequenceClassification.from_pretrained(\"roberta-base\", num_labels=2)\nmodel = PeftModel.from_pretrained(base, \"YOUR_REPOSITORY\")\n```"
    else:
        usage = "```python\n# See load_adapter.py in this repository.\nfrom load_adapter import load_model\nmodel, tokenizer = load_model(\".\")\n```"
    card = f"""---
language: en
library_name: transformers
base_model: roberta-base
tags:
- text-classification
- rte
- parameter-efficient-fine-tuning
---

# RoBERTa-base fine-tuned on GLUE RTE ({method})

This checkpoint is one learning experiment comparing Full Fine-Tuning,
BitFit, adapters, and LoRA on GLUE Recognizing Textual Entailment (RTE).
It uses seed {record['seed']} and selects the highest validation epoch.

| Metric | Value |
| --- | ---: |
| Best validation accuracy | {score:.4f} |
| Best epoch | {record['best_validation']['epoch']} |
| Trainable parameters | {record['trainable_parameters']:,} |
| Total parameters | {record['total_parameters']:,} |

## Method

`{method}`. See the project README for the exact shared training configuration.
This is an educational experiment, not a benchmark-level performance claim.

## Load

{usage}
"""
    (artifact_dir / "README.md").write_text(card, encoding="utf-8")


def save_best_artifact(
    model: torch.nn.Module,
    tokenizer: object,
    method: str,
    config: dict,
    record: dict,
    artifact_dir: Path,
) -> None:
    """Write an upload-ready directory for the selected best epoch."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(artifact_dir)
    (artifact_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (artifact_dir / "training_result.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    if method in {"full_ft", "bitfit", "lora"}:
        # LoRA's PEFT wrapper saves the compact adapter and classifier only.
        model.save_pretrained(artifact_dir, safe_serialization=True)
    else:
        # Custom adapter blocks require their construction code at load time.
        torch.save(snapshot_trainable_state(model), artifact_dir / "adapter_checkpoint.pt")
        shutil.copy2(Path("methods.py"), artifact_dir / "methods.py")
        loader = '''"""Load this experiment's custom RTE adapter checkpoint."""
from pathlib import Path
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from methods import configure_model

def load_model(path="."):
    path = Path(path)
    config = json.loads((path / "experiment_config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    model = configure_model(model, "adapter", config)
    state = torch.load(path / "adapter_checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    return model, tokenizer
'''
        (artifact_dir / "load_adapter.py").write_text(loader, encoding="utf-8")
    write_model_card(artifact_dir, record)


def main() -> None:
    args = parse_args()
    if platform.system() == "Windows" and not (3, 9) <= tuple(map(int, platform.python_version_tuple()[:2])) <= (3, 12):
        raise RuntimeError("Use Python 3.9–3.12 on Windows; see README.md.")

    config = load_config(args.config, args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    # `datasets` 5.x expects the Hub namespace. This is the maintained GLUE
    # repository; `rte` selects its Recognizing Textual Entailment subset.
    raw_datasets = load_dataset("nyu-mll/glue", "rte")

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            truncation=True,
            max_length=config["max_length"],
        )

    columns_to_remove = [
        column for column in raw_datasets["train"].column_names if column != "label"
    ]
    tokenized = raw_datasets.map(tokenize, batched=True, remove_columns=columns_to_remove)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch")
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        tokenized["train"],
        batch_size=config["train_batch_size"],
        shuffle=True,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        tokenized["validation"],
        batch_size=config["eval_batch_size"],
        shuffle=False,
        collate_fn=collator,
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=2
    )
    model = configure_model(base_model, args.method, config).to(device)
    summary = trainable_summary(model)
    print(f"Trainable parameters: {summary.trainable:,} / {summary.total:,} ({summary.percent:.3f}%)")

    optimizer = make_optimizer(model, config["weight_decay"], config["learning_rate"])
    updates_per_epoch = math.ceil(len(train_loader) / config["gradient_accumulation_steps"])
    total_updates = updates_per_epoch * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_updates * config["warmup_ratio"]),
        num_training_steps=total_updates,
    )

    best = {"accuracy": -1.0, "epoch": None}
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")
        for step, batch in enumerate(progress, start=1):
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            loss = model(**batch).loss / config["gradient_accumulation_steps"]
            loss.backward()
            if step % config["gradient_accumulation_steps"] == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{loss.item() * config['gradient_accumulation_steps']:.4f}")

        metrics = evaluate(model, eval_loader, device)
        print(f"Epoch {epoch}: validation accuracy={metrics['accuracy']:.4f}, loss={metrics['loss']:.4f}")
        if metrics["accuracy"] > best["accuracy"]:
            best = {**metrics, "epoch": epoch}
            best_state = snapshot_trainable_state(model)

    elapsed_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("No validation checkpoint was created.")
    model.load_state_dict(best_state, strict=False)
    artifact_dir = args.output_dir / f"roberta-base-rte-{args.method}-seed{args.seed}"
    record = {
        "method": args.method,
        "seed": args.seed,
        "best_validation": best,
        "trainable_parameters": summary.trainable,
        "total_parameters": summary.total,
        "trainable_percent": summary.percent,
        "elapsed_seconds": elapsed_seconds,
        "device": str(device),
        "config": config,
        "artifact_dir": str(artifact_dir),
    }
    save_best_artifact(model, tokenizer, args.method, config, record, artifact_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.results_dir / f"{args.method}_seed{args.seed}.json"
    result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved result to {result_path}")
    print(f"Saved best-epoch model artifact to {artifact_dir}")


if __name__ == "__main__":
    main()
