"""Parameter-selection strategies used by the RTE benchmark.

Every strategy starts from the same `RobertaForSequenceClassification` model.
The only intended difference between runs is which parameters receive gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


METHODS = ("full_ft", "bitfit", "adapter", "lora")


class BottleneckAdapter(nn.Module):
    """A residual down-project / nonlinearity / up-project adapter block."""

    def __init__(self, hidden_size: int, bottleneck_dim: int, dropout: float) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, hidden_size)
        # Starts as a near-identity residual path, as is common for adapters.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        update = self.up(self.dropout(self.activation(self.down(hidden_states))))
        return hidden_states + update


@dataclass(frozen=True)
class TrainableSummary:
    trainable: int
    total: int

    @property
    def percent(self) -> float:
        return 100 * self.trainable / self.total


def _freeze_all(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def _unfreeze_classifier(model: nn.Module) -> None:
    # RoBERTa sequence classification uses this task-specific head.
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def _add_adapters(model: nn.Module, bottleneck_dim: int, dropout: float) -> None:
    """Insert one adapter after every RoBERTa transformer layer.

    A forward hook is used deliberately so the benchmark stays in ordinary
    `transformers`, without relying on a second, adapter-specific model fork.
    """
    for layer in model.roberta.encoder.layer:
        adapter = BottleneckAdapter(
            hidden_size=model.config.hidden_size,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        )
        layer.add_module("rte_adapter", adapter)

        def apply_adapter(module: nn.Module, _inputs: tuple, output: object) -> object:
            if isinstance(output, tuple):
                return (module.rte_adapter(output[0]), *output[1:])
            return module.rte_adapter(output)

        layer.register_forward_hook(apply_adapter)


def configure_model(model: nn.Module, method: str, config: dict) -> nn.Module:
    """Configure `model` for one benchmark method and return it.

    LoRA is supplied by PEFT; the other three strategies are intentionally
    explicit so it is easy to inspect exactly what is trainable.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}.")

    if method == "full_ft":
        return model

    if method == "bitfit":
        _freeze_all(model)
        for name, parameter in model.named_parameters():
            if name.endswith(".bias"):
                parameter.requires_grad = True
        _unfreeze_classifier(model)
        return model

    if method == "adapter":
        _freeze_all(model)
        _add_adapters(model, config["adapter_dim"], config["adapter_dropout"])
        _unfreeze_classifier(model)
        return model

    # Import lazily so full FT / BitFit / adapters can still be inspected if
    # the optional LoRA dependency has not yet been installed.
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=["query", "value"],
        modules_to_save=["classifier"],
        bias="none",
    )
    return get_peft_model(model, lora_config)


def trainable_summary(model: nn.Module) -> TrainableSummary:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return TrainableSummary(trainable=trainable, total=total)
