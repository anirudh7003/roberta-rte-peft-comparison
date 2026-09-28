# RoBERTa-base on RTE: Full FT vs BitFit vs Adapters vs LoRA

This is a small, reproducible comparison of four ways to adapt the same
`roberta-base` model to the GLUE Recognizing Textual Entailment (RTE) task.
It is inspired by LoRA, rather than an exact reproduction of every setting in
the LoRA paper. The data, tokenizer, evaluation split, optimizer family, seed,
and training loop are shared; only the trainable-parameter strategy changes.

## Important hardware note

Your AMD Radeon RX 6500M is **not a CUDA GPU**. Ordinary Windows PyTorch uses
CUDA only for NVIDIA acceleration, so `--device auto` will normally choose CPU.
DirectML may work as an optional AMD route, but first get a CPU smoke test
working. With only 4 GB VRAM, keep the included batch size of 4 and gradient
accumulation of 4 (effective batch size 16).

Your installed Python is 3.14. Current stable Windows PyTorch supports Python
3.9–3.12, so install Python 3.12 first, then open a new PowerShell window.

## Setup (PowerShell)

```powershell
cd C:\Users\aniru\Desktop\LoRA
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents activation, run this once for your user account and
restart PowerShell:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Optional DirectML experiment after the CPU run succeeds:

```powershell
pip install torch-directml
python train.py --method lora --device directml
```

## First run

Start with LoRA because it trains few parameters and makes a fast functional
test. The first invocation downloads `roberta-base` and the `rte` subset from
the maintained `nyu-mll/glue` dataset repository.

```powershell
python train.py --method lora --seed 42
python summarize.py
```

Then run the other methods one at a time:

```powershell
python train.py --method full_ft --seed 42
python train.py --method bitfit --seed 42
python train.py --method adapter --seed 42
```

Each completed run now writes its best-validation model artifact to `outputs/`.
After all four artifacts exist and you have logged in with `hf auth login`,
publish them with:

```powershell
python publish_models.py --username YOUR_HUGGING_FACE_USERNAME
```

Once those work, run all four methods for seeds 42, 43, and 44:

```powershell
.\run_all.ps1
```

## Recorded experiment

This section records the completed learning experiment. All methods used
`roberta-base`, GLUE RTE, seed 42, maximum sequence length 256, batch size 4,
gradient accumulation 4 (effective batch size 16), 10 epochs, and CPU training
on an AMD Ryzen 7 5800H. Each published artifact is the model from the epoch
with the highest validation accuracy.

| Method | Best epoch | Validation accuracy | Trainable parameters | Trainable share | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full FT | 7 | 77.26% | 124,647,170 | 100.000% | 122.8 min |
| BitFit | 6 | 72.92% | 694,274 | 0.557% | 75.9 min |
| Adapters | 3 | 52.71% | 1,191,554 | 0.951% | 75.5 min |
| LoRA | 6 | 71.48% | 887,042 | 0.707% | 88.7 min |

For this single seed, Full FT achieved the highest validation accuracy. BitFit
was the strongest parameter-efficient method in this configuration: it slightly
outperformed LoRA while training fewer parameters. This is a one-seed learning
experiment on a small validation set, not a statistically reliable ranking of
the methods.

## Published Hugging Face models

| Method | Hugging Face repository | Artifact format |
| --- | --- | --- |
| Full FT | [Anirudh7003/roberta-base-rte-full_ft](https://huggingface.co/Anirudh7003/roberta-base-rte-full_ft) | Standard Transformers checkpoint |
| BitFit | [Anirudh7003/roberta-base-rte-bitfit](https://huggingface.co/Anirudh7003/roberta-base-rte-bitfit) | Standard Transformers checkpoint |
| Adapters | [Anirudh7003/roberta-base-rte-adapter](https://huggingface.co/Anirudh7003/roberta-base-rte-adapter) | Custom adapter checkpoint with `load_adapter.py` |
| LoRA | [Anirudh7003/roberta-base-rte-lora](https://huggingface.co/Anirudh7003/roberta-base-rte-lora) | PEFT LoRA adapter; load it onto `roberta-base` |

## Files

| File | Purpose |
| --- | --- |
| `requirements.txt` | Python libraries needed by the benchmark. |
| `config.json` | Shared experiment settings: model, sequence length, batches, epochs, and PEFT sizes. |
| `methods.py` | Defines which parameters train for Full FT, BitFit, adapters, and LoRA. |
| `train.py` | Downloads/tokenizes RTE, trains one method and seed, evaluates accuracy, and writes a result JSON file. |
| `summarize.py` | Reads result files and reports mean ± standard deviation across seeds. |
| `publish_models.py` | Creates the four Hugging Face model repositories and uploads saved artifacts. |
| `run_all.ps1` | Runs the complete 4-method × 3-seed comparison sequentially. |
| `results/` | Created automatically; contains one JSON record per finished run. |
| `outputs/` | Created automatically; each run saves its best-validation model artifact here for publication. |

## What each method means here

- **Full FT**: every model and classifier parameter updates.
- **BitFit**: every bias term and the task classifier update; all other
  pretrained weights are frozen.
- **Adapters**: a trainable 32-dimensional bottleneck adapter is added after
  each of RoBERTa's 12 encoder layers, while pretrained weights stay frozen.
  The classifier also trains.
- **LoRA**: PEFT adds rank-8 updates to attention query and value projections;
  pretrained weights stay frozen and the classifier trains.

## Fair-comparison rules

Do not change batch size, maximum sequence length, epochs, data split, or seed
between methods. Learning rate is intentionally method-specific: full FT uses
`2e-5`, BitFit/adapters `1e-3`, and LoRA `5e-4`. Record any later tuning in a
copy of `config.json` so the original baseline remains reproducible.

The metric is validation accuracy. RTE is small, so one seed is a smoke test,
not a conclusion: use several seeds and report the mean and standard deviation
printed by `summarize.py` for a more reliable comparison.
