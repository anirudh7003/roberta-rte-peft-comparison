"""Create public/private Hugging Face model repositories and upload artifacts.

Run only after all four training runs have produced folders in `outputs/` and
after `hf auth login` has completed on this machine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


METHODS = ("full_ft", "bitfit", "adapter", "lora")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True, help="Your Hugging Face username")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--private", action="store_true", help="Create private repositories")
    args = parser.parse_args()

    api = HfApi()
    for method in METHODS:
        model_name = f"roberta-base-rte-{method}"
        repo_id = f"{args.username}/{model_name}"
        artifact_dir = args.output_dir / f"{model_name}-seed{args.seed}"
        if not artifact_dir.is_dir():
            raise FileNotFoundError(f"Missing artifact: {artifact_dir}. Rerun train.py for {method}.")

        card_path = artifact_dir / "README.md"
        card_path.write_text(
            card_path.read_text(encoding="utf-8").replace("YOUR_REPOSITORY", repo_id),
            encoding="utf-8",
        )
        api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=artifact_dir,
            commit_message=f"Upload seed {args.seed} best-validation {method} checkpoint",
        )
        print(f"Published https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
