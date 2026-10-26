"""Registry of the published ORCA models and evaluation files, and how to locate them.

Models and data are downloaded from the Hugging Face Hub (cached by huggingface_hub) or taken from
local paths. The clamping thresholds of the published models are fixed values from the paper
experiments. This module does not import torch.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

MODELS = {
    "gemma-4b": {"repo": "BUT-FIT/orca-gemma-3-4b-it-multinomial", "var_threshold": 0.05},
    "llama-3b": {"repo": "BUT-FIT/orca-llama-3.2-3b-it-multinomial", "var_threshold": 0.05},
    "olmo-1b": {"repo": "BUT-FIT/orca-olmo-2-1b-multinomial", "var_threshold": 0.005},
}
DATASET_REPO = "BUT-FIT/orca-audio-qa-annotations"
DATASETS = {
    "mmau-pro": "s3-mmau-pro-human-judge-ratings.jsonl",
    "mmau-mmar": "s3-mmau-mmar-human-judge-ratings.jsonl",
}


@dataclass(frozen=True)
class ModelSpec:
    """Location and settings of an ORCA model.

    Attributes:
        model_path: Directory with config.yaml and lm/ (the "model" folder of a checkpoint).
        tokenizer_path: Tokenizer directory.
        score_type: Score type of the model (for example "multinomial").
        var_threshold: Clamping threshold of a published model, or None.
    """

    model_path: Path
    tokenizer_path: Path
    score_type: str
    var_threshold: Optional[float]


def import_hf_hub():
    """Import huggingface_hub with a helpful message if it is missing.

    Returns:
        The huggingface_hub module.
    """
    try:
        import huggingface_hub

        return huggingface_hub
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is needed to download from Hugging Face: pip install huggingface_hub"
        ) from exc


def resolve_model(model: str, tokenizer_path: Optional[str] = None) -> ModelSpec:
    """Find the files of a published model or of a local model directory.

    Accepted directories: the folder that contains model/ and tokenizer/ (the Hugging Face
    layout, or a training checkpoint such as .../best), or the model/ folder itself.

    Args:
        model: Name of a published model (a key of MODELS) or a directory path.
        tokenizer_path: Optional tokenizer directory overriding the default location.

    Returns:
        The ModelSpec. Only published models have a clamping threshold.

    Raises:
        FileNotFoundError: If the model name or directory, or the tokenizer, cannot be found.
    """
    threshold = None
    if model in MODELS:
        hf_hub = import_hf_hub()
        root = Path(hf_hub.snapshot_download(repo_id=MODELS[model]["repo"]))
        threshold = MODELS[model]["var_threshold"]
    else:
        root = Path(model.rstrip("/") or "/")
        if not root.is_dir():
            raise FileNotFoundError(
                f"model '{model}' is neither a published model ({', '.join(MODELS)}) nor an "
                "existing directory"
            )

    if (root / "model" / "config.yaml").exists():
        model_path, base = root / "model", root
    elif (root / "config.yaml").exists() and (root / "lm").is_dir():
        model_path, base = root, root.parent
    else:
        raise FileNotFoundError(
            f"{root} is not an ORCA model directory (expected model/config.yaml inside it)"
        )

    tokenizer = Path(tokenizer_path.rstrip("/")) if tokenizer_path else base / "tokenizer"
    if not tokenizer.is_dir():
        raise FileNotFoundError(
            f"tokenizer directory not found: {tokenizer} (use --tokenizer_path)"
        )

    config = yaml.safe_load((model_path / "config.yaml").read_text()) or {}
    return ModelSpec(model_path, tokenizer, config.get("score_type", "multinomial"), threshold)


def resolve_data(data: str) -> Path:
    """Find an input file: a hosted evaluation file or a local JSONL file.

    Args:
        data: Name of a hosted file (a key of DATASETS) or a path to a .jsonl file.

    Returns:
        Path of the local JSONL file.

    Raises:
        FileNotFoundError: If the local file does not exist.
        ValueError: If the file is not a .jsonl file.
    """
    if data in DATASETS:
        hf_hub = import_hf_hub()
        return Path(
            hf_hub.hf_hub_download(
                repo_id=DATASET_REPO, filename=DATASETS[data], repo_type="dataset"
            )
        )
    path = Path(data)
    if not path.is_file():
        raise FileNotFoundError(
            f"input file not found: {data} (hosted files: {', '.join(DATASETS)})"
        )
    if path.suffix != ".jsonl":
        raise ValueError(f"only JSONL input is supported (extension .jsonl), got: {path.name}")
    return path
