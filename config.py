from pathlib import Path
from typing import Any, Dict


def get_config() -> Dict[str, Any]:
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 1e-4,  # usually big learning rate at the start but gradually decrease it with each epoch
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config: Dict[str, Any], epoch: str) -> str:
    model_folder: str = config["model_folder"]
    model_basename: str = config["model_basename"]
    model_filename: str = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
