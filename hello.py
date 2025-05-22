from config import get_config

from datasets import load_dataset

config = get_config()

ds_raw = load_dataset(
    "Helsinki-NLP/opus_books",
    f"{config['lang_src']}-{config['lang_tgt']}",
    split="train",
)
print(ds_raw)  # Shows DatasetDict
print(ds_raw["train"])  # Shows one example
