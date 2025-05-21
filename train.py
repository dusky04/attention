from torch.utils.data import Dataset, DataLoader, random_split

from pathlib import Path

from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from data import BilingualDataset


def get_all_sentences(ds, lang: str):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang: str) -> Tokenizer:
    tokenizer_path: Path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer: Tokenizer = Tokenizer(WordLevel(vocab={}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer: WordLevelTrainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    else:
        tokenizer: Tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset(
        "Helsinki-NLP/opus_books", f"{config['lang_src']}-{config['lang_tgt']}"
    )

    # build the tokenizers
    src_tokenizer: Tokenizer = get_or_build_tokenizer(
        config, ds_raw, config["lang_src"]
    )
    target_tokenizer: Tokenizer = get_or_build_tokenizer(
        config, ds_raw, config["lang_tgt"]
    )

    # 90% training and 10% testing
    train_ds_size: int = int(0.9 * len(ds_raw))
    test_ds_size: int = len(ds_raw) - train_ds_size
    train_ds_raw, test_ds_raw = random_split(ds_raw, [train_ds_size, test_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        src_tokenizer,
        target_tokenizer,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    test_ds = BilingualDataset(
        test_ds_raw,
        src_tokenizer,
        target_tokenizer,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
