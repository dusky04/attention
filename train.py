from typing import Any, Dict, Tuple
from torch.utils.data import Dataset, DataLoader, random_split
from transformer import Transformer, build_transformer

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


def get_ds(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
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

    src_max_length: int = 0
    target_max_length: int = 0
    for item in ds_raw:
        src_ids = src_tokenizer.encode(item["translation"][config["lang_src"]]).ids
        target_ids = target_tokenizer.encode(
            item["translation"][config["lang_tgt"]]
        ).ids
        src_max_length = max(src_max_length, len(src_ids))
        target_max_length = max(target_max_length, len(target_ids))

    print("Max length of source sentence: ", src_max_length)
    print("Max Length of target sentence: ", target_max_length)

    # creating the dataloaders
    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    test_dataloader = DataLoader(
        test_ds, batch_size=1, shuffle=True
    )  # cuz we want to process each sentence one by one

    return train_dataloader, test_dataloader, src_tokenizer, target_tokenizer


def get_model(
    config: Dict[str, Any], vocab_src_len: int, vocab_target_len: int
) -> Transformer:
    return build_transformer(
        vocab_src_len,
        vocab_target_len,
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    )
