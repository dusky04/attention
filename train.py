from typing import Any, Dict, Tuple, cast

from tqdm import tqdm

from transformer import Transformer, build_transformer

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

import datasets
from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from data import BilingualDataset

from config import get_config, get_weights_file_path


def get_all_sentences(ds, lang: str):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang: str) -> Tokenizer:
    tokenizer_path: Path = Path(config["tokenizer_file"].format(lang))
    if not tokenizer_path.exists():
        tokenizer: Tokenizer = Tokenizer(WordLevel(vocab={}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer: WordLevelTrainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer: Tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
    ds_raw: datasets.Dataset = cast(
        datasets.Dataset,
        load_dataset(
            "Helsinki-NLP/opus_books",
            f"{config['lang_src']}-{config['lang_tgt']}",
            split="train",
        ),
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


def train_model(config: Dict[str, Any]):
    # get the device to train on
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device: ", device)

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, test_dataloader, src_tokenizer, target_tokenizer = get_ds(config)
    model: Transformer = get_model(
        config, src_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size()
    ).to(device)

    # tensorboard
    writer: SummaryWriter = SummaryWriter(config["experiment_name"])

    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        params=model.parameters(), lr=config["lr"], eps=1e-9
    )

    # for resuming the state of the optimizer
    initial_epoch: int = 0
    global_step: int = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print("Preloading model: ", model_filename)
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_func = nn.CrossEntropyLoss(
        ignore_index=src_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    # training loop
    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch: {epoch:02d}")
        for batch in batch_iterator:
            # TODO: add dimensions here
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            # run through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )
            proj_output = model.project(decoder_output)

            label = batch["label"].to(device)

            loss = loss_func(
                proj_output.view(-1, target_tokenizer.get_vocab_size()), label.view(-1)
            )
            batch_iterator.set_postfix({"Loss": f"{loss.item():6.3f}"})
            #  log it on tensorboard
            writer.add_scalar("Training Loss", loss.item(), global_step)
            writer.flush()

            # backward prop
            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            global_step += 1

        # save the model
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    config: Dict[str, Any] = get_config()

    # finally train the model
    train_model(config)
