from tokenizers import Tokenizer
import torch
from torch.utils.data import Dataset
import datasets

# TODO: Understand CASUAL MASK


class BilingualDataset(Dataset):
    def __init__(
        self,
        dataset,
        src_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        src_lang: str,
        target_lang: str,
        sequence_len: int,
    ) -> None:
        super().__init__()

        self.dataset: datasets.Dataset = dataset
        self.src_tokenizer: Tokenizer = src_tokenizer
        self.target_tokenizer: Tokenizer = target_tokenizer
        self.src_lang: str = src_lang
        self.target_lang: str = target_lang
        self.sequence_len: int = sequence_len

        # sos - start of sentence
        self.sos_token: torch.Tensor = torch.tensor(
            [src_tokenizer.token_to_id("SOS")], dtype=torch.int64
        )
        # eos - end of sentence
        self.eos_token: torch.Tensor = torch.tensor(
            [src_tokenizer.token_to_id("EOS")], dtype=torch.int64
        )
        # pad - padding
        self.pad_token: torch.Tensor = torch.tensor(
            [src_tokenizer.token_to_id("PAD")], dtype=torch.int64
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        src_target_pair = self.dataset[idx]
        src_text: str = src_target_pair["translation"][self.src_lang]
        target_text: str = src_target_pair["translation"][self.target_lang]

        # gives us the input ids of the tokens in the vocabulary
        enc_input_tokens = self.src_tokenizer.encode(src_text).ids
        dec_input_tokens = self.target_tokenizer.encode(target_text).ids

        # -2 for the [SOS] and [EOS] tokens
        enc_num_padding_tokens: int = self.sequence_len - len(enc_input_tokens) - 2
        # -1 since we only add the [SOS] token on the decoder size
        dec_num_padding_tokens: int = self.sequence_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long!!!")

        encoder_input: torch.Tensor = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        decoder_input: torch.Tensor = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        # add [EOS] to the label (what we expect as output from the decoder)
        label: torch.Tensor = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        # check if everything reaches the sequence length
        assert encoder_input.size(0) == self.sequence_len
        assert decoder_input.size(0) == self.sequence_len
        assert label.size(0) == self.sequence_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # (1, 1, sequence_len)
            "decoder_mask": (decoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            & casual_mask(
                decoder_input.size(0)
            ),  # (1, sequence_len) -> (1, sequence_len, sequence_len)
            "src_text": src_text,
            "tgt_text": target_text,
            "label": label,  # (sequence_len)
        }


def casual_mask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
