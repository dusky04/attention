from tokenizers import Tokenizer
import torch
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from transformer import Transformer


def greedy_decode(
    model: Transformer,
    source,
    src_mask,
    src_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    max_len: int,
    device: torch.device,
):
    sos_idx = target_tokenizer.token_to_id("[SOS]")
    eos_idx = target_tokenizer.token_to_id("[EOS]")

    # precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output: torch.Tensor = model.encode(source, src_mask)
    # initialize the decoder input with the [sos] token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for the target (decoder input)
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(src_mask).to(device)

        # calculate output of the decoder
        out = model.decode(encoder_output, src_mask, decoder_input, decoder_mask)

        # get the next token
        prob = model.project(out[:, -1])
        # select the token with the highest probability (since its a greedy search)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ]
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(
    model: Transformer,
    test_ds,
    src_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    max_len: int,
    device: torch.device,
    print_msg,
    global_step,
    writer: SummaryWriter,
    num_examples: int = 2,
):
    model.eval()
    count: int = 0

    source_texts = []
    expected = []
    predicted = []

    # size of the console window
    console_width: int = 80

    with torch.inference_mode():
        for batch in test_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_output = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                src_tokenizer,
                target_tokenizer,
                max_len,
                device,
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = target_tokenizer.encode(
                model_output.detach().cpu().numpy()
            )

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # to the console
            print_msg("-" * console_width)
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDICTED: {model_out_text}")

            if count == num_examples:
                break

    if writer:
        # evaluate the character error rate
        # compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar("validation cer", cer, global_step)
        writer.flush()

        # compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar("validation wer", wer, global_step)
        writer.flush()

        # compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar("validation BLEU", bleu, global_step)
        writer.flush()
