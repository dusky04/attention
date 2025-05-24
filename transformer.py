from typing import List

import torch
from torch import nn

from decoder import Decoder, DecoderBlock
from encoder import Encoder, EncoderBlock
from main import (
    FeedForwardNetwork,
    InputEmbeddings,
    MultiHeadAttention,
    PositionalEncoding,
    ProjectionLayer,
)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embeddings: InputEmbeddings,
        target_embeddings: InputEmbeddings,
        src_position_encoding: PositionalEncoding,
        target_position_encoding: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embeddings = src_embeddings
        self.target_embeddings = target_embeddings
        self.src_position_encoding = src_position_encoding
        self.target_position_encoding = target_position_encoding
        self.projection_layer = projection_layer

    def encode(self, src, src_mask) -> torch.Tensor:
        src = self.src_embeddings(src)
        src = self.src_position_encoding(src)
        return self.encoder(src, src_mask)

    def decode(
        self, encoder_output: torch.Tensor, src_mask, target, target_mask
    ) -> torch.Tensor:
        target = self.target_embeddings(target)
        target = self.target_position_encoding(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    target_vocab_size: int,
    src_sequence_len: int,
    target_sequence_len: int,
    d_model: int = 512,
    N: int = 6,  # number of encoder and decoder blocks used
    h: int = 8,  # number of heads
    d_ff: int = 2048,
    dropout: float = 0.1,
) -> Transformer:
    # creating the embedding layers
    src_embedding: InputEmbeddings = InputEmbeddings(src_vocab_size, d_model)
    target_embedding: InputEmbeddings = InputEmbeddings(target_vocab_size, d_model)

    # positional encoding layers
    src_position_encoding: PositionalEncoding = PositionalEncoding(
        src_sequence_len, d_model, dropout
    )
    target_position_encoding: PositionalEncoding = PositionalEncoding(
        target_sequence_len, d_model, dropout
    )

    # encoder blocks
    encoder_blocks: List[EncoderBlock] = []
    for _ in range(N):
        encoder_self_attention_block: MultiHeadAttention = MultiHeadAttention(
            h, d_model, dropout
        )
        feed_forward_block: FeedForwardNetwork = FeedForwardNetwork(
            d_model, d_ff, dropout
        )
        encoder_block: EncoderBlock = EncoderBlock(
            encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    # decoder blocks
    decoder_blocks: List[DecoderBlock] = []
    for _ in range(N):
        decoder_self_attention_block: MultiHeadAttention = MultiHeadAttention(
            h, d_model, dropout
        )
        decoder_cross_attention_block: MultiHeadAttention = MultiHeadAttention(
            h, d_model, dropout
        )
        feed_forward_block: FeedForwardNetwork = FeedForwardNetwork(
            d_model, d_ff, dropout
        )
        decoder_block: DecoderBlock = DecoderBlock(
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # creating encoder and decoder
    encoder: Encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder: Decoder = Decoder(nn.ModuleList(decoder_blocks))

    # creating the projection layer
    projection_layer: ProjectionLayer = ProjectionLayer(target_vocab_size, d_model)

    # finally the TRANSFORMER
    transformer: Transformer = Transformer(
        encoder,
        decoder,
        src_embedding,
        target_embedding,
        src_position_encoding,
        target_position_encoding,
        projection_layer,
    )

    # initialize the parameters
    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return transformer
