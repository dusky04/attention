import torch
from torch import nn

from main import (
    LayerNormalization,
    MultiHeadAttention,
    FeedForwardNetwork,
    ResidualConnection,
)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttention,
        feed_forward_block: FeedForwardNetwork,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attention_block: MultiHeadAttention = self_attention_block
        self.feed_forward_block: FeedForwardNetwork = feed_forward_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout), ResidualConnection(dropout)]
        )

    # src_mask - applied to the input of the encoder
    # to hide the padding word to not interact with the 'actual' words
    def forward(self, x: torch.Tensor, src_mask) -> torch.Tensor:
        x = self.residual_connection[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers: nn.ModuleList = layers
        self.norm: LayerNormalization = LayerNormalization()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
