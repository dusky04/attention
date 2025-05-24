import torch
from torch import nn

from main import (
    FeedForwardNetwork,
    LayerNormalization,
    MultiHeadAttention,
    ResidualConnection,
)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttention,
        cross_attention_block: MultiHeadAttention,
        feed_forward_block: FeedForwardNetwork,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attention_block: MultiHeadAttention = self_attention_block
        self.cross_attention_block: MultiHeadAttention = cross_attention_block
        self.feed_forward_block: FeedForwardNetwork = feed_forward_block
        self.residual_connection = nn.ModuleList(
            [
                ResidualConnection(dropout),
                ResidualConnection(dropout),
                ResidualConnection(dropout),
            ],
        )

    def forward(
        self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask, target_mask
    ) -> torch.Tensor:
        x = self.residual_connection[0](
            x, lambda x: self.self_attention_block(x, x, x, target_mask)
        )
        x = self.residual_connection[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers: nn.ModuleList = layers
        self.norm: LayerNormalization = LayerNormalization()

    def forward(
        self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask, target_mask
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)
