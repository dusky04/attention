import math
from typing import Tuple

import torch
from torch import nn

# TODO: Go through all dimension calculation


class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512) -> None:
        super().__init__()

        # size of our word embedding (chosen in the reference paper)
        self.d_model: int = d_model
        self.vocab_size: int = vocab_size

        # used to store word embeddings and retrieve them using indices.
        # like a dictionary
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # some meta detail to divide the input embeddings by sqrt(embedding_size)
        return self.embeddings(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(
        self, sequence_len: int, d_model: int = 512, dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.d_model: int = d_model
        self.sequence_len: int = sequence_len
        # dropout is helpful in avoiding overfitting
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        # creating the positional encoding this way
        # because for even idx we use sin() and for odd idx we use cos()
        positional_encodings: torch.Tensor = torch.zeros(sequence_len, d_model)
        positions: torch.Tensor = torch.arange(
            0, sequence_len, dtype=torch.float
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(1e4) / d_model)
        )
        positional_encodings[:, 0::2] = torch.sin(positions * div_term)
        positional_encodings[:, 1::2] = torch.cos(positions * div_term)

        # dimension: (1, sequence_len, d_model)
        positional_encodings = positional_encodings.unsqueeze(0)

        # used to register a buffer that should not to be considered a model parameter
        self.register_buffer("positional_encodings", positional_encodings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input dims: (batch_size, sequence_len, d_model)
        x = x + (self.positional_encodings[:, : x.shape[1], :]).requires_grad_(False)  # type: ignore
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()

        self.eps: float = eps

        # nn.Parameter() makes them learnable
        self.alpha = nn.Parameter(torch.ones(1))  # muliplicative factor
        self.bias = nn.Parameter(torch.ones(1))  # additive factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mean and std over the d_model(embedding) dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (self.alpha * ((x - mean) / (std + self.eps))) + self.bias


class FeedForwardNetwork(nn.Module):
    def __init__(
        self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.linear_block = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_block(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, d_model: int = 512, dropout: float = 0.1) -> None:
        super().__init__()

        assert d_model % h == 0, f"{d_model=} is not divisible by {h=} heads."

        # number of heads for MHA
        self.h: int = h
        self.d_model: int = d_model

        self.d_k: int = d_model // h
        self.w_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.w_o = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask,
        dropout: nn.Dropout,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k: int = q.shape[-1]

        # dimension: (batch_size, num_heads, sequence_len, sequence_len)
        attention_scores: torch.Tensor = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # dimension: (batch, num_heads, sequence_len, sequence_len) --> (batch, num_heads, sequence_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ v), attention_scores

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: bool
    ) -> torch.Tensor:
        # input dimension: (batch_size, sequence_len, d_model)
        q_prime: torch.Tensor = self.w_q(q)
        k_prime: torch.Tensor = self.w_k(k)
        v_prime: torch.Tensor = self.w_v(v)

        # dividing each q, k, v block into different num_heads
        # dimension: (batch_size, num_heads, sequence_len, d_model // num_heads (d_k))
        # tranposed so that each head sees the entire sentence
        q_prime = q_prime.view(
            q_prime.shape[0], q_prime.shape[1], self.h, self.d_k
        ).transpose(1, 2)
        k_prime = k_prime.view(
            k_prime.shape[0], k_prime.shape[1], self.h, self.d_k
        ).transpose(1, 2)
        v_prime = v_prime.view(
            v_prime.shape[0], v_prime.shape[1], self.h, self.d_k
        ).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(
            q_prime, k_prime, v_prime, mask, self.dropout
        )

        # concatenate the heads
        # dimension:
        # (batch, num_heads, sequence_len, d_k) -->
        # (batch, sequence_len, num_heads, d_k) -->
        # (batch, sequence_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class ProjectionLayer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512) -> None:
        super().__init__()
        self.projection = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, sequence_len, d_model) -> (batch_size, sequence_len, vocab_size)
        return torch.log_softmax(self.projection(x), dim=-1)
