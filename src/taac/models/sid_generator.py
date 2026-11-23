from __future__ import annotations

import torch
import torch.nn.functional as F


class SIDGenerator(torch.nn.Module):
    def __init__(
        self,
        *,
        user_count: int,
        item_count: int,
        codebook_sizes: list[int],
        maxlen: int = 101,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        code_embedding_size: int = 16,
        user_scalar_vocab_sizes: list[int] | None = None,
        user_array_vocab_sizes: list[int] | None = None,
        user_array_max_lengths: list[int] | None = None,
        user_feature_embedding_size: int = 16,
    ):
        super().__init__()
        self.codebook_sizes = codebook_sizes
        self.num_sid_levels = len(codebook_sizes)
        self.maxlen = maxlen
        self.hidden_size = hidden_size

        self.user_emb = torch.nn.Embedding(user_count + 1, hidden_size, padding_idx=0, sparse=True)
        self.action_emb = torch.nn.Embedding(5, hidden_size, padding_idx=4)
        self.time_emb = torch.nn.Embedding(64, hidden_size, padding_idx=0)
        self.position_emb = torch.nn.Embedding(maxlen, hidden_size)
        self.sid_embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(size, code_embedding_size, padding_idx=0)
                for size in codebook_sizes
            ]
        )
        self.sid_projection = torch.nn.Linear(
            self.num_sid_levels * code_embedding_size,
            hidden_size,
        )

        self.user_scalar_vocab_sizes = user_scalar_vocab_sizes or []
        self.user_array_vocab_sizes = user_array_vocab_sizes or []
        self.user_array_max_lengths = user_array_max_lengths or []
        self.user_scalar_embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(size, user_feature_embedding_size, padding_idx=0)
                for size in self.user_scalar_vocab_sizes
            ]
        )
        self.user_array_embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(size, user_feature_embedding_size, padding_idx=0)
                for size in self.user_array_vocab_sizes
            ]
        )
        user_feature_count = len(self.user_scalar_vocab_sizes) + len(self.user_array_vocab_sizes)
        self.user_feature_projection = (
            torch.nn.Linear(
                user_feature_count * user_feature_embedding_size,
                hidden_size,
            )
            if user_feature_count
            else None
        )

        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = torch.nn.LayerNorm(hidden_size)
        self.level_position = torch.nn.Embedding(self.num_sid_levels, hidden_size)
        self.level_transforms = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_size + code_embedding_size, hidden_size),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(max(0, self.num_sid_levels - 1))
            ]
        )
        self.output_heads = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, size) for size in codebook_sizes]
        )

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(maxlen, maxlen, dtype=torch.bool), diagonal=1),
            persistent=False,
        )
        self.register_buffer(
            "item_to_codes",
            torch.zeros((item_count + 1, self.num_sid_levels), dtype=torch.int16),
            persistent=False,
        )
        self.register_buffer(
            "user_scalar_features",
            torch.zeros((user_count + 1, len(self.user_scalar_vocab_sizes)), dtype=torch.int32),
            persistent=False,
        )
        for index, max_length in enumerate(self.user_array_max_lengths):
            self.register_buffer(
                f"user_array_features_{index}",
                torch.zeros((user_count + 1, max_length), dtype=torch.int32),
                persistent=False,
            )

    def encode(
        self,
        sequence: torch.Tensor,
        actions: torch.Tensor,
        time_buckets: torch.Tensor,
        user_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        codes = self.item_to_codes[sequence].long()
        sid_vectors = torch.cat(
            [
                embedding(codes[..., level])
                for level, embedding in enumerate(self.sid_embeddings)
            ],
            dim=-1,
        )
        hidden = self.sid_projection(sid_vectors)
        hidden = (
            hidden
            + self.action_emb(actions)
            + self.time_emb(time_buckets)
            + self.position_emb(torch.arange(sequence.shape[1], device=sequence.device).unsqueeze(0))
        )
        bos_index = torch.where(
            valid_mask.any(dim=1),
            valid_mask.float().argmax(dim=1),
            sequence.shape[1] - 1,
        )
        hidden[torch.arange(sequence.shape[0], device=sequence.device), bos_index] += self.encode_user(
            user_ids
        )
        encoded = self.encoder(
            hidden,
            mask=self.causal_mask[: sequence.shape[1], : sequence.shape[1]],
        )
        return self.final_norm(encoded[:, -1])

    def encode_user(self, user_ids: torch.Tensor) -> torch.Tensor:
        vectors = [self.user_emb(user_ids)]
        if self.user_feature_projection is None:
            return self.user_emb(user_ids)
        scalar_values = self.user_scalar_features[user_ids].long()
        vectors = [self.user_emb(user_ids)]
        for index, embedding in enumerate(self.user_scalar_embeddings):
            vectors.append(embedding(scalar_values[..., index]))
        feature_vectors = []
        for index, embedding in enumerate(self.user_array_embeddings):
            values = getattr(self, f"user_array_features_{index}")[user_ids].long()
            embedded = embedding(values)
            mask = (values != 0).unsqueeze(-1)
            counts = mask.sum(dim=-2).clamp_min(1)
            feature_vectors.append((embedded * mask).sum(dim=-2) / counts)
        return vectors[0] + self.user_feature_projection(torch.cat(vectors[1:] + feature_vectors, dim=-1))

    def predict_logits(
        self,
        state: torch.Tensor,
        teacher_codes: torch.Tensor,
    ) -> list[torch.Tensor]:
        logits = []
        hidden = state
        for level in range(self.num_sid_levels):
            level_hidden = hidden + self.level_position.weight[level]
            logits.append(self.output_heads[level](level_hidden))
            if level < self.num_sid_levels - 1:
                code_embedding = self.sid_embeddings[level](teacher_codes[:, level].long())
                hidden = hidden + self.level_transforms[level](
                    torch.cat([hidden, code_embedding], dim=-1)
                )
        return logits
