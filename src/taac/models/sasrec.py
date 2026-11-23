from __future__ import annotations

import torch
import torch.nn.functional as F


class HSTULayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        relative_bias_buckets: int = 32,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm1 = torch.nn.RMSNorm(hidden_size, eps=1e-8)
        self.uvqk = torch.nn.Linear(hidden_size, hidden_size * 6, bias=False)
        self.out = torch.nn.Linear(hidden_size * 3, hidden_size)
        self.norm2 = torch.nn.RMSNorm(hidden_size, eps=1e-8)
        self.dropout = torch.nn.Dropout(dropout)
        self.relative_bias = torch.nn.Embedding(relative_bias_buckets, num_heads)

    def forward(
        self,
        hidden: torch.Tensor,
        valid_mask: torch.Tensor,
        timestamps: torch.Tensor,
        use_relative_bias: bool,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden.shape
        normalized = self.norm1(hidden)
        u, v, q, k, ts_gate, legacy = self.uvqk(normalized).chunk(6, dim=-1)

        def split_heads(value: torch.Tensor) -> torch.Tensor:
            return value.view(
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        q_heads = split_heads(q)
        k_heads = split_heads(k)
        v_heads = split_heads(v)
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if use_relative_bias:
            minutes = (
                timestamps.unsqueeze(2) - timestamps.unsqueeze(1)
            ).abs().float() / 60.0
            bucket = torch.log1p(minutes).long().clamp_(
                0,
                self.relative_bias.num_embeddings - 1,
            )
            scores = scores + self.relative_bias(bucket).permute(0, 3, 1, 2)

        attention = F.silu(scores) / seq_len
        allowed = causal_mask.unsqueeze(0).unsqueeze(0) & valid_mask.unsqueeze(1).unsqueeze(2)
        attention = attention.masked_fill(~allowed, 0.0)
        attended = torch.einsum("bhnm,bhmd->bnhd", attention, v_heads)
        attended = attended.reshape(batch_size, seq_len, self.hidden_size)
        gated = self.norm2(attended) * u
        combined = torch.cat([gated, ts_gate, legacy], dim=-1)
        return hidden + self.dropout(self.out(combined))


class SASRecRetriever(torch.nn.Module):
    def __init__(
        self,
        *,
        user_count: int,
        item_count: int,
        maxlen: int = 101,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_action_types: int = 5,
        num_time_buckets: int = 64,
        item_feature_vocab_sizes: list[int] | None = None,
        item_feature_embedding_size: int = 16,
        user_scalar_vocab_sizes: list[int] | None = None,
        user_array_vocab_sizes: list[int] | None = None,
        user_feature_embedding_size: int = 16,
        user_array_max_lengths: list[int] | None = None,
        model_type: str = "sasrec",
        use_action: bool = True,
        use_time: bool = True,
        use_relative_bias: bool = True,
        norm_first: bool = True,
    ):
        super().__init__()
        self.maxlen = maxlen
        self.item_emb = torch.nn.Embedding(item_count + 1, hidden_size, padding_idx=0, sparse=True)
        self.user_emb = torch.nn.Embedding(user_count + 1, hidden_size, padding_idx=0, sparse=True)
        self.action_emb = torch.nn.Embedding(num_action_types, hidden_size, padding_idx=4)
        self.time_emb = torch.nn.Embedding(num_time_buckets, hidden_size, padding_idx=0)
        self.position_emb = torch.nn.Embedding(maxlen, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.model_type = model_type
        self.use_action = use_action
        self.use_time = use_time
        self.use_relative_bias = use_relative_bias

        if model_type == "sasrec":
            layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=norm_first,
            )
            self.encoder = torch.nn.TransformerEncoder(
                layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
        elif model_type == "hstu":
            self.encoder = torch.nn.ModuleList(
                [HSTULayer(hidden_size, num_heads, dropout) for _ in range(num_layers)]
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        self.final_norm = torch.nn.LayerNorm(hidden_size)
        self.item_feature_vocab_sizes = item_feature_vocab_sizes or []
        self.item_feature_embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(vocab_size, item_feature_embedding_size, padding_idx=0)
                for vocab_size in self.item_feature_vocab_sizes
            ]
        )
        self.item_feature_projection = (
            torch.nn.Linear(
                len(self.item_feature_vocab_sizes) * item_feature_embedding_size,
                hidden_size,
            )
            if self.item_feature_vocab_sizes
            else None
        )
        self.user_scalar_vocab_sizes = user_scalar_vocab_sizes or []
        self.user_array_vocab_sizes = user_array_vocab_sizes or []
        self.user_array_max_lengths = user_array_max_lengths or []
        self.user_scalar_embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(vocab_size, user_feature_embedding_size, padding_idx=0)
                for vocab_size in self.user_scalar_vocab_sizes
            ]
        )
        self.user_array_embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(vocab_size, user_feature_embedding_size, padding_idx=0)
                for vocab_size in self.user_array_vocab_sizes
            ]
        )
        user_feature_count = len(self.user_scalar_vocab_sizes) + len(self.user_array_vocab_sizes)
        self.user_feature_projection = (
            torch.nn.Linear(user_feature_count * user_feature_embedding_size, hidden_size)
            if user_feature_count
            else None
        )
        self.register_buffer(
            "item_features",
            torch.zeros(
                (item_count + 1, len(self.item_feature_vocab_sizes)),
                dtype=torch.int32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "user_scalar_features",
            torch.zeros(
                (user_count + 1, len(self.user_scalar_vocab_sizes)),
                dtype=torch.int32,
            ),
            persistent=False,
        )
        for index, max_length in enumerate(self.user_array_max_lengths):
            self.register_buffer(
                f"user_array_features_{index}",
                torch.zeros((user_count + 1, max_length), dtype=torch.int32),
                persistent=False,
            )
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(maxlen, maxlen, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

        self.apply(self._init_weights)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
            self.user_emb.weight[0].zero_()

    @staticmethod
    def _init_weights(module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def encode(
        self,
        sequence: torch.Tensor,
        actions: torch.Tensor,
        time_buckets: torch.Tensor,
        user_ids: torch.Tensor,
        timestamps: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = sequence.shape
        positions = torch.arange(seq_len, device=sequence.device).unsqueeze(0)
        hidden = self.item_emb(sequence) + self.position_emb(positions)
        if self.use_action:
            hidden = hidden + self.action_emb(actions)
        if self.use_time:
            hidden = hidden + self.time_emb(time_buckets)

        # User BOS is located immediately before the first history token.
        bos_index = torch.where(
            valid_mask.any(dim=1),
            valid_mask.float().argmax(dim=1),
            seq_len - 1,
        )
        user_vectors = self.user_emb(user_ids) + self.encode_user_features(user_ids)
        hidden[torch.arange(batch_size, device=sequence.device), bos_index] += user_vectors
        hidden = self.dropout(hidden)
        if self.model_type == "sasrec":
            encoded = self.encoder(
                hidden,
                mask=self.causal_mask[:seq_len, :seq_len],
            )
        else:
            encoded = hidden
            for layer in self.encoder:
                encoded = layer(
                    encoded,
                    valid_mask=valid_mask,
                    timestamps=timestamps,
                    use_relative_bias=self.use_time and self.use_relative_bias,
                    causal_mask=self.causal_mask[:seq_len, :seq_len],
                )
        return self.final_norm(encoded[:, -1])

    def score_items(
        self,
        query: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        item_vectors = F.normalize(self.encode_items(item_ids), dim=-1)
        return F.normalize(query, dim=-1) @ item_vectors.transpose(-2, -1)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        vectors = self.item_emb(item_ids)
        if not self.item_feature_vocab_sizes:
            return vectors
        feature_values = self.item_features[item_ids].long()
        feature_vectors = torch.cat(
            [
                embedding(feature_values[..., index])
                for index, embedding in enumerate(self.item_feature_embeddings)
            ],
            dim=-1,
        )
        return vectors + self.item_feature_projection(feature_vectors)

    def encode_user_features(self, user_ids: torch.Tensor) -> torch.Tensor:
        if not (self.user_scalar_vocab_sizes or self.user_array_vocab_sizes):
            return torch.zeros_like(self.user_emb(user_ids))
        vectors = []
        scalar_values = self.user_scalar_features[user_ids].long()
        for index, embedding in enumerate(self.user_scalar_embeddings):
            vectors.append(embedding(scalar_values[..., index]))
        for index, embedding in enumerate(self.user_array_embeddings):
            array_values = getattr(self, f"user_array_features_{index}")[user_ids].long()
            embedded = embedding(array_values)
            mask = (array_values != 0).unsqueeze(-1)
            summed = (embedded * mask).sum(dim=-2)
            counts = mask.sum(dim=-2).clamp_min(1)
            vectors.append(summed / counts)
        return self.user_feature_projection(torch.cat(vectors, dim=-1))
