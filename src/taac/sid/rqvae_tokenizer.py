from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

from taac.data.candidate_map import load_candidate_mapping
from taac.data.feature_store import load_indexer
from taac.evaluation.group_analysis import load_model
from taac.sid.tokenizer import build_collision_suffix, encode_all_items


class ResidualVQ(torch.nn.Module):
    def __init__(self, levels: int, codes: int, dimension: int):
        super().__init__()
        # Identity initialization keeps the K-Means initialization and the
        # encoder output in the same space. The linear maps may then adapt
        # smoothly instead of immediately collapsing the codebooks.
        self.encoder = torch.nn.Linear(dimension, dimension)
        self.decoder = torch.nn.Linear(dimension, dimension)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(dimension))
            self.encoder.bias.zero_()
            self.decoder.weight.copy_(torch.eye(dimension))
            self.decoder.bias.zero_()
        self.codebooks = torch.nn.ModuleList(
            [torch.nn.Embedding(codes, dimension) for _ in range(levels)]
        )

    def quantize(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = encoded
        quantized = torch.zeros_like(encoded)
        all_codes = []
        commitment = torch.zeros((), device=encoded.device)
        for codebook in self.codebooks:
            distance = (
                residual.square().sum(dim=-1, keepdim=True)
                - 2 * residual @ codebook.weight.T
                + codebook.weight.square().sum(dim=-1).unsqueeze(0)
            )
            codes = distance.argmin(dim=-1)
            vectors = codebook(codes)
            straight_through = residual + (vectors - residual).detach()
            quantized = quantized + straight_through
            commitment = commitment + F.mse_loss(residual.detach(), vectors)
            residual = residual - vectors.detach()
            all_codes.append(codes)
        return quantized, torch.stack(all_codes, dim=1), commitment

    def forward(self, inputs: torch.Tensor):
        encoded = self.encoder(inputs)
        quantized, codes, commitment = self.quantize(encoded)
        reconstruction = self.decoder(quantized)
        return reconstruction, codes, commitment

    @torch.no_grad()
    def encode_codes(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(inputs)
        _, codes, _ = self.quantize(encoded)
        return codes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vector-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--codes", type=int, default=256)
    parser.add_argument("--fit-samples", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--commitment-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    indexer = load_indexer(args.data_root)
    item_count = max(int(value) for value in indexer["i"].values())
    model_args = type(
        "A",
        (),
        {
            "data_root": args.data_root,
            "checkpoint": args.vector_checkpoint,
            "feature_cache_dir": args.feature_cache_dir,
        },
    )()
    vector_model, _ = load_model(model_args, device)

    all_embeddings = encode_all_items(vector_model, item_count, device)
    mean = all_embeddings.mean(axis=0, dtype=np.float64).astype(np.float32)
    rms = np.sqrt(np.mean((all_embeddings - mean) ** 2)).astype(np.float32)
    scale = max(float(rms), 1e-6)
    normalized = (all_embeddings - mean) / scale

    rng = np.random.default_rng(args.seed)
    fit_indices = rng.choice(
        len(normalized),
        size=min(args.fit_samples, len(normalized)),
        replace=False,
    )
    fit_data = torch.from_numpy(normalized[fit_indices]).to(device)
    dimension = fit_data.shape[1]
    rqvae = ResidualVQ(args.levels, args.codes, dimension).to(device)

    # K-Means initialization prevents an immediate random-codebook collapse.
    residual = fit_data.cpu().numpy()
    for codebook in rqvae.codebooks:
        kmeans = MiniBatchKMeans(
            n_clusters=args.codes,
            batch_size=8192,
            n_init=3,
            max_iter=100,
            random_state=args.seed,
        )
        kmeans.fit(residual)
        centers = kmeans.cluster_centers_.astype(np.float32)
        codebook.weight.data.copy_(torch.from_numpy(centers).to(device))
        labels = kmeans.predict(residual)
        residual = residual - centers[labels]

    optimizer = torch.optim.AdamW(
        rqvae.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(len(fit_data), device=device)
        total_loss = 0.0
        total_steps = 0
        for start in range(0, len(fit_data), args.batch_size):
            batch = fit_data[permutation[start : start + args.batch_size]]
            reconstruction, _, commitment = rqvae(batch)
            reconstruction_loss = F.mse_loss(reconstruction, batch)
            loss = reconstruction_loss + args.commitment_weight * commitment
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_steps += 1
        epoch_loss = total_loss / max(total_steps, 1)
        history.append(epoch_loss)
        print(f"epoch={epoch} loss={epoch_loss:.8f}", flush=True)

    normalized_tensor = torch.from_numpy(normalized).to(device)
    codes = torch.empty((len(normalized), args.levels), dtype=torch.int32)
    reconstruction_error = 0.0
    reconstruction_error_raw = 0.0
    with torch.no_grad():
        for start in range(0, len(normalized_tensor), 8192):
            batch = normalized_tensor[start : start + 8192]
            encoded = rqvae.encoder(batch)
            quantized, batch_codes, _ = rqvae.quantize(encoded)
            reconstruction = rqvae.decoder(quantized)
            reconstruction_error += float(
                F.mse_loss(reconstruction, batch, reduction="sum").item()
            )
            reconstruction_raw = reconstruction * scale + torch.from_numpy(mean).to(device)
            batch_raw = batch * scale + torch.from_numpy(mean).to(device)
            reconstruction_error_raw += float(
                F.mse_loss(reconstruction_raw, batch_raw, reduction="sum").item()
            )
            codes[start : start + len(batch)] = batch_codes.cpu()

    suffix, collision_stats = build_collision_suffix(codes.numpy(), args.codes)
    all_codes = np.concatenate(
        [codes.numpy(), suffix[:, None].astype(np.int32)],
        axis=1,
    )
    item_to_codes = np.zeros((item_count + 1, all_codes.shape[1]), dtype=np.int32)
    item_to_codes[1:] = all_codes
    mapping = load_candidate_mapping(args.data_root)
    candidate_rids = mapping["mapped_rids"].astype(np.int64, copy=False)
    candidate_codes = item_to_codes[candidate_rids]
    reconstruction_mse = reconstruction_error / np.prod(normalized.shape)
    reconstruction_mse_raw = reconstruction_error_raw / np.prod(normalized.shape)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        item_to_codes=item_to_codes,
        candidate_rids=candidate_rids,
        candidate_codes=candidate_codes,
        **{
            f"codebook_{index}": codebook.weight.detach().cpu().numpy()
            for index, codebook in enumerate(rqvae.codebooks)
        },
    )
    metadata = {
        "tokenizer": "rqvae",
        "levels": args.levels,
        "codes_per_level": args.codes,
        "codebook_sizes": [
            *[args.codes] * args.levels,
            collision_stats["suffix_vocab_size"],
        ],
        "item_count": item_count,
        "candidate_items": int(len(candidate_rids)),
        "fit_samples": int(len(fit_indices)),
        "reconstruction_mse": reconstruction_mse,
        "reconstruction_mse_raw": reconstruction_mse_raw,
        "normalization_mean_l2": float(np.linalg.norm(mean)),
        "normalization_rms": float(scale),
        "training_history": history,
        **collision_stats,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
