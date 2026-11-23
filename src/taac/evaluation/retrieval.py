from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch


def evaluate_full_candidate(
    *,
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    candidate_rids: np.ndarray,
    rid_to_candidate_index: np.ndarray,
    device: torch.device,
    max_batches: int | None = None,
    filter_history: bool = True,
) -> dict[str, float]:
    model.eval()
    candidate_tensor = torch.from_numpy(candidate_rids).to(device)
    candidate_vectors = torch.nn.functional.normalize(
        model.encode_items(candidate_tensor),
        dim=-1,
    )
    rid_index_tensor = torch.from_numpy(rid_to_candidate_index).to(device)

    hits = 0
    ndcg = 0.0
    total = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            query = model.encode(
                moved["sequence"],
                moved["actions"],
                moved["time_buckets"],
                moved["user_ids"],
                moved["timestamps"],
                moved["valid_mask"],
            )
            scores = torch.nn.functional.normalize(query, dim=-1) @ candidate_vectors.T

            if filter_history:
                history_indices = rid_index_tensor[moved["sequence"]]
                history_valid = history_indices >= 0
                rows = (
                    torch.arange(scores.shape[0], device=device)
                    .unsqueeze(1)
                    .expand_as(history_indices)
                )
                scores[rows[history_valid], history_indices[history_valid]] = -torch.inf

            target_indices = rid_index_tensor[moved["targets"]]
            valid_mask = target_indices >= 0
            if not bool(valid_mask.any()):
                continue

            target_tensor = target_indices[valid_mask]
            valid_rows_tensor = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
            row_scores = scores[valid_rows_tensor]
            target_scores = row_scores[
                torch.arange(target_tensor.numel(), device=device),
                target_tensor,
            ]
            ranks = (row_scores > target_scores.unsqueeze(1)).sum(dim=1) + 1
            hits += int((ranks <= 10).sum().item())
            ndcg += float(
                (
                    (ranks <= 10).float()
                    / torch.log2(ranks.float() + 1.0)
                ).sum().item()
            )
            total += int(valid_mask.sum().item())

    return {
        "users": total,
        "hr@10": hits / total if total else 0.0,
        "ndcg@10": ndcg / total if total else 0.0,
        "score": (0.31 * hits + 0.69 * ndcg) / total if total else 0.0,
    }
