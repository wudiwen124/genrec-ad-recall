from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from taac.data.candidate_map import load_candidate_mapping
from taac.data.click_dataset import ClickSampleDataset, collate_click_samples
from taac.data.feature_store import (
    USER_ARRAY_FEATURE_IDS,
    USER_ARRAY_MAX_LENGTHS,
    build_user_feature_store,
    load_indexer,
    user_feature_vocab_sizes,
)
from taac.evaluation.group_analysis import load_model
from taac.models.sid_generator import SIDGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--vector-checkpoint", type=Path, required=True)
    parser.add_argument("--sid-checkpoint", type=Path, required=True)
    parser.add_argument("--sid-path", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument(
        "--decode-mode",
        choices=("constrained", "unconstrained"),
        default="constrained",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--save-user-results", type=Path)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def load_sid_model(args: argparse.Namespace, device: torch.device) -> SIDGenerator:
    checkpoint = torch.load(args.sid_checkpoint, map_location="cpu", weights_only=False)
    indexer = load_indexer(args.data_root)
    user_count = len(indexer["u"])
    item_count = max(int(value) for value in indexer["i"].values())
    metadata = json.loads(args.sid_path.with_suffix(".json").read_text(encoding="utf-8"))
    user_scalars, user_arrays, _ = build_user_feature_store(
        args.data_root,
        args.feature_cache_dir,
    )
    scalar_vocab, array_vocab = user_feature_vocab_sizes(indexer)
    model = SIDGenerator(
        user_count=user_count,
        item_count=item_count,
        codebook_sizes=metadata["codebook_sizes"],
        maxlen=101,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        user_scalar_vocab_sizes=scalar_vocab,
        user_array_vocab_sizes=array_vocab,
        user_array_max_lengths=[
            USER_ARRAY_MAX_LENGTHS[feature_id] for feature_id in USER_ARRAY_FEATURE_IDS
        ],
    ).to(device)
    sid = np.load(args.sid_path)
    model.item_to_codes.copy_(torch.from_numpy(sid["item_to_codes"]))
    model.user_scalar_features.copy_(torch.from_numpy(user_scalars))
    for index, feature_id in enumerate(USER_ARRAY_FEATURE_IDS):
        getattr(model, f"user_array_features_{index}").copy_(
            torch.from_numpy(user_arrays[feature_id])
        )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def build_candidate_maps(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict, list[dict]]:
    sid = np.load(args.sid_path)
    candidate_rids = sid["candidate_rids"].astype(np.int64)
    candidate_codes = sid["candidate_codes"].astype(np.int64)
    code_to_rid = {
        tuple(code): int(rid)
        for code, rid in zip(candidate_codes, candidate_rids)
    }
    children: list[dict[tuple[int, ...], list[int]]] = [
        {} for _ in range(candidate_codes.shape[1])
    ]
    for code_tuple in candidate_codes:
        prefix: tuple[int, ...] = ()
        for level, code in enumerate(code_tuple.tolist()):
            children[level].setdefault(prefix, [])
            if code not in children[level][prefix]:
                children[level][prefix].append(code)
            prefix = prefix + (code,)
    for level in range(len(children)):
        for prefix in children[level]:
            children[level][prefix].sort()
    return candidate_rids, candidate_codes, code_to_rid, children


def score(ranks: np.ndarray, top_k: int = 10) -> dict[str, float | int]:
    hit = ranks <= top_k
    hr = float(hit.mean()) if len(hit) else 0.0
    ndcg = float(np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0).mean()) if len(hit) else 0.0
    return {
        "users": int(len(ranks)),
        "k": top_k,
        f"hr@{top_k}": hr,
        f"ndcg@{top_k}": ndcg,
        "score": 0.31 * hr + 0.69 * ndcg,
    }


def per_user_metrics(
    ranks: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hit = (ranks <= top_k).astype(np.float64)
    ndcg = np.where(hit > 0, 1.0 / np.log2(ranks + 1.0), 0.0)
    combined = 0.31 * hit + 0.69 * ndcg
    return hit, ndcg, combined


def paired_bootstrap(
    vector_ranks: np.ndarray,
    sid_ranks: np.ndarray,
    samples: int,
    seed: int,
    top_k: int,
) -> dict[str, dict[str, float]]:
    """Bootstrap the paired SID-minus-vector metric deltas over users."""
    vector_hr, vector_ndcg, vector_score = per_user_metrics(vector_ranks, top_k)
    sid_hr, sid_ndcg, sid_score = per_user_metrics(sid_ranks, top_k)
    deltas = {
        f"hr@{top_k}": sid_hr - vector_hr,
        f"ndcg@{top_k}": sid_ndcg - vector_ndcg,
        "score": sid_score - vector_score,
    }
    rng = np.random.default_rng(seed)
    result = {}
    user_count = len(vector_ranks)
    for metric, values in deltas.items():
        bootstrap_means = np.empty(samples, dtype=np.float64)
        for index in range(samples):
            sampled = rng.integers(0, user_count, size=user_count)
            bootstrap_means[index] = values[sampled].mean()
        result[metric] = {
            "mean_delta": float(values.mean()),
            "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
            "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        }
    return result


def complementarity_metrics(
    vector_topk: list[list[int]],
    sid_topk: list[list[int]],
    targets: list[int],
    top_k: int,
) -> dict[str, float | int]:
    user_count = len(targets)
    vector_hits = 0
    sid_hits = 0
    both_hits = 0
    vector_only_hits = 0
    sid_only_hits = 0
    neither_hits = 0
    overlap_sum = 0
    jaccard_sum = 0.0

    for vector_items, sid_items, target in zip(vector_topk, sid_topk, targets):
        vector_hit = target in vector_items
        sid_hit = target in sid_items
        vector_hits += int(vector_hit)
        sid_hits += int(sid_hit)
        both_hits += int(vector_hit and sid_hit)
        vector_only_hits += int(vector_hit and not sid_hit)
        sid_only_hits += int(sid_hit and not vector_hit)
        neither_hits += int(not vector_hit and not sid_hit)

        vector_set = set(vector_items)
        sid_set = set(sid_items)
        overlap = len(vector_set & sid_set)
        union = len(vector_set | sid_set)
        overlap_sum += overlap
        jaccard_sum += overlap / union if union else 0.0

    any_hits = vector_hits + sid_hits - both_hits
    union_hit_rate = any_hits / user_count if user_count else 0.0
    best_single_hit_rate = max(vector_hits, sid_hits) / user_count if user_count else 0.0
    return {
        "users": user_count,
        "vector_hit_users": vector_hits,
        "sid_hit_users": sid_hits,
        "both_hit_users": both_hits,
        "vector_only_hit_users": vector_only_hits,
        "sid_only_hit_users": sid_only_hits,
        "neither_hit_users": neither_hits,
        "union_hit_users": any_hits,
        "vector_hit_rate": vector_hits / user_count if user_count else 0.0,
        "sid_hit_rate": sid_hits / user_count if user_count else 0.0,
        "union_hit_rate": union_hit_rate,
        "union_lift_over_best_single": union_hit_rate - best_single_hit_rate,
        "sid_only_share_of_sid_hits": (
            sid_only_hits / sid_hits if sid_hits else 0.0
        ),
        "mean_overlap_at_k": overlap_sum / user_count if user_count else 0.0,
        "mean_jaccard_at_k": jaccard_sum / user_count if user_count else 0.0,
    }


def union_prefix_metrics(
    vector_topk: list[list[int]],
    sid_topk: list[list[int]],
    targets: list[int],
    prefix_per_route: int,
) -> dict[str, float | int]:
    user_count = len(targets)
    vector_hits = 0
    sid_hits = 0
    union_hits = 0
    vector_only_hits = 0
    sid_only_hits = 0
    union_size_sum = 0

    for vector_items, sid_items, target in zip(vector_topk, sid_topk, targets):
        vector_prefix = set(vector_items[:prefix_per_route])
        sid_prefix = set(sid_items[:prefix_per_route])
        vector_hit = target in vector_prefix
        sid_hit = target in sid_prefix
        union = vector_prefix | sid_prefix

        vector_hits += int(vector_hit)
        sid_hits += int(sid_hit)
        union_hits += int(target in union)
        vector_only_hits += int(vector_hit and not sid_hit)
        sid_only_hits += int(sid_hit and not vector_hit)
        union_size_sum += len(union)

    return {
        "prefix_per_route": prefix_per_route,
        "max_union_size": prefix_per_route * 2,
        "mean_union_size": union_size_sum / user_count if user_count else 0.0,
        "vector_hit_rate": vector_hits / user_count if user_count else 0.0,
        "sid_hit_rate": sid_hits / user_count if user_count else 0.0,
        "union_hit_rate": union_hits / user_count if user_count else 0.0,
        "vector_only_hit_users": vector_only_hits,
        "sid_only_hit_users": sid_only_hits,
        "union_hit_users": union_hits,
    }


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    if args.candidate_k <= 0:
        raise ValueError("--candidate-k must be positive")
    if args.beam_size < args.candidate_k:
        raise ValueError("--beam-size must be greater than or equal to --candidate-k")
    device = torch.device("cuda")
    vector_model, vector_args = load_model(
        type("A", (), {
            "data_root": args.data_root,
            "checkpoint": args.vector_checkpoint,
            "feature_cache_dir": args.feature_cache_dir,
        })(),
        device,
    )
    sid_model = load_sid_model(args, device)
    candidate_rids, _, code_to_rid, children = build_candidate_maps(args)
    candidate_set = set(candidate_rids.tolist())
    candidate_tensor = torch.from_numpy(candidate_rids).to(device)
    candidate_vectors = F.normalize(vector_model.encode_items(candidate_tensor), dim=-1)
    rid_to_position = {int(rid): pos for pos, rid in enumerate(candidate_rids.tolist())}

    dataset = ClickSampleDataset(
        args.samples_root / "test" / "part-00000.parquet",
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(collate_click_samples, maxlen=vector_args["maxlen"]),
    )

    vector_ranks = []
    sid_ranks = []
    sid_valid = 0
    generated_beams = 0
    candidate_mapping_beams = 0
    total = 0
    user_ids: list[int] = []
    target_rids: list[int] = []
    vector_topk_by_user: list[list[int]] = []
    sid_topk_by_user: list[list[int]] = []

    with torch.no_grad():
        for batch in loader:
            moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            history_sets = [
                set(int(item) for item in sequence if int(item) != 0)
                for sequence in moved["sequence"].cpu().tolist()
            ]
            targets = moved["targets"].cpu().numpy()
            user_ids.extend(moved["user_ids"].cpu().tolist())
            target_rids.extend(targets.tolist())

            vector_query = vector_model.encode(
                moved["sequence"],
                moved["actions"],
                moved["time_buckets"],
                moved["user_ids"],
                moved["timestamps"],
                moved["valid_mask"],
            )
            vector_scores = F.normalize(vector_query, dim=-1) @ candidate_vectors.T
            for row_index, history in enumerate(history_sets):
                positions = [rid_to_position[rid] for rid in history if rid in rid_to_position]
                if positions:
                    vector_scores[row_index, positions] = -torch.inf
            target_positions = torch.tensor(
                [rid_to_position[int(target)] for target in targets],
                device=device,
            )
            target_scores = vector_scores[
                torch.arange(len(target_positions), device=device),
                target_positions,
            ]
            ranks = (vector_scores > target_scores.unsqueeze(1)).sum(dim=1) + 1
            vector_ranks.extend(ranks.cpu().tolist())
            vector_top_positions = torch.topk(
                vector_scores,
                k=min(args.candidate_k, vector_scores.shape[1]),
                dim=1,
            ).indices.cpu()
            for row_index in range(batch_size := len(targets)):
                vector_topk_by_user.append(
                    [
                        int(candidate_rids[int(position)])
                        for position in vector_top_positions[row_index]
                    ]
                )

            state = sid_model.encode(
                moved["sequence"],
                moved["actions"],
                moved["time_buckets"],
                moved["user_ids"],
                moved["valid_mask"],
            )
            beam_width = 1
            beam_hidden = state.unsqueeze(1)
            beam_scores = torch.zeros((batch_size, 1), device=device)
            beam_histories: list[list[tuple[int, ...]]] = [[()] for _ in range(batch_size)]

            for level in range(sid_model.num_sid_levels):
                flat_hidden = beam_hidden.reshape(-1, sid_model.hidden_size)
                log_probs = F.log_softmax(
                    sid_model.output_heads[level](
                        flat_hidden + sid_model.level_position.weight[level]
                    ).float(),
                    dim=-1,
                )
                candidate_options: list[list[tuple[float, int, int]]] = [
                    [] for _ in range(batch_size)
                ]
                for owner in range(batch_size):
                    for beam_index, prefix in enumerate(beam_histories[owner]):
                        if args.decode_mode == "constrained":
                            allowed = children[level].get(prefix, [])
                        else:
                            allowed = list(
                                range(sid_model.codebook_sizes[level])
                            )
                        if not allowed:
                            continue
                        flat_index = owner * beam_width + beam_index
                        allowed_tensor = torch.tensor(allowed, device=device, dtype=torch.long)
                        values = (
                            log_probs[flat_index, allowed_tensor]
                            + beam_scores[owner, beam_index]
                        )
                        for value, code in zip(values.tolist(), allowed):
                            candidate_options[owner].append((float(value), int(code), beam_index))

                next_histories: list[list[tuple[int, ...]]] = [
                    [] for _ in range(batch_size)
                ]
                source_indices = []
                selected_codes = []
                selected_scores = []
                next_width = min(
                    args.beam_size,
                    max(1, min(len(options) for options in candidate_options)),
                )
                for owner, options in enumerate(candidate_options):
                    options.sort(reverse=True, key=lambda item: item[0])
                    options = options[:next_width]
                    owner_sources = []
                    owner_codes = []
                    owner_scores = []
                    for value, code, beam_index in options:
                        next_histories[owner].append(
                            beam_histories[owner][beam_index] + (code,)
                        )
                        owner_sources.append(owner * beam_width + beam_index)
                        owner_codes.append(code)
                        owner_scores.append(value)
                    source_indices.append(owner_sources)
                    selected_codes.append(owner_codes)
                    selected_scores.append(owner_scores)

                source_tensor = torch.tensor(source_indices, device=device, dtype=torch.long)
                code_tensor = torch.tensor(selected_codes, device=device, dtype=torch.long)
                selected_hidden = flat_hidden[source_tensor.reshape(-1)].reshape(
                    batch_size, next_width, sid_model.hidden_size
                )
                if level < sid_model.num_sid_levels - 1:
                    code_embedding = sid_model.sid_embeddings[level](code_tensor)
                    selected_hidden = selected_hidden + sid_model.level_transforms[level](
                        torch.cat([selected_hidden, code_embedding], dim=-1)
                    )
                beam_hidden = selected_hidden
                beam_scores = torch.tensor(selected_scores, device=device)
                beam_histories = next_histories
                beam_width = next_width

            for owner, histories in enumerate(beam_histories):
                target_rid = int(targets[owner])
                history = history_sets[owner]
                ranked = []
                seen = set()
                generated_beams += len(histories)
                for prefix in histories:
                    rid = code_to_rid.get(prefix)
                    if rid is not None:
                        candidate_mapping_beams += 1
                    if rid is None or rid in history or rid in seen or rid not in candidate_set:
                        continue
                    ranked.append(rid)
                    seen.add(rid)
                    if len(ranked) == args.candidate_k:
                        break
                if target_rid in ranked:
                    sid_ranks.append(ranked.index(target_rid) + 1)
                else:
                    sid_ranks.append(1_000_000)
                sid_topk_by_user.append(ranked)
                total += 1
                sid_valid += 1 if ranked else 0

    vector_metrics = score(np.asarray(vector_ranks), args.candidate_k)
    sid_metrics = score(np.asarray(sid_ranks), args.candidate_k)
    bootstrap = paired_bootstrap(
        np.asarray(vector_ranks),
        np.asarray(sid_ranks),
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
        top_k=args.candidate_k,
    )
    complementarity = complementarity_metrics(
        vector_topk_by_user,
        sid_topk_by_user,
        target_rids,
        args.candidate_k,
    )
    balanced_prefix = max(1, args.candidate_k // 2)
    balanced_union = union_prefix_metrics(
        vector_topk_by_user,
        sid_topk_by_user,
        target_rids,
        balanced_prefix,
    )
    elapsed_seconds = time.perf_counter() - started_at
    result = {
        "users": total,
        "candidate_items": len(candidate_rids),
        "decode_mode": args.decode_mode,
        "vector_retrieval": vector_metrics,
        "sid_retrieval": sid_metrics,
        "sid_nonempty_user_rate": sid_valid / total if total else 0.0,
        "generated_candidate_mapping_rate": (
            candidate_mapping_beams / generated_beams if generated_beams else 0.0
        ),
        "paired_bootstrap_sid_minus_vector": bootstrap,
        "complementarity": complementarity,
        "balanced_union": balanced_union,
        "candidate_k": args.candidate_k,
        "beam_size": args.beam_size,
        "evaluation_seconds": elapsed_seconds,
        "users_per_second": total / elapsed_seconds if elapsed_seconds else 0.0,
    }
    if args.save_user_results is not None:
        args.save_user_results.parent.mkdir(parents=True, exist_ok=True)
        user_results = [
            {
                "user_id": user_id,
                "target_rid": target_rid,
                "vector_topk": vector_items,
                "sid_topk": sid_items,
            }
            for user_id, target_rid, vector_items, sid_items in zip(
                user_ids,
                target_rids,
                vector_topk_by_user,
                sid_topk_by_user,
            )
        ]
        args.save_user_results.write_text(
            json.dumps(
                {
                    "decode_mode": args.decode_mode,
                    "beam_size": args.beam_size,
                    "summary": complementarity,
                    "users": user_results,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
