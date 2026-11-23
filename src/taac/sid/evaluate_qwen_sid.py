from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from taac.data.candidate_map import load_candidate_mapping
from taac.data.click_dataset import time_gap_buckets
from taac.data.feature_store import (
    USER_ARRAY_FEATURE_IDS,
    USER_SCALAR_FEATURE_IDS,
    build_user_feature_store,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--warmup-checkpoint", type=Path)
    parser.add_argument("--sid-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--debug-users", type=int, default=0)
    parser.add_argument("--logit-adjustment-tau", type=float, default=0.0)
    return parser.parse_args()


def build_candidate_children(candidate_codes: np.ndarray) -> list[dict]:
    children: list[dict[tuple[int, ...], list[int]]] = [
        {} for _ in range(candidate_codes.shape[1])
    ]
    for code_tuple in candidate_codes:
        prefix: tuple[int, ...] = ()
        for level, code in enumerate(code_tuple.tolist()):
            children[level].setdefault(prefix, [])
            if code not in children[level][prefix]:
                children[level][prefix].append(code)
            prefix = prefix + (int(code),)
    for level in range(len(children)):
        for prefix in children[level]:
            children[level][prefix].sort()
    return children


def sampled_indices(length: int, max_samples: int) -> np.ndarray:
    if max_samples and max_samples < length:
        return np.linspace(0, length - 1, num=max_samples, dtype=np.int64)
    return np.arange(length, dtype=np.int64)


class SidPriorLogitsProcessor:
    def __init__(
        self,
        penalties: list[torch.Tensor],
        prompt_length: int,
    ):
        self.penalties = penalties
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        generated_count = int(input_ids.shape[1] - self.prompt_length)
        if 0 <= generated_count < len(self.penalties):
            scores = scores - self.penalties[generated_count].to(
                device=scores.device,
                dtype=scores.dtype,
            )
        return scores


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    device = torch.device("cuda")
    metadata = json.loads(args.metadata_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    base_model.resize_token_embeddings(metadata["total_vocab_size"])
    if args.warmup_checkpoint is not None:
        head_state = torch.load(
            args.warmup_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        base_model.get_input_embeddings().weight.data.copy_(
            head_state["embed_tokens"]
        )
        base_model.get_output_embeddings().weight.data.copy_(
            head_state["lm_head"]
        )
    model = PeftModel.from_pretrained(base_model, args.adapter_path).to(device)
    model.eval()

    sid = np.load(args.sid_path)
    item_to_codes = sid["item_to_codes"]
    candidate_rids = sid["candidate_rids"].astype(np.int64)
    candidate_codes = sid["candidate_codes"].astype(np.int32)
    children = build_candidate_children(candidate_codes)
    code_to_rid = {
        tuple(int(code) for code in codes): int(rid)
        for codes, rid in zip(candidate_codes, candidate_rids)
    }
    token_ids = [
        np.asarray(ids, dtype=np.int64)
        for ids in metadata["token_ids_by_level"]
    ]
    token_to_code = []
    for level_ids in token_ids:
        token_to_code.append(
            {
                int(token_id): code
                for code, token_id in enumerate(level_ids.tolist())
            }
        )
    context = metadata["context_token_ids"]
    action_tokens = np.asarray(context["action"], dtype=np.int64)
    time_tokens = np.asarray(context["time"], dtype=np.int64)
    scalar_tokens = {
        feature_id: np.asarray(ids, dtype=np.int64)
        for feature_id, ids in context["scalar"].items()
    }
    array_tokens = {
        feature_id: np.asarray(ids, dtype=np.int64)
        for feature_id, ids in context["array"].items()
    }
    user_scalars, user_arrays, _ = build_user_feature_store(
        args.data_root,
        args.feature_cache_dir,
    )
    prior_penalties = []
    if args.logit_adjustment_tau > 0:
        train_targets = pq.read_table(
            args.samples_root / "train" / "part-00000.parquet",
            columns=["target_item_id"],
        )["target_item_id"].to_numpy()
        train_codes = item_to_codes[train_targets.astype(np.int64)]
        for level, level_tokens in enumerate(token_ids):
            counts = np.bincount(
                train_codes[:, level].astype(np.int64),
                minlength=len(level_tokens),
            )
            prior = (counts + 1.0) / (counts.sum() + len(level_tokens))
            penalty = torch.zeros(metadata["total_vocab_size"], dtype=torch.float32)
            for code, token_id in enumerate(level_tokens.tolist()):
                penalty[int(token_id)] = float(
                    args.logit_adjustment_tau * np.log(prior[code])
                )
            prior_penalties.append(penalty)

    table = pq.read_table(
        args.samples_root / "test" / "part-00000.parquet",
        columns=[
            "user_id",
            "history_item_ids",
            "history_action_types",
            "history_timestamps",
            "target_item_id",
        ],
    )
    indices = sampled_indices(len(table), args.max_samples)
    prompts: list[list[int]] = []
    histories: list[set[int]] = []
    targets: list[int] = []
    for source_index in indices.tolist():
        user_id = int(table["user_id"][source_index].as_py())
        history = [
            int(item)
            for item in table["history_item_ids"][source_index].as_py()
            if int(item) > 0
        ][-args.max_history :]
        actions = [
            int(value)
            for value in table["history_action_types"][source_index].as_py()
        ][-args.max_history :]
        timestamps = [
            int(value)
            for value in table["history_timestamps"][source_index].as_py()
        ][-args.max_history :]
        prompt = []
        for feature_index, feature_id in enumerate(USER_SCALAR_FEATURE_IDS):
            value = int(user_scalars[user_id][feature_index])
            if value > 0:
                prompt.append(int(scalar_tokens[feature_id][value]))
        for feature_id in USER_ARRAY_FEATURE_IDS:
            for value in user_arrays[feature_id][user_id].tolist():
                value = int(value)
                if value > 0:
                    prompt.append(int(array_tokens[feature_id][value]))
        gaps = time_gap_buckets(timestamps)
        for item, action, gap in zip(history, actions, gaps):
            prompt.append(int(action_tokens[min(action, 4)]))
            prompt.append(int(time_tokens[min(gap, 63)]))
            codes = item_to_codes[item]
            prompt.extend(
                int(token_ids[level][int(code)])
                for level, code in enumerate(codes)
            )
        prompts.append(prompt)
        histories.append(set(history))
        targets.append(int(table["target_item_id"][source_index].as_py()))

    generated_by_user: list[list[list[int]]] = []
    max_prompt_limit = metadata.get("max_prompt_tokens", 760)
    prompts = [prompt[-max_prompt_limit:] for prompt in prompts]
    for chunk_start in range(0, len(prompts), args.batch_size):
        chunk_prompts = prompts[chunk_start : chunk_start + args.batch_size]
        max_prompt_length = max(len(prompt) for prompt in chunk_prompts)
        input_ids = torch.full(
            (len(chunk_prompts), max_prompt_length),
            metadata["pad_token_id"],
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, prompt in enumerate(chunk_prompts):
            start = max_prompt_length - len(prompt)
            input_ids[row_index, start:] = torch.tensor(prompt)
            attention_mask[row_index, start:] = 1

        def prefix_allowed_tokens(
            batch_id: int,
            generated_ids: torch.Tensor,
        ) -> list[int]:
            generated_count = int(generated_ids.shape[0] - max_prompt_length)
            if generated_count <= 0:
                prefix: tuple[int, ...] = ()
            else:
                generated_tail = generated_ids[-generated_count:].tolist()
                prefix_list = []
                for level, token_id in enumerate(generated_tail):
                    code = token_to_code[level].get(int(token_id))
                    if code is None:
                        return []
                    prefix_list.append(code)
                prefix = tuple(prefix_list)
            allowed_codes = children[generated_count].get(prefix, [])
            return [
                int(token_ids[generated_count][code])
                for code in allowed_codes
            ]

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=5,
                num_beams=args.beam_size,
                num_return_sequences=args.beam_size,
                do_sample=False,
                early_stopping=True,
                pad_token_id=metadata["pad_token_id"],
                eos_token_id=metadata["eos_token_id"],
                logits_processor=(
                    [
                        SidPriorLogitsProcessor(
                            prior_penalties,
                            max_prompt_length,
                        )
                    ]
                    if prior_penalties
                    else None
                ),
                prefix_allowed_tokens_fn=prefix_allowed_tokens,
            )
        generated = generated[:, -5:].cpu().tolist()
        for row_index in range(len(chunk_prompts)):
            start = row_index * args.beam_size
            generated_by_user.append(
                generated[start : start + args.beam_size]
            )
        del input_ids, attention_mask, generated
        torch.cuda.empty_cache()

    ranks = []
    valid_lists = 0
    debug_rows = []
    for user_index, (target, history) in enumerate(zip(targets, histories)):
        ranked = []
        seen = set()
        for sid_token_ids in generated_by_user[user_index]:
            codes = []
            valid = True
            for level, token_id in enumerate(sid_token_ids):
                code = token_to_code[level].get(int(token_id))
                if code is None:
                    valid = False
                    break
                codes.append(code)
            if not valid:
                continue
            rid = code_to_rid.get(tuple(codes))
            if rid is None or rid in history or rid in seen:
                continue
            ranked.append(rid)
            seen.add(rid)
            if len(ranked) == args.top_k:
                break
        if ranked:
            valid_lists += 1
        ranks.append(ranked.index(target) + 1 if target in ranked else 1_000_000)
        if len(debug_rows) < args.debug_users:
            debug_rows.append(
                {
                    "target_rid": target,
                    "generated_rids": ranked[:10],
                }
            )

    rank_array = np.asarray(ranks, dtype=np.int64)
    metrics = {"users": len(ranks)}
    for k in (10, 50, 100):
        hit = rank_array <= k
        ndcg = np.where(hit, 1.0 / np.log2(rank_array + 1.0), 0.0)
        metrics[f"hr@{k}"] = float(hit.mean())
        metrics[f"ndcg@{k}"] = float(ndcg.mean())
    result = {
        "metrics": metrics,
        "valid_user_rate": valid_lists / len(ranks) if ranks else 0.0,
        "beam_size": args.beam_size,
        "top_k": args.top_k,
        "logit_adjustment_tau": args.logit_adjustment_tau,
        "max_history": args.max_history,
        "evaluation_seconds": time.perf_counter() - started,
    }
    if debug_rows:
        result["debug_users"] = debug_rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
