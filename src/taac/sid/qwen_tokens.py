from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from taac.data.feature_store import (
    USER_ARRAY_FEATURE_IDS,
    USER_SCALAR_FEATURE_IDS,
    load_indexer,
    user_feature_vocab_sizes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sid-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sid_token(level: int, code: int) -> str:
    return f"<sid_l{level}_{code}>"


def main() -> None:
    args = parse_args()
    sid = np.load(args.sid_path)
    codebook_sizes = sid["item_to_codes"].shape[1]
    max_codes = sid["item_to_codes"][1:].max(axis=0) + 1
    sizes = [int(value) for value in max_codes]
    if len(sizes) != codebook_sizes:
        raise RuntimeError("Unexpected SID codebook shape")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_vocab_size = len(tokenizer)

    tokens_by_level = []
    all_tokens = []
    for level, size in enumerate(sizes, start=1):
        level_tokens = [sid_token(level, code) for code in range(size)]
        tokens_by_level.append(level_tokens)
        all_tokens.extend(level_tokens)

    action_tokens = [f"<action_{value}>" for value in range(5)]
    time_tokens = [f"<time_{bucket}>" for bucket in range(64)]
    scalar_vocab, array_vocab = user_feature_vocab_sizes(load_indexer(args.data_root))
    scalar_tokens = {
        feature_id: [
            f"<user_{feature_id}_{value}>"
            for value in range(size)
        ]
        for feature_id, size in zip(USER_SCALAR_FEATURE_IDS, scalar_vocab)
    }
    array_tokens = {
        feature_id: [
            f"<user_{feature_id}_{value}>"
            for value in range(size)
        ]
        for feature_id, size in zip(USER_ARRAY_FEATURE_IDS, array_vocab)
    }
    context_tokens = [
        *action_tokens,
        *time_tokens,
        *[
            token
            for feature_tokens in scalar_tokens.values()
            for token in feature_tokens
        ],
        *[
            token
            for feature_tokens in array_tokens.values()
            for token in feature_tokens
        ],
    ]
    all_tokens.extend(context_tokens)

    added = tokenizer.add_tokens(all_tokens, special_tokens=False)
    token_ids_by_level = [
        tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_by_level
    ]
    expected = sum(sizes) + len(context_tokens)
    if len(all_tokens) != expected or added != expected:
        raise RuntimeError(
            f"SID token expansion expected {expected}, got tokens={len(all_tokens)}, "
            f"added={added}"
        )
    if len(tokenizer) != base_vocab_size + expected:
        raise RuntimeError("SID token IDs are not contiguous at the vocabulary tail")

    context_token_ids = {
        "action": tokenizer.convert_tokens_to_ids(action_tokens),
        "time": tokenizer.convert_tokens_to_ids(time_tokens),
        "scalar": {
            feature_id: tokenizer.convert_tokens_to_ids(tokens)
            for feature_id, tokens in scalar_tokens.items()
        },
        "array": {
            feature_id: tokenizer.convert_tokens_to_ids(tokens)
            for feature_id, tokens in array_tokens.items()
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)
    metadata = {
        "model_path": str(args.model_path),
        "sid_path": str(args.sid_path),
        "base_vocab_size": base_vocab_size,
        "sid_token_count": sum(sizes),
        "context_token_count": len(context_tokens),
        "total_vocab_size": len(tokenizer),
        "codebook_sizes": sizes,
        "tokens_by_level": tokens_by_level,
        "token_ids_by_level": token_ids_by_level,
        "context_token_ids": context_token_ids,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_prompt_tokens": 794,
    }
    (args.output_dir / "sid_tokens.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: metadata[key] for key in (
        "base_vocab_size",
        "sid_token_count",
        "context_token_count",
        "total_vocab_size",
        "codebook_sizes",
        "pad_token_id",
        "eos_token_id",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
