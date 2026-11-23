from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def load_candidate_mapping(data_root: Path) -> dict[str, np.ndarray]:
    """Map candidate retrieval positions to historical Item RIDs."""
    with (data_root / "indexer.pkl").open("rb") as handle:
        indexer = pickle.load(handle)

    candidate_file = next((data_root / "candidate").glob("*.parquet"))
    table = pq.read_table(candidate_file, columns=["item_id", "retrieval_id"])
    item_oids = table["item_id"].to_pylist()
    retrieval_ids = np.asarray(table["retrieval_id"].to_pylist(), dtype=np.int64)

    max_pos = int(retrieval_ids.max()) + 1 if len(retrieval_ids) else 0
    retrieval_to_rid = np.zeros(max_pos, dtype=np.int64)
    oid_to_rid = indexer["i"]

    for oid, retrieval_id in zip(item_oids, retrieval_ids):
        retrieval_to_rid[int(retrieval_id)] = int(oid_to_rid.get(oid, 0))

    mapped = retrieval_to_rid[retrieval_to_rid > 0]
    return {
        "retrieval_to_rid": retrieval_to_rid,
        "mapped_rids": mapped.astype(np.int64, copy=False),
    }


def rid_to_candidate_index(retrieval_to_rid: np.ndarray) -> dict[int, int]:
    return {
        int(rid): retrieval_index
        for retrieval_index, rid in enumerate(retrieval_to_rid)
        if rid > 0
    }


def build_rid_to_candidate_index_array(
    retrieval_to_rid: np.ndarray,
    max_rid: int | None = None,
) -> np.ndarray:
    if max_rid is None:
        max_rid = int(retrieval_to_rid.max()) if len(retrieval_to_rid) else 0
    mapping = np.full(max_rid + 1, -1, dtype=np.int64)
    retrieval_positions = np.flatnonzero(retrieval_to_rid > 0)
    mapping[retrieval_to_rid[retrieval_positions]] = retrieval_positions
    return mapping
