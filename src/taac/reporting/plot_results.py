from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 4.8))
    plt.bar(labels, values, color="#4472c4")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    training = {row["experiment_id"]: row for row in read_csv(args.results_root / "training_results.csv")}
    test_rows = {}
    for path in args.logs_root.glob("eval_*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        test_rows[path.name] = data

    model_names = ["B0_official_sasrec", "B1_sasrec_unified", "B2_hstu_unified", "N4_mixed_logq_hard"]
    model_labels = ["B0 SASRec\nBCE", "B1 SASRec\nInfoNCE", "B2 HSTU\nInfoNCE", "FINAL\nN4"]
    model_scores = [float(training[name]["val_score"]) for name in model_names]
    save_bar(
        args.output / "01_model_comparison.png",
        model_labels,
        model_scores,
        "Architecture and Objective Comparison",
        "Validation Score",
    )

    action_time_ids = ["M0_hstu_base", "M1_hstu_action", "M2_hstu_time", "M3_hstu_action_time"]
    action_time_labels = ["Base", "Action", "Time", "Action+Time"]
    save_bar(
        args.output / "02_action_time_ablation.png",
        action_time_labels,
        [float(training[name]["val_score"]) for name in action_time_ids],
        "HSTU Action × Time Ablation",
        "Validation Score",
    )

    negative_ids = [f"N{i}_{suffix}" for i, suffix in [
        (0, "uniform"),
        (1, "inbatch"),
        (2, "mixed"),
        (3, "mixed_logq"),
        (4, "mixed_logq_hard"),
        (5, "full_curriculum"),
    ]]
    save_bar(
        args.output / "03_negative_sampling.png",
        [f"N{i}" for i in range(6)],
        [float(training[name]["val_score"]) for name in negative_ids],
        "Negative Sampling and Training Strategy",
        "Validation Score",
    )

    loo_ids = ["N4_mixed_logq_hard", "L1_no_action", "L2_no_time", "L3_no_mixed", "L4_no_logq", "L5_no_hard"]
    loo_labels = ["FINAL", "-Action", "-Time", "-Mixed", "-LogQ", "-Hard"]
    save_bar(
        args.output / "04_final_leave_one_out.png",
        loo_labels,
        [float(training[name]["val_score"]) for name in loo_ids],
        "FINAL Leave-One-Out Ablation",
        "Validation Score",
    )

    seed_scores = [
        float(training["N4_mixed_logq_hard"]["val_score"]),
        float(training["FINAL_seed2027"]["val_score"]),
        float(training["FINAL_seed2028"]["val_score"]),
    ]
    plt.figure(figsize=(7, 4.5))
    plt.plot([2026, 2027, 2028], seed_scores, marker="o", linewidth=2)
    plt.xticks([2026, 2027, 2028])
    plt.ylabel("Validation Score")
    plt.title("FINAL Multi-Seed Stability")
    plt.tight_layout()
    plt.savefig(args.output / "05_seed_stability.png", dpi=160)
    plt.close()

    group = json.loads((args.logs_root / "group_analysis_final.json").read_text(encoding="utf-8"))
    history = group["history_length_groups"]
    save_bar(
        args.output / "06_history_length_groups.png",
        ["Short", "Medium", "Long"],
        [history[name]["score"] for name in ("short", "medium", "long")],
        "Test Performance by History Length",
        "Score",
    )
    popularity = group["target_popularity_groups"]
    pop_keys = ["unseen_in_train_targets", "low_1_5", "medium_6_100", "head_gt_100"]
    save_bar(
        args.output / "07_target_popularity_groups.png",
        ["Unseen", "1-5", "6-100", ">100"],
        [popularity[name]["score"] for name in pop_keys],
        "Test Performance by Target Frequency",
        "Score",
    )

    hnsw = json.loads((args.logs_root / "faiss_benchmark_final.json").read_text(encoding="utf-8"))
    flat = json.loads((args.logs_root / "faiss_benchmark_flat_final.json").read_text(encoding="utf-8"))
    batches = ["batch_1", "batch_16", "batch_64"]
    x = np.arange(len(batches))
    width = 0.36
    plt.figure(figsize=(8, 4.8))
    plt.bar(x - width / 2, [hnsw["search"][b]["qps"] for b in batches], width, label="HNSW")
    plt.bar(x + width / 2, [flat["search"][b]["qps"] for b in batches], width, label="FlatIP")
    plt.yscale("log")
    plt.xticks(x, ["Batch 1", "Batch 16", "Batch 64"])
    plt.ylabel("Batch QPS (log scale)")
    plt.title("Faiss Throughput")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output / "08_faiss_throughput.png", dpi=160)
    plt.close()

    semantic_path = args.results_root / "semantic_id_results.csv"
    if semantic_path.exists():
        semantic_rows = read_csv(semantic_path)
        labels = [row["label"] for row in semantic_rows]
        metrics = [
            ("hr10", "HR@10"),
            ("ndcg10", "NDCG@10"),
            ("score", "Score"),
        ]
        x = np.arange(len(labels))
        width = 0.25
        plt.figure(figsize=(9, 5))
        for index, (field, label) in enumerate(metrics):
            values = [float(row[field]) for row in semantic_rows]
            plt.bar(x + (index - 1) * width, values, width, label=label)
        plt.xticks(x, labels)
        plt.ylabel("Metric")
        plt.title("Vector Retrieval vs Semantic ID Generation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output / "09_vector_vs_sid.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
