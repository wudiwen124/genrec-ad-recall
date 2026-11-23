from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    training_rows = []
    for path in sorted(args.experiments_root.glob("*/train_result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        config = result["args"]
        best_epoch = result.get("best_epoch")
        epoch_row = next(
            (row for row in result["epochs"] if row["epoch"] == best_epoch),
            None,
        )
        if epoch_row is None:
            continue
        validation = epoch_row["validation"]
        training_rows.append(
            {
                "experiment_id": result.get("experiment_id") or path.parent.name,
                "model_type": config.get("model_type"),
                "loss_type": config.get("loss_type"),
                "negative_mode": config.get("negative_mode"),
                "logq": config.get("logq"),
                "hard_negative": config.get("hard_negative"),
                "curriculum": config.get("curriculum"),
                "use_action": config.get("use_action"),
                "use_time": config.get("use_time"),
                "seed": config.get("seed"),
                "best_epoch": best_epoch,
                "val_hr@10": validation["hr@10"],
                "val_ndcg@10": validation["ndcg@10"],
                "val_score": validation["score"],
                "train_seconds": result["total_seconds"],
            }
        )

    test_rows = []
    for path in sorted(args.logs_root.glob("eval_*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result["metrics"]
        test_rows.append(
            {
                "file": path.name,
                "checkpoint": result["checkpoint"],
                "split": result["split"],
                "filter_history": result["filter_history"],
                "users": metrics["users"],
                "hr@10": metrics["hr@10"],
                "ndcg@10": metrics["ndcg@10"],
                "score": metrics["score"],
            }
        )

    write_csv(args.output_dir / "training_results.csv", training_rows)
    write_csv(args.output_dir / "test_results.csv", test_rows)
    print(f"Wrote {len(training_rows)} training rows and {len(test_rows)} test rows")


if __name__ == "__main__":
    main()
