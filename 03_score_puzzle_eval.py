"""
Compute step-level and puzzle-level accuracy from model predictions.

Expected inputs:
  - steps.parquet (from build_puzzle_eval.py): ground truth.
  - predictions.parquet / .csv / .jsonl: model's predicted move per step.
    Must contain columns: PuzzleId, step_idx, predicted_move (UCI).

Usage:
    python score_puzzle_eval.py \
        --steps ./data/steps.parquet \
        --predictions ./outputs/model_predictions.parquet \
        [--output ./outputs/scores.json]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def load_predictions(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported predictions format: {suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    steps = pd.read_parquet(args.steps)
    preds = load_predictions(args.predictions)

    required = {"PuzzleId", "step_idx", "predicted_move"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(f"Predictions missing required columns: {missing}")

    # Join on (PuzzleId, step_idx)
    merged = steps.merge(
        preds[["PuzzleId", "step_idx", "predicted_move"]],
        on=["PuzzleId", "step_idx"],
        how="left",
    )
    missing_preds = merged["predicted_move"].isna().sum()
    if missing_preds > 0:
        print(f"WARNING: {missing_preds:,} steps have no prediction "
              f"(will count as incorrect)")

    merged["correct"] = merged["predicted_move"] == merged["best_move"]

    # --- Step-level accuracy -------------------------------------------------
    step_acc = merged["correct"].mean()

    # --- Puzzle-level accuracy (AND across steps) ----------------------------
    puzzle_correct = merged.groupby("PuzzleId")["correct"].all()
    puzzle_acc = puzzle_correct.mean()

    # --- Slices --------------------------------------------------------------
    by_bin_step = merged.groupby("rating_bin")["correct"].mean().to_dict()
    puzzle_bins = merged.groupby("PuzzleId")["rating_bin"].first()
    by_bin_puzzle = (
        puzzle_correct.to_frame("solved")
        .join(puzzle_bins)
        .groupby("rating_bin")["solved"]
        .mean()
        .to_dict()
    )

    # By solution length (only makes sense for puzzle-level)
    puzzle_lengths = merged.groupby("PuzzleId")["solution_ply"].first()
    by_length = (
        puzzle_correct.to_frame("solved")
        .join(puzzle_lengths)
        .groupby("solution_ply")["solved"]
        .agg(["mean", "count"])
        .to_dict(orient="index")
    )

    # By theme (step-level, since themes are puzzle-wide this double-counts
    # within a puzzle but is still informative). We flatten themes.
    theme_rows = []
    for _, r in merged.iterrows():
        themes = r["Themes"]
        if themes is None:
            continue
        for t in themes:
            theme_rows.append({"theme": t, "correct": r["correct"]})
    theme_df = pd.DataFrame(theme_rows)
    by_theme_step = (
        theme_df.groupby("theme")["correct"]
        .agg(["mean", "count"])
        .sort_values("count", ascending=False)
        .head(30)
        .to_dict(orient="index")
    )

    # --- Report --------------------------------------------------------------
    report = {
        "step_accuracy": float(step_acc),
        "puzzle_accuracy": float(puzzle_acc),
        "n_steps": int(len(merged)),
        "n_puzzles": int(len(puzzle_correct)),
        "missing_predictions": int(missing_preds),
        "step_accuracy_by_rating_bin": {
            k: float(v) for k, v in by_bin_step.items()
        },
        "puzzle_accuracy_by_rating_bin": {
            k: float(v) for k, v in by_bin_puzzle.items()
        },
        "puzzle_accuracy_by_solution_length": {
            int(k): {"accuracy": float(v["mean"]), "n": int(v["count"])}
            for k, v in by_length.items()
        },
        "step_accuracy_by_theme_top30": {
            k: {"accuracy": float(v["mean"]), "n": int(v["count"])}
            for k, v in by_theme_step.items()
        },
    }

    print(f"Step accuracy:   {step_acc:.4f}  (n={len(merged):,})")
    print(f"Puzzle accuracy: {puzzle_acc:.4f}  (n={len(puzzle_correct):,})")
    print(f"Gap (step - puzzle): {step_acc - puzzle_acc:.4f}")
    print()
    print("By rating bin:")
    for k in sorted(by_bin_step.keys()):
        print(f"  {k}: step={by_bin_step[k]:.3f}  puzzle={by_bin_puzzle.get(k, 0):.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
