"""
Build a stratified eval set from the Lichess puzzle database.

Pipeline:
  1. Load Lichess/chess-puzzles from HuggingFace.
  2. Quality filter (rating deviation, play count, popularity).
  3. Preprocess: apply the opponent's setup move, then walk the solution line,
     capturing (fen, best_move) at every player turn.
  4. Optional dedup vs. training FENs (puzzle-level, using the first step).
  5. Stratified sample by rating bin, at the puzzle level.
  6. Write two parquet files:
       - puzzles.parquet: one row per puzzle, `steps` column holds the per-move
         list. Use this for the full-puzzle AND metric (group by PuzzleId).
       - steps.parquet: one row per player move (fen, best_move). This is what
         you feed the model. Supports step-level accuracy and per-bin /
         per-theme slicing directly.

Two metrics to compute at eval time:
  - Step accuracy: mean(predicted_move == best_move) across steps.parquet.
  - Puzzle accuracy: for each PuzzleId, AND all step-level correctness flags,
    then take the mean. A puzzle counts as solved only if every player move
    in the solution line matches.

Usage:
    python build_puzzle_eval.py \\
        --output-dir ./puzzle_eval \\
        --target-size 25000 \\
        [--train-fens-file data/legal_moves_train_fen_100k.parquet \\
         --train-fen-column metadata.board_fen]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import chess
import pandas as pd
from datasets import load_dataset

# ---- Config ----------------------------------------------------------------

# Rating bin edges. Puzzles outside [min, max] are dropped.
# Bins are [lo, hi); the last bin includes its upper bound.
RATING_BINS = [
    (800, 1200),
    (1200, 1600),
    (1600, 2000),
    (2000, 2400),
    (2400, 3000),
]

# Quality filters. Puzzles failing these are excluded before sampling.
MAX_RATING_DEVIATION = 80   # Glicko-2 RD; lower = more stable rating
MIN_NB_PLAYS = 50           # filters puzzles with unconverged ratings
MIN_POPULARITY = 80         # in [-100, 100]; filters disputed / bad puzzles

RANDOM_SEED = 42

# ---- Preprocessing ---------------------------------------------------------

def preprocess_row(fen: str, moves_str: str):
    """
    Lichess convention: `FEN` is the position before the opponent's setup move,
    and `Moves[0]` is that setup move. The puzzle starts after applying Moves[0].
    Player moves are at odd indices (1, 3, 5, ...), opponent replies at even
    indices (2, 4, ...).

    Returns a dict with per-step evaluation data:
      - steps: list of {step_idx, fen, best_move, side_to_move}, one entry per
        player move in the solution.
      - solution_ply: number of player moves (== len(steps)).
      - first_eval_fen / first_best_move / first_side_to_move: convenience
        copies of step 0, for single-step-only analysis.

    Returns None if the puzzle is malformed.
    """
    moves = moves_str.split()
    if len(moves) < 2:
        return None  # need at least setup + one player move

    try:
        board = chess.Board(fen)
        setup_move = chess.Move.from_uci(moves[0])
        if setup_move not in board.legal_moves:
            return None
        board.push(setup_move)

        steps = []
        # Player moves at indices 1, 3, 5, ...; opponent replies at 2, 4, ...
        step_idx = 0
        i = 1
        while i < len(moves):
            player_uci = moves[i]
            player_move = chess.Move.from_uci(player_uci)
            if player_move not in board.legal_moves:
                return None
            steps.append({
                "step_idx": step_idx,
                "fen": board.fen(),
                "best_move": player_uci,
                "side_to_move": "white" if board.turn == chess.WHITE else "black",
            })
            board.push(player_move)
            step_idx += 1

            # Apply opponent reply if present
            if i + 1 < len(moves):
                opp_move = chess.Move.from_uci(moves[i + 1])
                if opp_move not in board.legal_moves:
                    return None
                board.push(opp_move)
            i += 2

    except (ValueError, AssertionError):
        return None

    if not steps:
        return None

    return {
        "steps": steps,
        "solution_ply": len(steps),
        "first_eval_fen": steps[0]["fen"],
        "first_best_move": steps[0]["best_move"],
        "first_side_to_move": steps[0]["side_to_move"],
    }

# ---- Sampling --------------------------------------------------------------

def load_train_fens(path: Path, fen_column: str) -> set:
    """Load FENs from .txt (one per line) or .parquet. Normalizes to the
    position-only part of the FEN (first 4 fields) so move counters don't
    cause false misses.

    For parquet, `fen_column` can be a dotted path to access nested dict
    fields, e.g. 'metadata.board_fen'."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".csv"):
        with open(path) as f:
            raw = [line.strip() for line in f if line.strip()]
    elif suffix == ".parquet":
        parts = fen_column.split(".")
        top = parts[0]
        tdf = pd.read_parquet(path, columns=[top])
        if top not in tdf.columns:
            raise ValueError(
                f"Column '{top}' not found in {path}. "
                f"Available: {list(tdf.columns)}"
            )
        series = tdf[top]
        # Walk nested fields for dict/struct columns
        for key in parts[1:]:
            series = series.apply(
                lambda v, k=key: v.get(k) if isinstance(v, dict) else (
                    v[k] if hasattr(v, "__getitem__") and v is not None else None
                )
            )
        raw = series.dropna().astype(str).tolist()
        if not raw:
            raise ValueError(
                f"No FENs extracted from path '{fen_column}'. "
                f"Check that the field exists and is populated."
            )
    else:
        raise ValueError(f"Unsupported train fens file extension: {suffix}")
    return {" ".join(fen.split()[:4]) for fen in raw}

def assign_bin(rating: int):
    for lo, hi in RATING_BINS:
        if lo <= rating < hi:
            return f"{lo}-{hi}"
    # include upper bound of last bin
    lo, hi = RATING_BINS[-1]
    if rating == hi:
        return f"{lo}-{hi}"
    return None

def stratified_sample(df: pd.DataFrame, target_size: int, seed: int):
    """Equal count per rating bin. If a bin is short, take everything."""
    per_bin = target_size // len(RATING_BINS)
    parts = []
    shortfalls = {}
    for lo, hi in RATING_BINS:
        key = f"{lo}-{hi}"
        bin_df = df[df["rating_bin"] == key]
        if len(bin_df) < per_bin:
            shortfalls[key] = (len(bin_df), per_bin)
            parts.append(bin_df)
        else:
            parts.append(bin_df.sample(n=per_bin, random_state=seed))
    return pd.concat(parts, ignore_index=True), shortfalls

# ---- Main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("./data"))
    ap.add_argument("--target-size", type=int, default=25000)
    ap.add_argument(
        "--train-fens-file",
        type=Path,
        default=None,
        help="Optional file to dedup against. Supports .txt (one FEN per line) "
             "and .parquet (requires --train-fen-column).",
    )
    ap.add_argument(
        "--train-fen-column",
        type=str,
        default="fen",
        help="Column name containing FENs when --train-fens-file is a parquet. "
             "Use dotted path for nested dict fields, e.g. 'metadata.board_fen'.",
    )
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    print("Loading Lichess/chess-puzzles ...")
    ds = load_dataset("Lichess/chess-puzzles", split="train")
    df = ds.to_pandas()
    print(f"  Total puzzles: {len(df):,}")

    # 2. Quality filter
    print("Applying quality filters ...")
    n0 = len(df)
    df = df[
        (df["RatingDeviation"] <= MAX_RATING_DEVIATION)
        & (df["NbPlays"] >= MIN_NB_PLAYS)
        & (df["Popularity"] >= MIN_POPULARITY)
    ].copy()
    print(f"  After quality filter: {len(df):,} ({len(df)/n0:.1%} kept)")

    # 3. Rating bin assignment
    df["rating_bin"] = df["Rating"].apply(assign_bin)
    df = df[df["rating_bin"].notna()].copy()
    print(f"  After rating range filter: {len(df):,}")
    print("  Per-bin counts (available):")
    for lo, hi in RATING_BINS:
        key = f"{lo}-{hi}"
        print(f"    {key}: {(df['rating_bin'] == key).sum():,}")

    # 4. Preprocess: extract per-step player moves
    print("Preprocessing FENs and extracting per-step player moves ...")
    processed = df.apply(
        lambda r: preprocess_row(r["FEN"], r["Moves"]), axis=1
    )
    valid = processed.notna()
    print(f"  Valid after preprocessing: {valid.sum():,} / {len(df):,}")
    df = df[valid].copy()
    extracted = pd.DataFrame(processed[valid].tolist(), index=df.index)
    df = pd.concat([df, extracted], axis=1)

    # 5. Optional dedup vs training FENs
    # Dedup at the puzzle level using the first eval FEN. We can't dedup
    # per-step because sampling is done at the puzzle level — a puzzle must
    # be entirely in or entirely out of the eval set for the full-puzzle
    # metric to be well-defined.
    if args.train_fens_file and args.train_fens_file.exists():
        print(f"Deduping against {args.train_fens_file} ...")
        train_fens = load_train_fens(args.train_fens_file, args.train_fen_column)
        print(f"  Loaded {len(train_fens):,} training FENs")
        df["fen_key"] = df["first_eval_fen"].apply(
            lambda f: " ".join(f.split()[:4])
        )
        before = len(df)
        df = df[~df["fen_key"].isin(train_fens)].copy()
        df = df.drop(columns=["fen_key"])
        print(f"  Removed {before - len(df):,} overlapping puzzles")

    # 6. Stratified sample (at the puzzle level)
    print(f"Stratified sampling to {args.target_size:,} puzzles ...")
    sampled, shortfalls = stratified_sample(df, args.target_size, args.seed)
    if shortfalls:
        print("  WARNING: bins with insufficient data:")
        for k, (have, want) in shortfalls.items():
            print(f"    {k}: have {have:,}, wanted {want:,}")
    print(f"  Final puzzle count: {len(sampled):,}")
    print(f"  Total player-move steps: {sampled['solution_ply'].sum():,}")

    # 7. Write per-puzzle parquet (one row per puzzle, steps as a nested list)
    puzzle_cols = [c for c in sampled.columns if c != "steps"]
    puzzle_df = sampled[puzzle_cols + ["steps"]].copy()
    out_puzzles = args.output_dir / "puzzles.parquet"
    puzzle_df.to_parquet(out_puzzles, index=False)
    print(f"Wrote {out_puzzles}  (one row per puzzle)")

    # 8. Write per-step parquet (one row per player move — this is what the
    # model actually runs inference on)
    step_rows = []
    for _, row in sampled.iterrows():
        for s in row["steps"]:
            step_rows.append({
                "PuzzleId": row["PuzzleId"],
                "GameId": row["GameId"],
                "step_idx": s["step_idx"],
                "fen": s["fen"],
                "best_move": s["best_move"],
                "side_to_move": s["side_to_move"],
                "Rating": row["Rating"],
                "rating_bin": row["rating_bin"],
                "Themes": row["Themes"],
                "solution_ply": row["solution_ply"],
            })
    steps_df = pd.DataFrame(step_rows)
    out_steps = args.output_dir / "steps.parquet"
    steps_df.to_parquet(out_steps, index=False)
    print(f"Wrote {out_steps}  ({len(steps_df):,} rows, one per player move)")

    # 9. Stats summary
    stats = {
        "puzzle_count": len(sampled),
        "step_count": int(len(steps_df)),
        "quality_filter": {
            "max_rating_deviation": MAX_RATING_DEVIATION,
            "min_nb_plays": MIN_NB_PLAYS,
            "min_popularity": MIN_POPULARITY,
        },
        "per_bin_puzzle_counts": {
            k: int((sampled["rating_bin"] == k).sum())
            for k in sampled["rating_bin"].unique()
        },
        "per_bin_step_counts": {
            k: int((steps_df["rating_bin"] == k).sum())
            for k in steps_df["rating_bin"].unique()
        },
        "rating_stats": {
            "mean": float(sampled["Rating"].mean()),
            "median": float(sampled["Rating"].median()),
            "min": int(sampled["Rating"].min()),
            "max": int(sampled["Rating"].max()),
        },
        "solution_length_distribution": {
            int(k): int(v)
            for k, v in Counter(sampled["solution_ply"].tolist()).items()
        },
        "top_themes": dict(
            Counter(
                t for themes in sampled["Themes"] for t in themes
            ).most_common(30)
        ),
        "side_to_move_steps": dict(
            Counter(steps_df["side_to_move"].tolist())
        ),
    }
    stats_path = args.output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Wrote {stats_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()