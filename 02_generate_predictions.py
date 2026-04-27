"""
Generate predictions on steps.parquet using a chess model via vLLM.

Aligned with eval_fast.py:
  - Tokenizer compat patch (transformers 5.x + vLLM 0.11)
  - TP auto-adjustment based on attention head divisibility
  - Olmo-3-Think thinking-prefix stripping (matches SFT input distribution)
  - bfloat16, max_model_len=4096, prefix caching enabled
  - Greedy decoding (temperature=0)
  - resolve_model_path supports HF names and FSDP checkpoint dirs

Differences from eval_fast.py:
  - Input is steps.parquet (FEN + UCI ground truth), not pre-built prompts.
    We construct the chat prompt from the FEN here.
  - We convert model SAN output to UCI before writing, so predictions match
    the UCI ground truth in steps.parquet directly. score_puzzle_eval.py
    can be run on the output without modification.

Usage:
    python generate_predictions.py \\
        --model /path/to/model_or_hf_name \\
        --steps ./data/steps.parquet \\
        --output ./outputs/predictions.parquet \\
        [--task optimal_move_fen --tp 1]
"""

import argparse
import re
import time
from pathlib import Path

import chess
import pandas as pd
from tqdm import tqdm

# ---- Prompt builders (must match training format) --------------------------

# Matches the optimal_move_fen prompt structure from training.
SYSTEM_PROMPT_FEN = (
    "You are a helpful assistant who plays chess professionally. First, think "
    "through the reasoning process internally and then provide the user with "
    "the best move. The reasoning process and the answer must be enclosed "
    "within <think> </think> and <answer> </answer> tags, respectively.\n"
    "The reasoning process should describe how you analyze the position and "
    "decide on the best move, including:\n"
    "- A strategic evaluation of the position.\n"
    "- A comparison of key candidate moves.\n"
    "- For each candidate, consider the opponent's likely response and outcome.\n"
    "- Conclude with a clear justification for the final choice.\n"
    "The answer must be in SAN notation, restricted to the moving piece and "
    "destination square (e.g., Nf3, Rxf2, c5).\n"
    "Now, the user provides the board in FEN format.\n"
    "After analyzing the position, clearly state the best move in SAN notation "
    "within <answer> </answer> tags. i.e., <answer> Nf3 </answer>.\n"
    "Reminder of chess rules:\n"
    "- Bishops move diagonally.\n"
    "- Rooks move horizontally or vertically.\n"
    "- Knights jump in an L-shape.\n"
    "- Queens combine rook and bishop movements.\n"
    "- Kings move one square in any direction.\n"
    "- Pawns move forward, capture diagonally, and can promote."
)


def build_user_message_fen(fen: str) -> str:
    # Training FEN uses 4-field format (no halfmove/fullmove counters)
    trimmed = " ".join(fen.split()[:4])
    return f"Current board in FEN: {trimmed}."


def build_chat(fen: str, task: str = "optimal_move_fen"):
    if task == "optimal_move_fen":
        return [
            {"role": "system", "content": SYSTEM_PROMPT_FEN},
            {"role": "user", "content": build_user_message_fen(fen)},
        ]
    raise ValueError(f"Unsupported task: {task}. Add prompt builder if needed.")


# ---- vLLM setup helpers (mirrored from eval_fast.py) -----------------------

def _patch_tokenizer_compat():
    """Restore all_special_tokens_extended for transformers 5.x + vLLM 0.11."""
    try:
        from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
        for cls in (PreTrainedTokenizer, PreTrainedTokenizerFast):
            if not hasattr(cls, "all_special_tokens_extended"):
                cls.all_special_tokens_extended = property(
                    lambda self: list(self.all_special_tokens)
                )
    except Exception:
        pass


def _find_best_tp(model_path: str, requested_tp: int) -> int:
    """Largest valid TP <= requested_tp that divides num_attention_heads."""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        num_heads = getattr(config, "num_attention_heads", None)
        if num_heads is None:
            return requested_tp
        if num_heads % requested_tp == 0:
            return requested_tp
        best = 1
        for p in [1, 2, 4, 8]:
            if p <= requested_tp and num_heads % p == 0:
                best = p
        print(f"[INFO] {num_heads} attention heads not divisible by tp={requested_tp}, "
              f"using tp={best} instead")
        return best
    except Exception:
        return requested_tp


def resolve_model_path(model_arg: str) -> str:
    """Resolve to HF name, FSDP checkpoint, or HF dir."""
    model_path = Path(model_arg)
    if not model_path.exists():
        return model_arg

    latest_file = model_path / "latest_checkpointed_iteration.txt"
    if latest_file.exists():
        step = latest_file.read_text().strip()
        hf_path = model_path / f"global_step_{step}" / "huggingface"
        if hf_path.exists():
            print(f"[INFO] Resolved FSDP checkpoint -> {hf_path}")
            return str(hf_path)
        step_dir = model_path / f"global_step_{step}"
        if step_dir.exists():
            return str(step_dir)

    step_dirs = sorted(model_path.glob("global_step_*"),
                       key=lambda d: int(d.name.split("_")[-1]))
    if step_dirs:
        hf_path = step_dirs[-1] / "huggingface"
        if hf_path.exists():
            print(f"[INFO] Resolved checkpoint -> {hf_path}")
            return str(hf_path)
        return str(step_dirs[-1])

    return str(model_path)


def get_thinking_prefix(llm) -> str:
    """Detect a thinking prefix the chat template appends (Olmo-3-Think etc.).

    During SFT this prefix is stripped from the prompt so the model learns to
    emit it as its first token. We replicate that here so the eval input
    distribution exactly matches training.
    """
    try:
        tokenizer = llm.get_tokenizer()
        test_messages = [{"role": "user", "content": "test"}]
        formatted = tokenizer.apply_chat_template(
            test_messages, add_generation_prompt=True, tokenize=False
        )
        for prefix in ["<think>\n", "<think>"]:
            if formatted.endswith(prefix):
                print(f"[INFO] Chat template adds thinking prefix: {repr(prefix)}")
                return prefix
    except Exception as e:
        print(f"[WARN] Could not detect thinking prefix: {e}")
    return ""


def load_vllm_model(model_path: str, tp: int, gpu_mem_util: float = 0.90,
                    max_model_len: int = 4096):
    _patch_tokenizer_compat()
    from vllm import LLM
    tp = _find_best_tp(model_path, tp)
    print(f"\n[INFO] Loading model with vLLM (tp={tp})...")
    print(f"[INFO] Model: {model_path}")
    t0 = time.time()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
    )
    print(f"[INFO] Model loaded in {time.time()-t0:.1f}s (tp={tp})")
    return llm


# ---- Inference (mirrors eval_fast.run_vllm_inference) ----------------------

def run_vllm_inference(llm, conversations, max_new_tokens: int = 4096,
                       thinking_prefix: str = ""):
    from vllm import SamplingParams
    sampling = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    if thinking_prefix:
        tokenizer = llm.get_tokenizer()
        prefix_ids = tokenizer.encode(thinking_prefix, add_special_tokens=False)
        prompts = []
        for messages in conversations:
            token_ids = list(tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ))
            if prefix_ids and token_ids[-len(prefix_ids):] == prefix_ids:
                token_ids = token_ids[:-len(prefix_ids)]
            prompts.append({"prompt_token_ids": token_ids})
        outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=True)
    else:
        outputs = llm.chat(
            messages=conversations,
            sampling_params=sampling,
            use_tqdm=True,
        )
    return [o.outputs[0].text for o in outputs]


# ---- Response parsing -------------------------------------------------------

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def extract_san(response: str):
    m = ANSWER_RE.search(response)
    if not m:
        return None
    return m.group(1).strip()


def check_format(text: str) -> dict:
    """Match eval_fast.py's format compliance check."""
    has_think = "<think>" in text and "</think>" in text
    has_answer = "<answer>" in text and "</answer>" in text
    return {
        "has_think": has_think,
        "has_answer": has_answer,
        "format_valid": has_think and has_answer,
    }


def san_to_uci(san, fen):
    """Convert SAN -> UCI given the board at `fen`. Returns (uci, status)."""
    if san is None or san == "":
        return None, "empty"
    try:
        board = chess.Board(fen)
    except ValueError:
        return None, "bad_fen"
    try:
        move = board.parse_san(san)
    except Exception:
        cleaned = san.split()[0] if san.split() else san
        try:
            move = board.parse_san(cleaned)
        except Exception:
            return None, "parse_fail"
    if move not in board.legal_moves:
        return None, "illegal_move"
    return move.uci(), "ok"


# ---- Main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    help="HF name or local checkpoint path "
                         "(FSDP dirs supported, see resolve_model_path).")
    ap.add_argument("--steps", type=Path, required=True,
                    help="Path to steps.parquet from build_puzzle_eval.py")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output predictions.parquet path")
    ap.add_argument("--task", type=str, default="optimal_move_fen",
                    choices=["optimal_move_fen"],
                    help="Prompt template. Matches eval_fast.py task names.")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of prompts (debugging)")
    args = ap.parse_args()

    # 1. Load eval set
    print(f"\nLoading {args.steps} ...")
    steps = pd.read_parquet(args.steps)
    if args.limit:
        steps = steps.head(args.limit).copy()
    print(f"  {len(steps):,} prediction tasks")

    # 2. Load model
    model_path = resolve_model_path(args.model)
    llm = load_vllm_model(model_path, tp=args.tp, gpu_mem_util=args.gpu_mem,
                          max_model_len=args.max_model_len)
    thinking_prefix = get_thinking_prefix(llm)

    # 3. Build conversations
    conversations = [build_chat(fen, args.task) for fen in steps["fen"].tolist()]

    # Debug: show first prompt
    print(f"\n[DEBUG] Example prompt (sample 0):")
    for msg in conversations[0]:
        content_preview = msg["content"][:200].replace("\n", " ")
        print(f"  [{msg['role']}]: {content_preview}{'...' if len(msg['content']) > 200 else ''}")

    # 4. Generate
    print(f"\nGenerating (max_new_tokens={args.max_new_tokens}) ...")
    t0 = time.time()
    raw_outputs = run_vllm_inference(
        llm, conversations,
        max_new_tokens=args.max_new_tokens,
        thinking_prefix=thinking_prefix,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(raw_outputs)/elapsed:.1f} samples/s)")

    # 5. Parse and convert SAN -> UCI
    print("\nParsing responses ...")
    records = []
    status_counts = {}
    fmt_counts = {"has_think": 0, "has_answer": 0, "format_valid": 0}
    for row, response in tqdm(zip(steps.itertuples(index=False), raw_outputs),
                              total=len(raw_outputs)):
        fmt = check_format(response)
        for k, v in fmt.items():
            if v:
                fmt_counts[k] += 1
        san = extract_san(response)
        if san is None:
            uci, status = None, "no_answer_tag"
        else:
            uci, status = san_to_uci(san, row.fen)
        status_counts[status] = status_counts.get(status, 0) + 1
        records.append({
            "PuzzleId": row.PuzzleId,
            "step_idx": int(row.step_idx),
            "predicted_move": uci,           # UCI; matches best_move column
            "predicted_san": san,
            "parse_status": status,
            "format_valid": fmt["format_valid"],
            "raw_response": response,
        })

    preds = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(args.output, index=False)
    print(f"\nWrote {args.output}  ({len(preds):,} rows)")

    # 6. Summary
    n = len(preds)
    print(f"\nFormat compliance:")
    print(f"  has_think    : {fmt_counts['has_think']/n:.1%}")
    print(f"  has_answer   : {fmt_counts['has_answer']/n:.1%}")
    print(f"  format_valid : {fmt_counts['format_valid']/n:.1%}")

    print(f"\nParse status:")
    for k, v in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<20s}  {v:>7,}  ({v/n:.1%})")

    merged = preds.merge(
        steps[["PuzzleId", "step_idx", "best_move"]],
        on=["PuzzleId", "step_idx"],
    )
    step_acc = (merged["predicted_move"] == merged["best_move"]).mean()
    print(f"\nStep accuracy (quick check): {step_acc:.4f}")
    print("Run score_puzzle_eval.py for the full breakdown "
          "(per-bin, per-theme, puzzle-level AND).")


if __name__ == "__main__":
    main()
