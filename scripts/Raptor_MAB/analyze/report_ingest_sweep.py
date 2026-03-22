import argparse
import json
from pathlib import Path

import pandas as pd


def load_events(paths):
    rows = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main(pattern: str):
    paths = sorted(Path(".").glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")

    df = load_events(paths)
    if df.empty:
        print("No events found.")
        return

    for col in [
        "run_id",
        "task_name",
        "instance_idx",
        "requested_target_chunks",
        "actual_chunk_count",
        "event_type",
        "token_bucket",
        "total_tokens",
        "estimated_tokens",
        "latency_ms",
        "chunk_strategy",
    ]:
        if col not in df.columns:
            df[col] = None

    df["token_value"] = pd.to_numeric(df["total_tokens"], errors="coerce").fillna(
        pd.to_numeric(df["estimated_tokens"], errors="coerce")
    ).fillna(0)

    chunk_plan = (
        df[df["event_type"] == "chunk_plan"][
            ["run_id", "task_name", "instance_idx", "requested_target_chunks", "actual_chunk_count", "chunk_strategy"]
        ]
        .drop_duplicates(subset=["run_id"])
    )

    tree_build = (
        df[df["event_type"] == "tree_build"][["run_id", "latency_ms"]]
        .rename(columns={"latency_ms": "tree_build_latency_ms"})
        .drop_duplicates(subset=["run_id"])
    )

    token_sum = (
        df[df["event_type"] == "model_call"]
        .groupby(["run_id", "token_bucket"], dropna=False)["token_value"]
        .sum()
        .reset_index()
    )

    token_pivot = token_sum.pivot(index="run_id", columns="token_bucket", values="token_value").reset_index()
    token_pivot = token_pivot.fillna(0)

    out = chunk_plan.merge(tree_build, on="run_id", how="left").merge(token_pivot, on="run_id", how="left")

    ingest_cols = [c for c in out.columns if isinstance(c, str) and c.startswith("ingest_")]
    out["ingest_total_tokens"] = out[ingest_cols].sum(axis=1)

    out_dir = Path("out/chunk_sweep_reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "ingest_sweep_report.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate ingest sweep event logs")
    parser.add_argument(
        "--pattern",
        type=str,
        default="log/*.events.jsonl",
        help="Glob pattern for event jsonl files",
    )
    args = parser.parse_args()
    main(args.pattern)
