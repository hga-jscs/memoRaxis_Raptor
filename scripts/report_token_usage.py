import argparse
import json
from pathlib import Path

import pandas as pd


def load_events(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main(event_path: str):
    event_file = Path(event_path)
    if not event_file.exists():
        raise FileNotFoundError(f"Event file not found: {event_file}")

    df = load_events(event_file)
    if df.empty:
        print("No events found.")
        return

    for col in [
        "total_tokens",
        "estimated_tokens",
        "task_name",
        "adaptor",
        "token_bucket",
        "stage",
        "run_id",
        "instance_idx",
        "question_idx",
    ]:
        if col not in df.columns:
            df[col] = None

    df["token_value"] = pd.to_numeric(df["total_tokens"], errors="coerce").fillna(
        pd.to_numeric(df["estimated_tokens"], errors="coerce")
    ).fillna(0)

    overall = (
        df.groupby(["task_name", "adaptor", "token_bucket"], dropna=False)["token_value"]
        .sum()
        .reset_index()
        .sort_values(["task_name", "adaptor", "token_bucket"])
    )
    per_question = (
        df.groupby(["run_id", "task_name", "instance_idx", "question_idx", "adaptor", "token_bucket"], dropna=False)["token_value"]
        .sum()
        .reset_index()
    )
    per_stage = (
        df.groupby(["task_name", "adaptor", "stage"], dropna=False)["token_value"]
        .sum()
        .reset_index()
        .sort_values(["task_name", "adaptor", "stage"])
    )

    out_dir = Path("out/token_reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    overall.to_csv(out_dir / "token_overall.csv", index=False, encoding="utf-8-sig")
    per_question.to_csv(out_dir / "token_per_question.csv", index=False, encoding="utf-8-sig")
    per_stage.to_csv(out_dir / "token_per_stage.csv", index=False, encoding="utf-8-sig")

    with open(out_dir / "token_report.md", "w", encoding="utf-8") as f:
        f.write("# Token 成本报告\n\n")
        f.write(f"事件文件: `{event_file}`\n\n")
        f.write("## 总桶统计\n\n")
        try:
            overall_md = overall.to_markdown(index=False)
            stage_md = per_stage.to_markdown(index=False)
        except Exception:
            # 在无 tabulate 依赖时回退为纯文本表格，保证脚本可用性。
            overall_md = overall.to_string(index=False)
            stage_md = per_stage.to_string(index=False)

        f.write(overall_md)
        f.write("\n\n## 分阶段统计\n\n")
        f.write(stage_md)

    print(f"报告已生成: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate token usage from JSONL events")
    parser.add_argument("event_path", type=str, help="Path to *.events.jsonl")
    args = parser.parse_args()
    main(args.event_path)
