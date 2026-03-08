import json
from pathlib import Path

import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


SPLITS = {
    "Accurate_Retrieval-00000-of-00001.parquet": "Accurate_Retrieval",
    "Conflict_Resolution-00000-of-00001.parquet": "Conflict_Resolution",
    "Long_Range_Understanding-00000-of-00001.parquet": "Long_Range_Understanding",
    "Test_Time_Learning-00000-of-00001.parquet": "Test_Time_Learning",
}


def convert_split(parquet_name: str, output_folder_name: str) -> None:
    data_path = Path("MemoryAgentBench/data") / parquet_name
    output_dir = Path("MemoryAgentBench/preview_samples") / output_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"缺少数据文件：{data_path}")

    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        raise RuntimeError(
            f"读取 parquet 失败：{data_path}。请确认已安装 pyarrow，原始错误：{e}"
        ) from e

    print(f"[OK] Processing {parquet_name} ({len(df)} instances)...")
    for i in range(len(df)):
        instance_data = df.iloc[i].to_dict()
        file_path = output_dir / f"instance_{i}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(instance_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print(f"[OK] Converted {parquet_name} -> {output_dir}")



def main() -> None:
    failures = []
    for parquet_name, output_folder_name in SPLITS.items():
        try:
            convert_split(parquet_name, output_folder_name)
        except Exception as e:
            failures.append(f"{parquet_name}: {e}")
            print(f"[FAIL] {parquet_name}: {e}")

    if failures:
        joined = "\n".join(failures)
        raise SystemExit(f"convert_all_data.py 失败：\n{joined}")


if __name__ == "__main__":
    main()