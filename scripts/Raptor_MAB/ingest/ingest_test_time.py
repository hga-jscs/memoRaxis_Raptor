import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.benchmark_utils import group_units_exact, parse_instance_indices
from src.logger import bind_trace, get_event_file_path, get_logger, log_event
from src.raptor_memory import RaptorTreeMemory

logger = get_logger()


def chunk_dialogues(context: str) -> List[str]:
    parts = re.split(r"\n(Dialogue \d+:)", "\n" + context)
    chunks = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        full_text = f"{header}\n{body}"
        if len(full_text) > 10:
            chunks.append(full_text)
    return chunks


def chunk_accumulation(context: str, min_chars: int = 800) -> List[str]:
    lines = [line.strip() for line in context.split("\n") if line.strip()]
    chunks = []
    current_chunk_lines = []
    current_length = 0

    for line in lines:
        current_chunk_lines.append(line)
        current_length += len(line)

        if current_length > min_chars:
            chunks.append("\n".join(current_chunk_lines))
            current_chunk_lines = []
            current_length = 0

    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))

    return chunks


def ingest_one_instance(
    instance_idx: int,
    save_dir: str,
    tb_num_layers: int,
    target_chunks: int | None = None,
):
    logger.info(f"=== Processing TTL Instance {instance_idx} (RAPTOR) ===")

    data_path = f"MemoryAgentBench/preview_samples/Test_Time_Learning/instance_{instance_idx}.json"

    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading instance {instance_idx}: {e}")
        return

    context = data["context"]

    if target_chunks is not None:
        if "Dialogue 1:" in context[:500]:
            base_units = chunk_dialogues(context)
            chunks = group_units_exact(base_units, target_chunks, joiner="\n\n")
            chunk_strategy = "target_chunks_grouped_dialogues"
        else:
            base_units = [line.strip() for line in context.split("\n") if line.strip()]
            chunks = group_units_exact(base_units, target_chunks, joiner="\n")
            chunk_strategy = "target_chunks_grouped_lines"
    else:
        if "Dialogue 1:" in context[:500]:
            logger.info("Strategy: Regex Split (Dialogue mode)")
            chunks = chunk_dialogues(context)
            chunk_strategy = "default_dialogue_regex"
        else:
            logger.info("Strategy: Accumulation > 800 chars (ShortText mode)")
            chunks = chunk_accumulation(context, min_chars=800)
            chunk_strategy = "default_800char_accumulation"

    logger.info(f"Building RAPTOR tree with {len(chunks)} chunks (instance={instance_idx})")
    memory = RaptorTreeMemory(tb_num_layers=tb_num_layers)

    run_id = f"ttl_ingest_inst{instance_idx}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    with bind_trace(
        run_id=run_id,
        task_name="test_time_learning",
        instance_idx=instance_idx,
        adaptor="INGEST",
        stage="ingest.tree_build",
    ):
        log_event(
            "chunk_plan",
            stage="ingest.chunking",
            chunk_strategy=chunk_strategy,
            requested_target_chunks=target_chunks,
            actual_chunk_count=len(chunks),
            success=True,
        )

        print(
            f"[DEBUG] Chunk strategy={chunk_strategy}, "
            f"requested_target_chunks={target_chunks}, actual_chunk_count={len(chunks)}"
        )
        print(f"Starting ingestion for Instance {instance_idx} ({len(chunks)} chunks)...")
        for i, chunk in enumerate(chunks):
            memory.add_memory(chunk, metadata={"chunk_id": i, "instance_idx": instance_idx})
            if i % 50 == 0:
                print(f"  Queued {i}/{len(chunks)}...", end="\r", flush=True)

        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tree_path = out_dir / f"raptor_ttl_{instance_idx}.pkl"
        memory.save_tree(str(tree_path))

    print(f"\nInstance {instance_idx} complete. RAPTOR tree saved -> {tree_path}\n")
    logger.info("Ingest event file: %s", get_event_file_path())


def main():
    parser = argparse.ArgumentParser(description="Ingest Test_Time_Learning data (RAPTOR)")
    parser.add_argument("--instance_idx", type=str, default="0-5", help="Index range (e.g., '0-5')")
    parser.add_argument(
        "--target_chunks",
        type=int,
        default=None,
        help="Force exact chunk count for ingest-cost sweeps; overrides default auto chunking when set",
    )
    parser.add_argument("--save_dir", type=str, default="out/raptor_trees", help="Where to save RAPTOR trees")
    parser.add_argument("--tb_num_layers", type=int, default=3, help="RAPTOR tree layers")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")

    for idx in indices:
        ingest_one_instance(
            idx,
            args.save_dir,
            args.tb_num_layers,
            target_chunks=args.target_chunks,
        )


if __name__ == "__main__":
    main()
