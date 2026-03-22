import argparse
import json
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


def chunk_facts(context: str, min_chars: int = 800) -> List[str]:
    lines = [line.strip() for line in context.split("\n") if line.strip()]

    chunks = []
    current_chunk_lines = []
    current_length = 0

    for line in lines:
        current_chunk_lines.append(line)
        current_length += len(line)

        if current_length > min_chars:
            chunk_text = "\n".join(current_chunk_lines)
            chunks.append(chunk_text)
            current_chunk_lines = []
            current_length = 0

    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines)
        chunks.append(chunk_text)

    return chunks


def ingest_one_instance(
    instance_idx: int,
    min_chars: int,
    save_dir: str,
    tb_num_layers: int,
    target_chunks: int | None = None,
):
    logger.info(f"=== Processing Conflict_Resolution Instance {instance_idx} (RAPTOR) ===")

    data_path = f"MemoryAgentBench/preview_samples/Conflict_Resolution/instance_{instance_idx}.json"

    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading instance {instance_idx}: {e}")
        return

    lines = [line.strip() for line in data["context"].split("\n") if line.strip()]
    if target_chunks is not None:
        chunks = group_units_exact(lines, target_chunks, joiner="\n")
        chunk_strategy = "target_chunks_grouped_facts"
    else:
        chunks = chunk_facts(data["context"], min_chars=min_chars)
        chunk_strategy = "default_min_chars_accumulation"

    logger.info(f"Building RAPTOR tree with {len(chunks)} chunks (instance={instance_idx})")
    memory = RaptorTreeMemory(tb_num_layers=tb_num_layers)

    run_id = f"conflict_resolution_ingest_inst{instance_idx}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    with bind_trace(
        run_id=run_id,
        task_name="conflict_resolution",
        instance_idx=instance_idx,
        adaptor="INGEST",
        stage="ingest.tree_build",
    ):
        log_event(
            "chunk_plan",
            stage="ingest.chunking",
            chunk_strategy=chunk_strategy,
            requested_target_chunks=target_chunks,
            min_chars=min_chars,
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
            if i % 10 == 0:
                print(f"  Queued {i}/{len(chunks)}...", end="\r", flush=True)

        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tree_path = out_dir / f"raptor_conflict_{instance_idx}.pkl"
        memory.save_tree(str(tree_path))

    print(f"\nInstance {instance_idx} complete. RAPTOR tree saved -> {tree_path}\n")
    logger.info("Ingest event file: %s", get_event_file_path())


def main():
    parser = argparse.ArgumentParser(description="Ingest Conflict_Resolution data (RAPTOR)")
    parser.add_argument("--instance_idx", type=str, default="0-7", help="Index range (e.g., '0-7')")
    parser.add_argument("--min_chars", type=int, default=800, help="Minimum chars per chunk")
    parser.add_argument(
        "--target_chunks",
        type=int,
        default=None,
        help="Force exact chunk count for ingest-cost sweeps; overrides min_chars chunking when set",
    )
    parser.add_argument("--save_dir", type=str, default="out/raptor_trees", help="Where to save RAPTOR trees")
    parser.add_argument("--tb_num_layers", type=int, default=3, help="RAPTOR tree layers")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")

    for idx in indices:
        ingest_one_instance(
            idx,
            args.min_chars,
            args.save_dir,
            args.tb_num_layers,
            target_chunks=args.target_chunks,
        )


if __name__ == "__main__":
    main()
