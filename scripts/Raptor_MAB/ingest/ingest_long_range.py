import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.benchmark_utils import chunk_context, parse_instance_indices
from src.logger import bind_trace, get_event_file_path, get_logger, log_event
from src.raptor_memory import RaptorTreeMemory

logger = get_logger()


def ingest_one_instance(
    instance_idx: int,
    chunk_size: int,
    overlap: int,
    save_dir: str,
    tb_num_layers: int,
    target_chunks: int | None = None,
):
    logger.info(f"=== Processing Long_Range_Understanding Instance {instance_idx} (RAPTOR) ===")

    data_path = f"MemoryAgentBench/preview_samples/Long_Range_Understanding/instance_{instance_idx}.json"

    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading instance {instance_idx}: {e}")
        return

    if target_chunks is not None:
        chunks = chunk_context(data["context"], target_chunks=target_chunks)
        chunk_strategy = "target_chunks_exact_text"
    else:
        chunks = chunk_context(data["context"], chunk_size=chunk_size, overlap=overlap)
        chunk_strategy = "default_chunk_size_or_document_regex"

    logger.info(f"Building RAPTOR tree with {len(chunks)} chunks (instance={instance_idx})")
    memory = RaptorTreeMemory(tb_num_layers=tb_num_layers)

    run_id = f"long_range_ingest_inst{instance_idx}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    with bind_trace(
        run_id=run_id,
        task_name="long_range_understanding",
        instance_idx=instance_idx,
        adaptor="INGEST",
        stage="ingest.tree_build",
    ):
        log_event(
            "chunk_plan",
            stage="ingest.chunking",
            chunk_strategy=chunk_strategy,
            requested_target_chunks=target_chunks,
            chunk_size=chunk_size,
            overlap=overlap,
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
            if i % 100 == 0:
                print(f"  Queued {i}/{len(chunks)}...", end="\r", flush=True)

        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tree_path = out_dir / f"raptor_long_range_{instance_idx}.pkl"
        memory.save_tree(str(tree_path))

    print(f"\nInstance {instance_idx} complete. RAPTOR tree saved -> {tree_path}\n")
    logger.info("Ingest event file: %s", get_event_file_path())


def main():
    parser = argparse.ArgumentParser(description="Ingest Long_Range_Understanding data (RAPTOR)")
    parser.add_argument("--instance_idx", type=str, default="0-39", help="Index range (e.g., '0-39')")
    parser.add_argument("--chunk_size", type=int, default=1200, help="Chunk size for sliding window")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap for sliding window")
    parser.add_argument(
        "--target_chunks",
        type=int,
        default=None,
        help="Force exact chunk count for ingest-cost sweeps; overrides default chunking strategy when set",
    )
    parser.add_argument("--save_dir", type=str, default="out/raptor_trees", help="Where to save RAPTOR trees")
    parser.add_argument("--tb_num_layers", type=int, default=3, help="RAPTOR tree layers")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")
    logger.info(
        f"Config: Chunk Size={args.chunk_size}, Overlap={args.overlap}, "
        f"Target Chunks={args.target_chunks}, Layers={args.tb_num_layers}"
    )

    for idx in indices:
        ingest_one_instance(
            idx,
            args.chunk_size,
            args.overlap,
            args.save_dir,
            args.tb_num_layers,
            target_chunks=args.target_chunks,
        )


if __name__ == "__main__":
    main()
