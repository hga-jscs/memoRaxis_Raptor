import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.benchmark_utils import chunk_context, load_benchmark_data, parse_instance_indices
from src.logger import bind_trace, get_event_file_path, get_logger, log_event
from src.raptor_memory import RaptorTreeMemory

logger = get_logger()


def ingest_one_instance(
    instance_idx: int,
    chunk_size: int,
    save_dir: str,
    tb_num_layers: int,
    target_chunks: int | None = None,
):
    logger.info(f"=== Processing Instance {instance_idx} (RAPTOR) ===")
    data_path = "MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet"

    try:
        data = load_benchmark_data(data_path, instance_idx)
    except Exception as e:
        logger.error(f"Error loading instance {instance_idx}: {e}")
        return

    if target_chunks is not None:
        chunks = chunk_context(data["context"], target_chunks=target_chunks)
        chunk_strategy = "target_chunks_exact_text"
    else:
        chunks = chunk_context(data["context"], chunk_size=chunk_size)
        chunk_strategy = "default_chunk_size_or_document_regex"

    memory = RaptorTreeMemory(tb_num_layers=tb_num_layers)

    run_id = f"accurate_retrieval_ingest_inst{instance_idx}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    with bind_trace(
        run_id=run_id,
        task_name="accurate_retrieval",
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
            actual_chunk_count=len(chunks),
            success=True,
        )

        print(
            f"[DEBUG] Chunk strategy={chunk_strategy}, "
            f"requested_target_chunks={target_chunks}, actual_chunk_count={len(chunks)}"
        )
        print(f"Starting ingestion of {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            memory.add_memory(chunk, metadata={"doc_id": i, "instance_idx": instance_idx})
            if i % 10 == 0:
                print(f"Queued {i}/{len(chunks)} chunks...", end="\r", flush=True)

        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tree_path = out_dir / f"raptor_acc_ret_{instance_idx}.pkl"
        memory.save_tree(str(tree_path))

    print(f"\nIngestion complete. RAPTOR tree saved to {tree_path}.")
    logger.info("Ingest event file: %s", get_event_file_path())


def main():
    parser = argparse.ArgumentParser(description="Ingest MemoryAgentBench data (RAPTOR)")
    parser.add_argument("--instance_idx", type=str, default="0", help="Index range (e.g., '0', '0-5', '1,3')")
    parser.add_argument("--chunk_size", type=int, default=850, help="Fallback chunk size")
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

    for idx in indices:
        ingest_one_instance(
            idx,
            args.chunk_size,
            args.save_dir,
            args.tb_num_layers,
            target_chunks=args.target_chunks,
        )


if __name__ == "__main__":
    main()
