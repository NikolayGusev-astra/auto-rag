#!/usr/bin/env python3
"""Pre-load все модели в LM Studio."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "rag_core"))


def warmup_embedding():
    from embedding_service import EmbeddingService
    print("Warming up embedding model...")
    t0 = time.time()
    svc = EmbeddingService.get()
    svc.embed("warmup query for embedding model")
    print(f"  Loaded in {time.time()-t0:.1f}s")


def warmup_reranker():
    from reranker_service import RerankerService
    print("Warming up reranker model...")
    t0 = time.time()
    svc = RerankerService.get()
    svc.rerank("warmup query", ["document 1", "document 2"])
    print(f"  Loaded in {time.time()-t0:.1f}s")


def warmup_llm():
    from llm_service import get_llm
    from rag_config import (
        LLM_CLASSIFY_MODEL, LLM_VERIFY_MODEL, LLM_EVAL_MODEL,
    )
    models = set([LLM_CLASSIFY_MODEL, LLM_VERIFY_MODEL, LLM_EVAL_MODEL])
    llm = get_llm()
    for model in models:
        print(f"Warming up LLM: {model}")
        t0 = time.time()
        try:
            llm.chat(
                messages=[{"role": "user", "content": "warmup"}],
                model=model,
                max_tokens=5,
            )
            print(f"  Loaded in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-only", action="store_true")
    parser.add_argument("--reranker-only", action="store_true")
    parser.add_argument("--llm-only", action="store_true")
    args = parser.parse_args()

    from lm_studio_monitor import get_lm_studio
    status = get_lm_studio().get_status(force=True)
    print(f"LM Studio: {status['base_url']}")
    print(f"Available: {status['available']}")
    if not status["available"]:
        print(f"Error: {status.get('error')}")
        sys.exit(1)

    print()
    if not args.llm_only and not args.reranker_only:
        warmup_embedding()
        print()
    if not args.llm_only and not args.embedding_only:
        warmup_reranker()
        print()
    if not args.embedding_only and not args.reranker_only:
        warmup_llm()

    print()
    print("WARMUP COMPLETE")


if __name__ == "__main__":
    main()
