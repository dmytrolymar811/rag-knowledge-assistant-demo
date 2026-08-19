from __future__ import annotations

import argparse
from pathlib import Path

from .retriever import KnowledgeBase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search a local document knowledge base.")
    parser.add_argument("question", help="Question to search for")
    parser.add_argument("--documents", type=Path, default=Path("sample_documents"))
    parser.add_argument("--limit", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    knowledge_base = KnowledgeBase.from_directory(args.documents)
    results = knowledge_base.search(args.question, limit=args.limit)

    print(f"Question: {args.question}\n")
    if not results:
        print("No sufficiently grounded passage was found. Ask for clarification or add relevant documents.")
        return

    print("Grounded context:")
    for result in results:
        print(f"\n{result.text}\n\nSource: {result.source} (score: {result.score:.2f})")


if __name__ == "__main__":
    main()
