#!/usr/bin/env bash
# RAG pipeline entrypoint.
# Usage: bash run.sh <questions_path> <predictions_path>

set -euo pipefail

python3 run_pipeline.py "$1" "$2" --retriever dense --top-k 5
