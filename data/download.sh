#!/bin/sh
# Benchmarks used in the paper (public; original licences apply). MATH-500 (MIT) is included in data/MATH-500.
set -e
mkdir -p data/AIME24 data/AIME25 data/AMC23 data/GSM8K data/MMLU-Pro
curl -L -o data/AIME24/test.jsonl https://huggingface.co/datasets/math-ai/aime24/resolve/main/test.jsonl
curl -L -o data/AIME25/test.jsonl https://huggingface.co/datasets/math-ai/aime25/resolve/main/test.jsonl
curl -L -o data/AMC23/test.jsonl  https://huggingface.co/datasets/math-ai/amc23/resolve/main/test-00000-of-00001.parquet
curl -L -o data/GSM8K/test.parquet https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet
curl -L -o data/MMLU-Pro/test.parquet https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet
echo "Model checkpoints are downloaded from the Hugging Face Hub by the ids listed in results/keys and code/exp/ARTIFACTS.md."
