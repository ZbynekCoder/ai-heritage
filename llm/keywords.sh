export CUDA_VISIBLE_DEVICES=2

python keywords.py \
  --input results/results.jsonl \
  --output results/results_with_keywords.jsonl \
  --model models/Qwen3-4B \
  --gpu_mem_util 0.25 \
  --temperature 0.0 \
  --top_p 1.0 \
  --k 10