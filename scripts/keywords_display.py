import json

# INPUT = "../results/results_with_keywords.jsonl"
# OUTPUT = "../results/keywords_only.jsonl"

INPUT = "../results/dep_results.jsonl"
OUTPUT = "../results/dep_results_keywords_only.jsonl"

with open(INPUT, "r", encoding="utf-8") as fin, open(OUTPUT, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        kws = obj.get("keywords", [])
        # 只保留 keywords
        out = {"keywords": kws}
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")

print(f"Done -> {OUTPUT}")
