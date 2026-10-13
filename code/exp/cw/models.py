"""Model/benchmark table of the Calibrated Wait matrix (D50 primaries).  host roots: hostA <WORK>/arr_oct, hostA $ARR_ROOT."""
THINK_BF = "\n\nI need to stop reasoning and give the final answer now.\n</think>\n\n**Final Answer**\n\\boxed{"
PLAIN_BF = "\n\nFinal answer: \\boxed{"
MODELS = {
    # key: template, T, top_p, dir (relative to the host root), bf suffix, caps (math500, aime/amc, gsm8k)
    "rlzero":     {"template": "olmo_rlzero", "T": 1.0, "top_p": 1.0, "dir": {"hostA": "models/Olmo-3-1025-7B-rlzero"}, "bf": PLAIN_BF, "caps": (8192, 12288, 4096)},
    "olmo3think": {"template": "olmo_think", "T": 0.6, "top_p": 0.95, "dir": {"hostA": "models/S2/1375"}, "bf": THINK_BF, "caps": (12288, 16384, 8192)},
    "r1d7":       {"template": "qwen_boxed", "T": 0.6, "top_p": 0.95, "dir": {"hostA": "models/DeepSeek-R1-Distill-Qwen-7B", "hostA": "models/B/r1-distill-qwen-7b"}, "bf": THINK_BF, "caps": (12288, 16384, 8192)},
    "r1_0528":    {"template": "qwen_boxed", "T": 0.6, "top_p": 0.95, "dir": {"hostA": "models/DeepSeek-R1-0528-Qwen3-8B-fixtok"}, "bf": THINK_BF, "caps": (12288, 16384, 8192)},   # secondary (D50e)
    "qwen3_8b":   {"template": "qwen_boxed", "T": 0.6, "top_p": 0.95, "dir": {"hostA": "models/Qwen3-8B", "hostA": "models/Qwen3-8B"}, "bf": THINK_BF, "caps": (12288, 16384, 8192)},
}


def benches(root, key):
    c = MODELS[key]["caps"]
    return [{"name": "math500", "path": root + "/data/MATH-500/test.jsonl", "n": 500, "samples": 1, "cap": c[0]},
            {"name": "aime24", "path": root + "/data/aime24/test-00000-of-00001.parquet", "n": 30, "samples": 4, "cap": c[1]},
            {"name": "aime25", "path": root + "/data/aime25/test.jsonl", "n": 30, "samples": 4, "cap": c[1]},
            {"name": "amc23", "path": root + "/data/amc23/test-00000-of-00001.parquet", "n": 40, "samples": 4, "cap": c[1]},
            {"name": "gsm8k", "path": root + "/data/gsm8k/main/test-00000-of-00001.parquet", "n": 500, "samples": 1, "cap": c[2]}]
