"""Build models/S7seed/<ARM>512_sft_Y: symlinks to the SAME SFT-only weights/tokenizer + a config with YaRN rope scaling (D46a branch ii).
 max_position_embeddings 9216 = prompt 1024 + response 8192; rope_parameters yarn factor F (default 2.25 = 9216/4096) with original_max_position_embeddings 4096.
 Why 2.25 and not 2.0: vLLM's YaRN cos/sin cache has original*factor positions (4096*2.0 = 8192), so with factor 2.0 prompt+response could never exceed 8192
 (positions >= 8192 would read outside the cache); 8192 response tokens + prompt needs 9216 = 4096*2.25.  HF 5.x takes `factor` verbatim from rope_parameters.
 python make_yarn_dir.py ARM [FACTOR]"""
import json, os, sys
R = os.environ.get("ARR_ROOT", "/<CLUSTER>/<USER>/arr_oct"); arm = sys.argv[1]; F = float(sys.argv[2]) if len(sys.argv) > 2 else 2.25
src, dst = "%s/models/S7seed/%s512_sft" % (R, arm), "%s/models/S7seed/%s512_sft_Y" % (R, arm)
os.makedirs(dst, exist_ok=True)
for f in "model.safetensors tokenizer.json tokenizer_config.json generation_config.json chat_template.jinja sft_meta.json".split():
    if os.path.lexists(dst + "/" + f): os.remove(dst + "/" + f)
    os.symlink(src + "/" + f, dst + "/" + f)
c = json.load(open(src + "/config.json"))
assert c["max_position_embeddings"] == 4096 and c["rope_parameters"]["rope_type"] == "default"
c["max_position_embeddings"] = int(4096 * F)
c["rope_parameters"] = {"rope_theta": c["rope_parameters"]["rope_theta"], "rope_type": "yarn", "factor": F, "original_max_position_embeddings": 4096}
json.dump(c, open(dst + "/config.json", "w"), indent=2)
print(dst, c["max_position_embeddings"], c["rope_parameters"])
