"""Fix for transformers 5.9.0 Olmo3 (the version pinned by verl 0.9.1 in envs/rl).

Bug: modeling_olmo3.Olmo3Model builds ONE rotary embedding (YaRN, attention_factor 1.208) and feeds it to every layer. OLMo-3 (OLMo-core, and
vLLM's olmo2.py: "Rope scaling is only applied on full attention layers") uses YaRN on the 8 full-attention layers only; the 24 sliding-window
layers use plain RoPE (theta 5e5). With the bug, FSDP-actor log-probs differ from the vLLM rollout log-probs under IDENTICAL weights
(verl log: rollout_corr/kl ~0.57, Pearson 0.68 at step 1). transformers >= 5.18 already has per-layer-type rope (envs/gen has 5.18.0).

Use: put this directory on PYTHONPATH and pass  actor_rollout_ref.model.external_lib=olmo3_rope_fix  (verl imports it in every worker).
The patch is stateless (no new buffers, so FSDP/meta-device init is unaffected): the rotary module's forward additionally returns the plain-RoPE
cos/sin, and Olmo3Attention.forward picks the pair that matches its layer type. No-op on transformers versions that already fix this.
"""
import inspect
import torch
from transformers.models.olmo3 import modeling_olmo3 as M

_PATCHED = "_p1_rope_fix"


def _apply():
    if getattr(M.Olmo3Attention, _PATCHED, False):
        return
    if "layer_type" in inspect.signature(M.Olmo3RotaryEmbedding.forward).parameters:   # >= 5.18: already per-layer-type
        M.Olmo3Attention._p1_rope_fix = "native"
        return
    orig_rot = M.Olmo3RotaryEmbedding.forward
    orig_att = M.Olmo3Attention.forward

    def rot_forward(self, x, position_ids):
        cos, sin = orig_rot(self, x, position_ids)                # YaRN pair (full-attention layers)
        cfg = self.config
        dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        base = cfg.rope_parameters["rope_theta"]
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=x.device, dtype=torch.float) / dim))
        inv = inv[None, :, None].expand(position_ids.shape[0], -1, 1)
        with torch.autocast(device_type=x.device.type if x.device.type != "mps" else "cpu", enabled=False):
            freqs = (inv.float() @ position_ids[:, None, :].float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            return cos, sin, emb.cos().to(x.dtype), emb.sin().to(x.dtype)   # (yarn cos, yarn sin, plain cos, plain sin)

    def att_forward(self, hidden_states, position_embeddings, attention_mask, *args, **kwargs):
        if len(position_embeddings) == 4:
            position_embeddings = position_embeddings[2:] if self.sliding_window is not None else position_embeddings[:2]
        return orig_att(self, hidden_states, position_embeddings, attention_mask, *args, **kwargs)

    M.Olmo3RotaryEmbedding.forward = rot_forward
    M.Olmo3Attention.forward = att_forward
    M.Olmo3Attention._p1_rope_fix = "patched"


_apply()
