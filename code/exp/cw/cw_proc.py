"""Calibrated Wait (D50): single-pass vLLM V1 logits processor.  Config: env CW_CONFIG -> JSON with the frozen tables (cw_prep.py) plus the run settings:
  mode      none | sham | nowait | random | late | soft | cw | probe | cgrs   (cgrs: CGRS baseline, Huang et al. 2025 Alg. 1: at EVERY step mask the trigger tokens (cfg cgrs_ids) with the per-request probability extra_args["p"], set by the probe loop)
  lam       soft: constant logit penalty (nats) on all marker forms at sentence starts (uniform soft marker penalty, D50a)
  tau       cw: mask the marker forms at a sentence start iff  s = mean token entropy (nats) of the just-completed sentence  <  tau   (low entropy = no error signal)
  p_bins / bin_edges  random: suppression probability per response-position bin (edges in generated tokens), rate-matched to CW's realised per-bin rate
  t_late    late: mask at every sentence start once the response has >= t_late generated tokens
  stats_path  json written periodically/at removal: per cw_id {n_boundary, n_suppressed, n_marker_after_boundary, n_tokens, bins}
Requests carry SamplingParams(extra_args={'cw_id': int}); requests without it are untouched.  sham computes everything and never masks (must reproduce vanilla).
The decision applies to ONE position (the token right after a sentence end, and the token after a following connector); nothing else is changed.
LLM(..., logits_processors=['exp.cw.cw_proc:CalibratedWaitProcessor'], async_scheduling=False)
"""
from __future__ import annotations

import json
import os

import torch

from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates


class _Req:
    __slots__ = ("pcg", "s0", "cw_id", "out", "ent_sum", "n_sent", "carry", "last_seen", "n_boundary", "n_supp", "n_marker_after", "bins_b", "bins_s")

    def __init__(self, cw_id, out, nb, s0=None, pcg=0.0):
        self.pcg = pcg; self.s0 = s0; self.cw_id = cw_id; self.out = out; self.ent_sum = 0.0; self.n_sent = 0; self.carry = False; self.last_seen = 0
        self.n_boundary = self.n_supp = self.n_marker_after = 0; self.bins_b = [0] * nb; self.bins_s = [0] * nb


class CalibratedWaitProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        cfg = json.load(open(os.environ["CW_CONFIG"]))
        self.cfg = cfg; self.device = device; self.mode = cfg.get("mode", "cw"); self.tau = float(cfg.get("tau", 0.0)); self.t_late = int(cfg.get("t_late", 1 << 30))
        self.edges = cfg.get("bin_edges", [0, 500, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000]); self.p_bins = cfg.get("p_bins", [0.0] * len(self.edges))
        self.stats_path = cfg.get("stats_path"); self.lam = float(cfg.get("lam", 0.0)); self.tau_p = float(cfg.get("tau_p", 0.0))
        self.marker = torch.tensor(cfg["marker_ids"], dtype=torch.long, device=device)
        self.cgrs = torch.tensor(cfg.get("cgrs_ids", [0]), dtype=torch.long, device=device)
        self.B = set(cfg["boundary_ids"]); self.C = set(cfg["connector_ids"]); self.M = set(cfg["marker_ids"]); self.D = set(cfg["digit_end_ids"])
        self.dot_ids = {i for i in cfg.get("dot_ids", [])}
        self.reqs: dict[int, _Req] = {}; self.done: dict[int, dict] = {}; self.steps = 0
        self.gen = torch.Generator(device="cpu"); self.gen.manual_seed(int(cfg.get("seed", 0)))

    def is_argmax_invariant(self) -> bool:
        return False      # also for sham: vLLM skips argmax-invariant processors for greedy batches, which would silence the entropy/probe bookkeeping in greedy checks

    def _bin(self, n):
        b = 0
        for k, e in enumerate(self.edges):
            if n >= e:
                b = k
        return b

    def _dump(self):
        if not self.stats_path:
            return
        live = {r.cw_id: self._row(r) for r in self.reqs.values()}
        tmp = self.stats_path + ".tmp"
        json.dump({"done": self.done, "live": live}, open(tmp, "w")); os.replace(tmp, self.stats_path)

    @staticmethod
    def _row(r):
        return {"n_boundary": r.n_boundary, "n_suppressed": r.n_supp, "n_marker_after_boundary": r.n_marker_after, "n_tokens": len(r.out), "bins_b": r.bins_b, "bins_s": r.bins_s}

    def update_state(self, batch_update: BatchUpdate | None):
        if not batch_update:
            return
        for index in batch_update.removed:
            r = self.reqs.get(index)
            if r is not None:
                self.done[r.cw_id] = self._row(r)
        nb = len(self.edges)
        process_dict_updates(self.reqs, batch_update, lambda params, _p, out: _Req(params.extra_args["cw_id"], out, nb, params.extra_args.get("s0"), params.extra_args.get("p", 0.0)) if params.extra_args and "cw_id" in params.extra_args else None)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.mode == "none" or not self.reqs:
            return logits
        self.steps += 1
        if os.environ.get("CW_PROBE_LOG") and self.reqs:   # D51 sanity: probe scores aligned with the logits rows (logged for the first rows only)
            import exp.cw.probe_state as PS
            if PS.S is not None and self.steps % int(os.environ.get("CW_PROBE_EVERY", "50")) == 0:
                with open(os.environ["CW_PROBE_LOG"], "a") as f:
                    f.write(json.dumps({"step": self.steps, "rows": int(logits.shape[0]), "S_rows": int(PS.S.shape[0]), "S_head": [round(float(x), 5) for x in PS.S[:4].tolist()], "cw_id0": (self.reqs[min(self.reqs)].cw_id if self.reqs else None), "n_out0": (len(self.reqs[min(self.reqs)].out) if self.reqs else None)}) + "\n")
        if self.mode == "cgrs":
            idx = [i for i in self.reqs if i < logits.shape[0] and self.reqs[i].pcg > 0.0]
            if idx:
                ps = torch.tensor([self.reqs[i].pcg for i in idx], dtype=torch.float32)
                hit = [i for i, u in zip(idx, torch.rand(len(idx), generator=self.gen).tolist()) if u < float(ps[idx.index(i)])]
                if hit:
                    rows = torch.tensor(hit, dtype=torch.long, device=logits.device)
                    logits[rows[:, None], self.cgrs[None, :]] = float("-inf")
            return logits
        idxs = [i for i in self.reqs if i < logits.shape[0]]
        if not idxs:
            return logits
        ii = torch.tensor(idxs, dtype=torch.long, device=logits.device)
        lp = torch.log_softmax(logits[ii].float(), dim=-1)
        ent = (-(lp.exp() * lp).sum(-1)).tolist()                          # exact entropy of the distribution the next token is drawn from
        supp = []
        for k, i in enumerate(idxs):
            r = self.reqs[i]; n = len(r.out)
            t = r.out[-1] if n else -1
            decide = False
            if n == 0 and r.s0 is not None and self.mode == "probe":
                r.n_boundary += 1; b = 0; r.bins_b[0] += 1
                decide = r.s0 < self.tau_p                  # DEV continuations: s0 = teacher-forced probe P(error) at the prefix's last token
                r.carry = decide
                if decide:
                    r.n_supp += 1; r.bins_s[0] += 1
            elif n == 0 and r.s0 is not None:      # DEV continuations: the prefix ended at a sentence end; its mean entropy s0 is supplied (teacher-forced, same quantity)
                r.n_boundary += 1; b = 0; r.bins_b[0] += 1
                decide = (self.mode == "nowait") or (self.mode == "cw" and r.s0 < self.tau)
                r.carry = decide
                if decide:
                    r.n_supp += 1; r.bins_s[0] += 1
            if n > 0:
                if t in self.M and r.last_seen == 1:
                    r.n_marker_after += 1
                is_dec = (t in self.B) and not (t in self.dot_ids and n >= 2 and r.out[-2] in self.D)
                if is_dec:
                    s = (r.ent_sum / r.n_sent) if r.n_sent else None
                    r.ent_sum = 0.0; r.n_sent = 0
                    r.n_boundary += 1; b = self._bin(n); r.bins_b[b] += 1
                    if self.mode == "nowait":
                        decide = True
                    elif self.mode == "cw":
                        decide = s is not None and s < self.tau
                    elif self.mode == "random":
                        decide = bool(torch.rand((), generator=self.gen).item() < self.p_bins[b])
                    elif self.mode == "late":
                        decide = n >= self.t_late
                    elif self.mode == "soft":
                        decide = True
                    elif self.mode == "probe":
                        import exp.cw.probe_state as PS
                        decide = PS.S is not None and k < PS.S.shape[0] and float(PS.S[i]) < self.tau_p
                    r.carry = decide; r.last_seen = 1
                    if decide:
                        r.n_supp += 1; r.bins_s[b] += 1
                elif t in self.C and r.carry:
                    decide = True; r.last_seen = 1                           # 'But' + wait: keep the decision for one more position
                else:
                    r.carry = False; r.last_seen = 0
            r.ent_sum += ent[k]; r.n_sent += 1
            if decide and self.mode != "sham":
                supp.append(i)
        if supp:
            rows = torch.tensor(supp, dtype=torch.long, device=logits.device)
            if self.mode == "soft":
                logits[rows[:, None], self.marker[None, :]] -= self.lam
            else:
                logits[rows[:, None], self.marker[None, :]] = float("-inf")
        if self.steps % 200 == 0:
            self._dump()
        return logits
