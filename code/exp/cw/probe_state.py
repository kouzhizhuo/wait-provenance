"""Shared (same-process) state between the forward hook (probe_hook.install) and the logits processor: S = per-request probe score for the current step (aligned with the logits rows)."""
S = None
w = None
b = 0.0
