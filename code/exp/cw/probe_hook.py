"""D51 plumbing: a forward hook on decoder layer L inside the vLLM worker (UniProc executor => same process as the EngineCore and the logits processors) that turns the residual-stream state at the LAST scheduled token of every request into a
logistic-probe score  S = sigmoid(w . x + b)  (x = hidden + residual, fp32), aligned with the logits rows of that step.  Requires enforce_eager=True (a compiled/cudagraph model would skip Python hooks).
    llm.collective_rpc(probe_hook.install, args=(layer_index, w_list_or_None, bias))"""
import torch


def install(worker, L, w=None, b=0.0):
    from vllm.forward_context import get_forward_context

    import exp.cw.probe_state as PS
    model = worker.model_runner.get_model()
    layers = model.model.layers
    D = model.config.hidden_size
    PS.w = (torch.tensor(w, dtype=torch.float32, device="cuda") if w is not None else torch.randn(D, device="cuda") / D ** 0.5); PS.b = float(b)

    def hook(mod, inp, out):
        h, r = (out if isinstance(out, tuple) else (out, None))
        x = h + r if r is not None else h
        md = get_forward_context().attn_metadata
        m = next(iter(md.values())) if isinstance(md, dict) else md
        qsl = m.query_start_loc
        idx = qsl[1:] - 1
        PS.S = torch.sigmoid(x[idx].float() @ PS.w + PS.b)
    layers[L].register_forward_hook(hook)
    return {"layer": L, "hidden": D, "n_layers": len(layers)}


class WorkerExt:
    """worker_extension_cls for LLM(...): llm.collective_rpc('install_probe', args=(layer, w, b)) -- no pickled callables, so no insecure serialization is needed."""

    def install_probe(self, L, w=None, b=0.0):
        return install(self, L, w, b)
