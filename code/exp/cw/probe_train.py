"""D51 probe training (pre-specified: ONE layer, L2 logistic regression C = 1.0, standardised features).  Target: erroneous (1) vs correct (0) boundary state.
Training rows: DEV-A (even pid) injected-error pairs (clean=0 / corrupted=1; pm1+swap families only: sign-flip held out as in D50a) [+ natural steps from PRM labels, --natural file with the same npz layout and y].  Evaluation: DEV-B (odd pid) AUROC overall and on the held-out sign-flip family.
Outputs <out>.json with w, b (for the processor hook, standardisation folded in), mu/sd, auroc, and writes scores jsonl (prefix_id, s_ent := P(error)) so that exp.cw.dev_calibrate / dev_cont can reuse the D50f tau rule with 'suppress iff P(error) < tau_p'.
    python -m exp.cw.probe_train --feats feats.npz --out probe.json --scores_out scores_probe.jsonl"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--feats", required=True); ap.add_argument("--natural"); ap.add_argument("--out", required=True); ap.add_argument("--scores_out", required=True); a = ap.parse_args()
    d = np.load(a.feats); X = d["X"].astype(np.float32); y = (d["src"] == "corrupted").astype(int); pid = d["pid"]; kind = d["kind"]
    trA = (pid % 2 == 0) & ((kind == "pm1") | (kind == "swap") | (d["src"] == "clean"))
    Xt, yt = X[trA], y[trA]
    if a.natural:
        n = np.load(a.natural); Xt = np.concatenate([Xt, n["X"].astype(np.float32)]); yt = np.concatenate([yt, n["y"]])
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit((Xt - mu) / sd, yt)
    p = clf.predict_proba((X - mu) / sd)[:, 1]
    B = pid % 2 == 1
    res = {"n_train": int(len(yt)), "auroc_B_all": float(roc_auc_score(y[B], p[B])), "auroc_B_signflip": float(roc_auc_score(y[B & ((kind == "sign") | (d["src"] == "clean"))], p[B & ((kind == "sign") | (d["src"] == "clean"))])),
           "auroc_A_train": float(roc_auc_score(y[trA], p[trA]))}
    w = clf.coef_[0] / sd; b = float(clf.intercept_[0] - (clf.coef_[0] * mu / sd).sum())
    res.update({"w": w.tolist(), "b": b})
    json.dump(res, open(a.out, "w"))
    with open(a.scores_out, "w") as f:
        for i in range(len(p)):
            f.write(json.dumps({"prefix_id": str(d["prefix_id"][i]), "pid": int(pid[i]), "src": str(d["src"][i]), "kind": str(kind[i]), "s_ent": float(p[i])}) + "\n")
    print({k: v for k, v in res.items() if k not in ("w",)})


if __name__ == "__main__":
    main()
