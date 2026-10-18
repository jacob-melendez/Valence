"""M3: GRU + attention over the raw 15-minute tick sequence (benchmark only; M2 is served).

    python -m valence.models.temporal [--epochs 15] [--device auto]

Same simulator corpus and session split as the LightGBM ladder (from artifacts/model_meta.json).
Writes artifacts/m3.pt, reports/m3_metrics.json (macro-F1 overall and vs history, merged into
reports/metrics.json by evaluate.py) and reports/figs/m3_attention.png ("which minutes mattered").
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from valence.eval.evaluate import HISTORY_TICKS
from valence.features.build_dataset import build
from valence.models.train_ladder import split_sessions
from valence.sim.generator import simulate_corpus

SEQ = 180
CHANNELS = ["ambient_db", "light_lux", "motion_energy", "posture_slump", "head_in_hands_frac",
            "clenched_fists_count", "laughter_count", "compliments_count", "social_withdrawal_flag", "n_people",
            "hours_since_break", "tod_sin", "tod_cos", "face_valence", "face_visible"]


def sequences(df: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
    """Left-padded (N, 180, C + 1) arrays for the windows in ``meta`` (last channel = valid mask)."""
    people = df[df["person_id"] != 0].copy()
    people["tod_sin"] = np.sin(2 * np.pi * people["tod_h"] / 24)
    people["tod_cos"] = np.cos(2 * np.pi * people["tod_h"] / 24)
    out = np.zeros((len(meta), SEQ, len(CHANNELS) + 1), dtype=np.float32)
    by_key = {k: g.sort_values("tick") for k, g in people.groupby(["session_id", "person_id"])}
    cache: dict = {}
    for i, row in enumerate(meta.itertuples(index=False)):
        key = (row.session_id, row.person_id)
        if key not in cache:
            g = by_key[key]
            cache[key] = (g["tick"].to_numpy(), g[CHANNELS].to_numpy(dtype=np.float32))
        ticks, vals = cache[key]
        end = int(np.searchsorted(ticks, row.tick)) + 1
        seg = vals[max(0, end - row.hist_ticks):end]
        out[i, SEQ - len(seg):, :-1] = seg
        out[i, SEQ - len(seg):, -1] = 1.0
    return out


class GRUAttention(nn.Module):
    def __init__(self, n_in: int, n_classes: int, hidden: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.att = nn.Sequential(nn.Linear(hidden, 32), nn.Tanh(), nn.Linear(32, 1))
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = x[..., -1] > 0
        h, _ = self.gru(x)
        scores = self.att(h).squeeze(-1).masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=1)
        return self.head((w.unsqueeze(-1) * h).sum(1)), w


def normalize(X: np.ndarray, stats: tuple[np.ndarray, np.ndarray] | None = None):
    vals = X[..., :-1]
    valid = X[..., -1:] > 0
    if stats is None:
        flat = vals[np.broadcast_to(valid, vals.shape)].reshape(-1, vals.shape[-1])
        mu, sd = np.nanmean(flat, axis=0), np.nanstd(flat, axis=0) + 1e-6
        stats = (mu, sd)
    mu, sd = stats
    z = np.where(valid, (vals - mu) / sd, 0.0)
    z = np.nan_to_num(z, nan=0.0)  # missing readings -> channel mean; the mask channel marks padding only
    return np.concatenate([z, X[..., -1:]], axis=-1).astype(np.float32), stats


def predict(model: nn.Module, X: np.ndarray, device: str, batch: int = 512) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, att = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            lo, w = model(torch.from_numpy(X[i:i + batch]).to(device))
            logits.append(lo.cpu().numpy())
            att.append(w.cpu().numpy())
    return np.concatenate(logits), np.concatenate(att)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--step", type=int, default=12)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    t0 = time.time()
    meta_json = json.loads((Path(args.artifacts) / "model_meta.json").read_text())
    states = meta_json["states"]
    s2i = {s: i for i, s in enumerate(states)}
    df = simulate_corpus(meta_json["sessions"], meta_json["seed"], state_set=meta_json["state_set"],
                         noise_scale=meta_json["noise_scale"])
    _, full_meta = build(df, seed=meta_json["seed"])
    split = split_sessions(full_meta, meta_json["seed"])
    groups = {k: set(full_meta.loc[getattr(split, k), "session_id"]) for k in ("train", "early", "calib", "test")}
    _, meta = build(df, step=args.step, seed=meta_json["seed"] + 1)  # random-length windows, like the ladder
    y = meta["label"].map(s2i).to_numpy()
    part = {k: meta["session_id"].isin(v).to_numpy() for k, v in groups.items()}
    X_raw = sequences(df, meta)
    X_train, stats = normalize(X_raw[part["train"]])
    X_early, _ = normalize(X_raw[part["early"]], stats)
    X_test, _ = normalize(X_raw[part["test"]], stats)
    print(f"sequences: {X_raw.shape} in {time.time() - t0:.0f}s on {device}")

    model = GRUAttention(X_train.shape[-1], len(states)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    cw = np.bincount(y[part["train"]], minlength=len(states)).astype(float)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw.sum() / (len(states) * np.maximum(cw, 1)), dtype=torch.float32).to(device))
    best, best_state, patience = -1.0, None, 0
    yt = torch.from_numpy(y[part["train"]]).long()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            logits, _ = model(torch.from_numpy(X_train[idx.numpy()]).to(device))
            loss = loss_fn(logits, yt[idx].to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        f1 = f1_score(y[part["early"]], predict(model, X_early, device)[0].argmax(1), average="macro")
        print(f"  epoch {epoch + 1}: early-stop macro-F1 {f1:.3f} ({time.time() - t0:.0f}s)")
        if f1 > best + 1e-3:
            best, best_state, patience = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 3:
                break
    model.load_state_dict(best_state)
    logits, att = predict(model, X_test, device)
    test_f1 = float(f1_score(y[part["test"]], logits.argmax(1), average="macro"))

    # history curve on the same windows evaluate.py uses
    test_df = df[df["session_id"].isin(groups["test"])]
    curve = []
    for minutes, ticks in HISTORY_TICKS.items():
        _, m = build(test_df, step=6, seed=11, fixed_len=ticks)
        m = m[m["hist_ticks"] == ticks].reset_index(drop=True)
        Xh, _ = normalize(sequences(df, m), stats)
        pred = predict(model, Xh, device)[0].argmax(1)
        curve.append({"minutes": minutes, "macro_f1": round(float(f1_score(m["label"].map(s2i), pred, average="macro")), 4)})

    # attention by minutes-ago for full-length windows, per true class
    full = X_test[:, 0, -1] > 0
    att_by_class = {s: att[full & (y[part["test"]] == i)].mean(axis=0).tolist()
                    for s, i in s2i.items() if (full & (y[part["test"]] == i)).any()}
    reports = Path(args.reports)
    (reports / "figs").mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "stats": [s.tolist() for s in stats], "channels": CHANNELS, "states": states},
               Path(args.artifacts) / "m3.pt")
    out = {"model": "m3_gru_attention", "device": device, "test_macro_f1": round(test_f1, 4),
           "early_stop_macro_f1": round(best, 4), "accuracy_vs_history": curve, "attention_by_class": att_by_class,
           "train_windows": int(part["train"].sum()), "seconds": round(time.time() - t0)}
    (reports / "m3_metrics.json").write_text(json.dumps(out))
    from valence.eval import style

    style.attention_chart(att_by_class, reports / "figs" / "m3_attention.png")
    print(json.dumps({k: v for k, v in out.items() if k != "attention_by_class"}, indent=1))


if __name__ == "__main__":
    main()
