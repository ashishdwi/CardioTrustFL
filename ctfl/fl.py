"""Federated training core: model, DP-SGD local training (per-sample clipping,
Poisson sampling, RDP accounting), FedAvg / FedProx / SCAFFOLD local
objectives, deterministic aggregation rules, and evaluation."""
from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap
from opacus.accountants import RDPAccountant
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class HeartNet(nn.Module):
    """Same MLP family as the conference version (64-32-16) without dropout/BN
    (both interfere with per-sample DP gradients)."""

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).double().numpy()


def load_flat(model, vec):
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            k = p.numel()
            p.copy_(torch.from_numpy(vec[i:i + k].reshape(p.shape)).float())
            i += k


def canonical_hash(vec: np.ndarray) -> str:
    """SHA3-256 of the canonical little-endian float32 serialisation."""
    return hashlib.sha3_256(np.asarray(vec, dtype="<f4").tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def make_loss(kind, gamma=2.0, alpha=0.25):
    def bce(logit, y):
        return nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none")

    if kind == "bce":
        return lambda logit, y: bce(logit, y)
    if kind == "focal":
        def fl(logit, y):
            p = torch.sigmoid(logit)
            pt = y * p + (1 - y) * (1 - p)
            a_t = y * alpha + (1 - y) * (1 - alpha)
            return a_t * (1 - pt) ** gamma * bce(logit, y)
        return fl
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    rounds: int = 30
    local_epochs: int = 1
    lr: float = 0.1
    batch: int = 8                # expected (Poisson) batch size
    dp: bool = True
    sigma: float = 1.5            # noise multiplier
    clip: float = 0.5             # per-sample clipping norm C
    delta: float = 1e-5
    loss: str = "bce"             # bce | focal
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    class_balanced_loss: bool = False  # weight samples by inverse local class freq
    local_algo: str = "fedavg"    # fedavg | fedprox | scaffold
    mu: float = 0.01              # FedProx proximal coefficient
    agg: str = "balance"          # fedavg | uniform | balance | median | trimmed | krum
    weight_cap: float = 2.0       # balance weights capped at cap * median (None = off)
    filt: str = "none"            # none | cos_fixed | cos_mad
    cos_tau: float = 0.3
    mad_k: float = 3.0
    pvs_k: float = 2.0
    vote: bool = False            # round-level validation vote with rollback
    vote_tol: float = 0.02        # accept if val loss rises by < tol (relative)
    vote_threshold: float = 2 / 3
    norm_clip: bool = False       # clip each update to the median update norm
    trim_f: int = 1
    krum_f: int = 1
    seed: int = 0


# ---------------------------------------------------------------------------
# local training
# ---------------------------------------------------------------------------
class LocalTrainer:
    def __init__(self, d, cfg: Config):
        self.cfg = cfg
        self.model = HeartNet(d)
        self.loss = make_loss(cfg.loss, cfg.focal_gamma, cfg.focal_alpha)
        names = [n for n, _ in self.model.named_parameters()]
        self.names = names

        def sample_loss(params, x, y, w):
            logit = functional_call(self.model, params, (x.unsqueeze(0),))
            return w * self.loss(logit, y.unsqueeze(0)).squeeze()

        self.per_sample_grad = vmap(grad(sample_loss), in_dims=(None, 0, 0, 0))

    def run(self, w_global, client, rng, accountant=None, c_global=None, c_local=None,
            flip_labels=False):
        cfg = self.cfg
        load_flat(self.model, w_global)
        params = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        X = torch.from_numpy(client.X_train).float()
        y = torch.from_numpy(client.y_train).float()
        if flip_labels:
            y = 1 - y
        n = len(y)
        if cfg.class_balanced_loss:
            pos = max(float(y.sum()), 1.0); neg = max(float(n - y.sum()), 1.0)
            sw = torch.where(y > 0.5, n / (2 * pos), n / (2 * neg))
        else:
            sw = torch.ones(n)
        q = min(cfg.batch / n, 1.0)
        steps = max(1, int(round(cfg.local_epochs / q)))
        wg = {k: v.clone() for k, v in params.items()}
        corr = None
        if cfg.local_algo == "scaffold" and c_global is not None:
            corr = _unflatten(c_global - c_local, params)
        for _ in range(steps):
            if cfg.dp:
                mask = torch.from_numpy(rng.rand(n) < q)
                idx = torch.nonzero(mask).squeeze(-1)
                g = {k: torch.zeros_like(v) for k, v in params.items()}
                if len(idx) > 0:
                    gs = self.per_sample_grad(params, X[idx], y[idx], sw[idx])
                    norms = torch.sqrt(sum((gs[k].reshape(len(idx), -1) ** 2).sum(1) for k in gs))
                    factor = torch.clamp(cfg.clip / (norms + 1e-12), max=1.0)
                    for k in gs:
                        g[k] = (gs[k] * factor.view(-1, *[1] * (gs[k].dim() - 1))).sum(0)
                gen = torch.Generator().manual_seed(int(rng.randint(2 ** 31)))
                for k in g:
                    g[k] = (g[k] + torch.randn(g[k].shape, generator=gen) * cfg.sigma * cfg.clip) / (q * n)
                if accountant is not None:
                    accountant.step(noise_multiplier=cfg.sigma, sample_rate=q)
            else:
                bs = min(cfg.batch, n)
                idx = torch.from_numpy(rng.choice(n, bs, replace=False))
                gs = self.per_sample_grad(params, X[idx], y[idx], sw[idx])
                g = {k: gs[k].mean(0) for k in gs}
            # data-independent terms are added after privatisation (post-processing)
            for k in params:
                upd = g[k]
                if cfg.local_algo == "fedprox":
                    upd = upd + cfg.mu * (params[k] - wg[k])
                if corr is not None:
                    upd = upd + corr[k]
                params[k] = params[k] - cfg.lr * upd
        w_new = torch.cat([params[k].reshape(-1) for k in self.names]).double().numpy()
        delta = w_new - w_global
        new_c_local = None
        if cfg.local_algo == "scaffold" and c_global is not None:
            new_c_local = c_local - c_global - delta / (steps * cfg.lr)
        return delta, steps, new_c_local


def _unflatten(vec, like):
    out, i = {}, 0
    for k, v in like.items():
        m = v.numel()
        out[k] = torch.from_numpy(vec[i:i + m].reshape(v.shape)).float()
        i += m
    return out


# ---------------------------------------------------------------------------
# deterministic aggregation (executed identically by every hospital)
# ---------------------------------------------------------------------------
def cosine_distances(deltas):
    D = np.stack(deltas)
    med = np.median(D, axis=0)
    mn = np.linalg.norm(med) + 1e-12
    return np.array([1 - float(d @ med) / (np.linalg.norm(d) * mn + 1e-12) for d in D])


def apply_filter(deltas, cfg: Config):
    n = len(deltas)
    if cfg.filt == "none" or n <= 2:
        return list(range(n)), np.zeros(n)
    if cfg.filt == "mkrum":
        # Multi-Krum selection (Blanchard et al. 2017): keep the n - f updates
        # with the smallest sum of squared distances to their n - f - 2 nearest
        # neighbours. Euclidean (not cosine) so that norm inflation is caught.
        D = np.stack(deltas)
        f = cfg.krum_f
        k = max(1, n - f - 2)
        dist2 = ((D[:, None, :] - D[None, :, :]) ** 2).sum(-1)
        scores = np.array([np.sort(dist2[i][np.arange(n) != i])[:k].sum() for i in range(n)])
        keep = sorted(np.argsort(scores, kind="stable")[: n - f].tolist())
        return keep, scores
    d = cosine_distances(deltas)
    if cfg.filt == "cos_fixed":
        keep = [i for i in range(n) if d[i] <= cfg.cos_tau]
    elif cfg.filt == "cos_mad":
        med = np.median(d)
        mad = np.median(np.abs(d - med)) * 1.4826 + 1e-6
        keep = [i for i in range(n) if d[i] <= med + cfg.mad_k * mad]
    else:
        raise ValueError(cfg.filt)
    if len(keep) == 0:
        keep = [int(np.argmin(d))]
    return keep, d


def val_loss(vec, X, y, d):
    p = np.clip(predict(vec, X, d), 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def pvs_votes(deltas, ctx, cfg):
    """Peer-validation scoring. Every hospital j evaluates every revealed
    update i on its PRIVATE validation set: s_ij = L_j(w) - L_j(w + D_i).
    Hospital j votes 'accept' for i iff its robust z-score
    (s_ij - med_j) / MAD_j >= -k  or  s_ij >= 0. Only the vote bit is
    published. Malicious hospitals vote accept for colluders and reject
    for everybody else (worst case). Returns the K x K bit matrix."""
    w, fed, d, mal = ctx["w"], ctx["fed"], ctx["d"], ctx.get("attackers", set())
    n = len(deltas)
    bits = np.zeros((n, n), dtype=int)   # bits[j, i] = voter j accepts update i
    for j, c in enumerate(fed.clients):
        if j in mal:
            bits[j] = [1 if i in mal else 0 for i in range(n)]
            continue
        if len(c.y_val) == 0:
            bits[j] = 1
            continue
        base = val_loss(w, c.X_val, c.y_val, d)
        s = np.array([base - val_loss(w + dl, c.X_val, c.y_val, d) for dl in deltas])
        med = np.median(s)
        mad = np.median(np.abs(s - med)) * 1.4826 + 1e-9
        bits[j] = ((s - med) / mad >= -cfg.pvs_k) | (s >= 0)
    return bits


def apply_pvs(deltas, ctx, cfg):
    bits = pvs_votes(deltas, ctx, cfg)
    n = len(deltas)
    support = bits.sum(0)
    keep = [i for i in range(n) if support[i] * 2 > n]      # strict majority
    if not keep:
        keep = [int(np.argmax(support))]
    return keep, support


def aggregate(deltas, claimed_w, cfg: Config, ctx=None):
    """Returns (aggregate delta, kept indices, cosine distances)."""
    if cfg.filt == "pvs":
        keep, dist = apply_pvs(deltas, ctx, cfg)
    elif cfg.filt == "cos_fixed_full":
        # conference implementation: cosine distance on FULL weight vectors
        dist = cosine_distances([ctx["w"] + dl for dl in deltas])
        keep = [i for i in range(len(deltas)) if dist[i] <= cfg.cos_tau] or [int(np.argmin(dist))]
    else:
        keep, dist = apply_filter(deltas, cfg)
    D = np.stack([deltas[i] for i in keep])
    if cfg.norm_clip:
        norms = np.linalg.norm(D, axis=1)
        ref = np.median(np.linalg.norm(np.stack(deltas), axis=1))
        D = D * np.minimum(1.0, ref / (norms + 1e-12))[:, None]
    w = np.array([claimed_w[i] for i in keep], dtype=np.float64)
    if cfg.agg in ("fedavg", "balance", "uniform"):
        if cfg.agg == "uniform":
            w = np.ones_like(w)
        if cfg.agg == "balance" and cfg.weight_cap:
            all_w = np.array(claimed_w, dtype=np.float64)
            cap = cfg.weight_cap * np.median(all_w)
            w = np.minimum(w, cap)
        agg = (w[:, None] * D).sum(0) / w.sum()
    elif cfg.agg == "median":
        agg = np.median(D, axis=0)
    elif cfg.agg == "trimmed":
        f = min(cfg.trim_f, (len(D) - 1) // 2)
        S = np.sort(D, axis=0)
        agg = S[f:len(D) - f].mean(0) if f > 0 else S.mean(0)
    elif cfg.agg == "krum":
        m = len(D)
        f = cfg.krum_f
        k = max(1, m - f - 2)
        dist2 = ((D[:, None, :] - D[None, :, :]) ** 2).sum(-1)
        scores = [np.sort(dist2[i][np.arange(m) != i])[:k].sum() for i in range(m)]
        agg = D[int(np.argmin(scores))]
    else:
        raise ValueError(cfg.agg)
    return agg, keep, dist


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def predict(model_vec, X, d):
    m = HeartNet(d)
    load_flat(m, model_vec)
    m.eval()
    with torch.no_grad():
        p = torch.sigmoid(m(torch.from_numpy(X).float())).numpy()
    return np.nan_to_num(p, nan=0.5)


def choose_threshold(p, y):
    """Threshold maximising balanced accuracy on VALIDATION data only."""
    best_t, best = 0.5, -1
    for t in np.linspace(0.05, 0.95, 91):
        b = balanced_accuracy_score(y, (p >= t).astype(int)) if len(np.unique(y)) > 1 else 0
        if b > best:
            best, best_t = b, t
    return best_t


def metrics(p, y, t):
    pred = (p >= t).astype(int)
    out = dict(acc=accuracy_score(y, pred), bal_acc=balanced_accuracy_score(y, pred),
               f1=f1_score(y, pred, zero_division=0),
               prec=precision_score(y, pred, zero_division=0),
               rec=recall_score(y, pred, zero_division=0),
               spec=recall_score(1 - y, 1 - pred, zero_division=0))
    out["auc"] = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    return out


def evaluate(model_vec, fed, d, per_site=False):
    Xv, yv = fed.pooled("val")
    pv = predict(model_vec, Xv, d)
    t = choose_threshold(pv, yv)
    val_auc = roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else np.nan
    if "global_test" in fed.meta:
        Xt, yt = fed.meta["global_test"]
    else:
        Xt, yt = fed.pooled("test")
    pt = predict(model_vec, Xt, d)
    res = metrics(pt, yt, t)
    m05 = metrics(pt, yt, 0.5)
    res.update(f1_05=m05["f1"], bal_acc_05=m05["bal_acc"], acc_05=m05["acc"], mean_p=float(pt.mean()))
    res["threshold"] = t
    res["val_auc"] = val_auc
    res["val_bal_acc"] = balanced_accuracy_score(yv, (pv >= t).astype(int))
    if per_site:
        res["sites"] = {}
        for c in fed.clients:
            if len(c.y_test):
                res["sites"][c.name] = metrics(predict(model_vec, c.X_test, d), c.y_test, t)
    return res


# ---------------------------------------------------------------------------
# federated run
# ---------------------------------------------------------------------------
def claimed_weights(fed, cfg):
    if cfg.agg == "balance":
        return [max(min(c.pos, c.neg), 1) for c in fed.clients]
    return [c.n for c in fed.clients]


def run_federated(fed, cfg: Config, attack=None, log_every=0, return_history=False):
    """attack: optional object with .attackers (set of client indices) and
    .craft(round, w_global, honest_deltas_now, honest_deltas_prev, own_benign,
    cfg) -> list of malicious deltas, plus .claim(weights) -> weights."""
    rng = np.random.RandomState(cfg.seed)
    torch.manual_seed(cfg.seed)
    d = fed.clients[0].X_train.shape[1]
    w = flat(HeartNet(d))
    trainers = LocalTrainer(d, cfg)
    accs = [RDPAccountant() for _ in fed.clients]
    K = len(fed.clients)
    c_global = np.zeros_like(w) if cfg.local_algo == "scaffold" else None
    c_local = [np.zeros_like(w) for _ in range(K)] if c_global is not None else [None] * K
    claimed = claimed_weights(fed, cfg)
    if attack is not None:
        claimed = attack.claim(claimed, cfg)
    history = []
    prev_honest = None
    timing = dict(train=0.0, agg=0.0, hash=0.0)
    flags_honest = flags_mal = n_honest_obs = n_mal_obs = 0
    rejected_rounds = 0
    for r in range(cfg.rounds):
        deltas = [None] * K
        t0 = time.perf_counter()
        for i, c in enumerate(fed.clients):
            flip = attack is not None and i in attack.attackers and attack.kind == "labelflip"
            delta, steps, new_c = trainers.run(w, c, rng, accs[i] if cfg.dp else None,
                                               c_global, c_local[i], flip_labels=flip)
            deltas[i] = delta
            if new_c is not None:
                c_local[i] = new_c
        timing["train"] += time.perf_counter() - t0
        if attack is not None and attack.kind != "labelflip":
            honest_idx = [i for i in range(K) if i not in attack.attackers]
            honest_now = [deltas[i] for i in honest_idx]
            own = [deltas[i] for i in sorted(attack.attackers)]
            mal = attack.craft(r, w, honest_now, prev_honest, own, cfg, claimed)
            for j, i in enumerate(sorted(attack.attackers)):
                deltas[i] = mal[j]
            prev_honest = honest_now
        t0 = time.perf_counter()
        hashes = [canonical_hash(dl) for dl in deltas]
        timing["hash"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        ctx = dict(w=w, fed=fed, d=d, attackers=attack.attackers if attack is not None else set())
        agg, keep, dist = aggregate(deltas, claimed, cfg, ctx)
        timing["agg"] += time.perf_counter() - t0
        if attack is not None:
            for i in range(K):
                flagged = i not in keep
                if i in attack.attackers:
                    n_mal_obs += 1; flags_mal += flagged
                else:
                    n_honest_obs += 1; flags_honest += flagged
        if c_global is not None:
            c_global = np.mean(c_local, axis=0)
        accepted = True
        if cfg.vote:
            mal = attack.attackers if attack is not None else set()
            yes = 0
            for j, c in enumerate(fed.clients):
                if j in mal or len(c.y_val) == 0:
                    yes += 1; continue
                before = val_loss(w, c.X_val, c.y_val, d)
                after = val_loss(w + agg, c.X_val, c.y_val, d)
                yes += after <= before * (1 + cfg.vote_tol)
            accepted = yes >= cfg.vote_threshold * K - 1e-9
            rejected_rounds += (not accepted)
        if accepted:
            w = w + agg
        if return_history or (log_every and (r + 1) % log_every == 0):
            ev = evaluate(w, fed, d)
            dist_full = cosine_distances([w - agg + dl for dl in deltas])
            ev.update(round=r + 1, kept=keep, dist=cosine_distances(deltas).tolist(),
                      dist_full=dist_full.tolist())
            history.append(ev)
    final = evaluate(w, fed, d, per_site=True)
    eps = [a.get_epsilon(cfg.delta) if cfg.dp else float("inf") for a in accs]
    final["eps_per_site"] = eps
    final["eps_max"] = float(max(eps))
    final["timing"] = timing
    if attack is not None:
        final["tpr"] = flags_mal / max(n_mal_obs, 1)
        final["fpr"] = flags_honest / max(n_honest_obs, 1)
    final["history"] = history
    final["rejected_rounds"] = rejected_rounds
    final["weights"] = w
    return final
