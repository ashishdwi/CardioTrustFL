"""Data loading and federated partitioning for CardioTrustFL experiments.

Scenarios
---------
natural : four real institutions of the UCI Heart Disease collection
          (Hungary, Switzerland, VA Long Beach, Cleveland). Site membership is
          recovered from the row order of the combined 1190-record file
          (rows 0-293 Hungary, 294-416 Switzerland, 417-616 VA, 617-919
          Cleveland, 920-1189 Statlog). Statlog is dropped because 237 of its
          270 rows duplicate Cleveland records.
cleveland_dirichlet : the conference-version setting (Cleveland only,
          80/20 split with random_state=42, Dirichlet alpha=0.5 over 5 clients,
          seed 42) -- reproduces the H4 = 1:88 hospital.
pooled_dirichlet : the 920 natural-site records pooled and re-partitioned with
          a Dirichlet label prior; used for scaling (K clients, alpha).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

SITE_ROWS = {
    "Hungary": (0, 294),
    "Switzerland": (294, 417),
    "VA": (417, 617),
    "Cleveland": (617, 920),
}


@dataclass
class Client:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    @property
    def n(self):
        return len(self.y_train)

    @property
    def pos(self):
        return int(self.y_train.sum())

    @property
    def neg(self):
        return int(len(self.y_train) - self.y_train.sum())


@dataclass
class Federation:
    clients: list
    feature_names: list
    scenario: str
    meta: dict = field(default_factory=dict)

    def pooled(self, part):
        X = np.concatenate([getattr(c, f"X_{part}") for c in self.clients])
        y = np.concatenate([getattr(c, f"y_{part}") for c in self.clients])
        return X, y


# ---------------------------------------------------------------------------
# preprocessing for the 11-feature combined file
# ---------------------------------------------------------------------------
def _encode_combined(df: pd.DataFrame):
    df = df.copy()
    df = df.rename(columns={
        "chest pain type": "cp", "resting bp s": "trestbps", "cholesterol": "chol",
        "fasting blood sugar": "fbs", "resting ecg": "restecg",
        "max heart rate": "thalach", "exercise angina": "exang",
        "ST slope": "slope"})
    # zeros are missing-value codes in this file
    df["chol_missing"] = (df["chol"] == 0).astype(float)
    df.loc[df["chol"] == 0, "chol"] = np.nan
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
    cat = {"cp": [1, 2, 3, 4], "restecg": [0, 1, 2], "slope": [1, 2, 3]}
    for col, levels in cat.items():
        for lv in levels[1:]:
            df[f"{col}_{lv}"] = (df[col] == lv).astype(float)
        df = df.drop(columns=col)
    y = df.pop("target").values.astype(np.float32)
    return df, y


def _federated_standardise(train_frames, all_frames, num_cols):
    """Mean/std computed from per-site sufficient statistics (sum, sum of
    squares, count) of the *training* partitions only -- a one-shot aggregate
    that reveals no individual record. Missing numerics are imputed with the
    federated mean."""
    s = {c: 0.0 for c in num_cols}
    ss = {c: 0.0 for c in num_cols}
    cnt = {c: 0 for c in num_cols}
    for f in train_frames:
        for c in num_cols:
            v = f[c].dropna().values
            s[c] += v.sum(); ss[c] += (v ** 2).sum(); cnt[c] += len(v)
    mu = {c: s[c] / cnt[c] for c in num_cols}
    sd = {c: np.sqrt(max(ss[c] / cnt[c] - mu[c] ** 2, 1e-12)) for c in num_cols}
    out = []
    for f in all_frames:
        f = f.copy()
        for c in num_cols:
            f[c] = (f[c].fillna(mu[c]) - mu[c]) / sd[c]
        out.append(f.values.astype(np.float32))
    return out


def _split_site(X, y, seed, val=0.2, test=0.2):
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=test, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=val / (1 - test), random_state=seed, stratify=y_tmp)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def load_natural(seed: int = 0) -> Federation:
    raw = pd.read_csv(os.path.join(DATA_DIR, "heart_combined_1190.csv"))
    df, y = _encode_combined(raw)
    num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    splits = {}
    for name, (a, b) in SITE_ROWS.items():
        idx = np.arange(a, b)
        tr, va, te = _split_idx(y[idx], seed)
        splits[name] = (idx[tr], idx[va], idx[te])
    train_frames = [df.iloc[s[0]] for s in splits.values()]
    clients = []
    for name, (tr, va, te) in splits.items():
        Xtr, Xva, Xte = _federated_standardise(
            train_frames, [df.iloc[tr], df.iloc[va], df.iloc[te]], num_cols)
        clients.append(Client(name, Xtr, y[tr], Xva, y[va], Xte, y[te]))
    return Federation(clients, list(df.columns), "natural")


def _split_idx(y, seed, val=0.2, test=0.2):
    idx = np.arange(len(y))
    tmp, te = train_test_split(idx, test_size=test, random_state=seed, stratify=y)
    tr, va = train_test_split(tmp, test_size=val / (1 - test), random_state=seed,
                              stratify=y[tmp])
    return tr, va, te


def dirichlet_split(y, num_clients, alpha, seed):
    """Identical to the conference code (label-wise Dirichlet proportions)."""
    np.random.seed(seed)
    client_indices = [[] for _ in range(num_clients)]
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        np.random.shuffle(c_idx)
        props = np.random.dirichlet([alpha] * num_clients)
        props = (props * len(c_idx)).astype(int)
        props[-1] = len(c_idx) - props[:-1].sum()
        start = 0
        for cid, cnt in enumerate(props):
            client_indices[cid].extend(c_idx[start:start + cnt].tolist())
            start += cnt
    return [np.array(ix, dtype=int) for ix in client_indices]


def _client_from_pool(name, X, y, rng_seed):
    """Split a Dirichlet client's records 60/20/20 (stratified when possible)."""
    n = len(y)
    idx = np.arange(n)
    rng = np.random.RandomState(rng_seed)
    strat = y if min(y.sum(), n - y.sum()) >= 3 else None
    try:
        tmp, te = train_test_split(idx, test_size=0.2, random_state=rng_seed, stratify=strat)
        strat2 = y[tmp] if strat is not None else None
        tr, va = train_test_split(tmp, test_size=0.25, random_state=rng_seed, stratify=strat2)
    except ValueError:
        rng.shuffle(idx)
        a, b = int(0.6 * n), int(0.8 * n)
        tr, va, te = idx[:a], idx[a:b], idx[b:]
    return Client(name, X[tr], y[tr], X[va], y[va], X[te], y[te])


def load_pooled_dirichlet(num_clients=10, alpha=0.5, seed=0, min_size=12) -> Federation:
    nat = load_natural(seed)
    X = np.concatenate([np.concatenate([c.X_train, c.X_val, c.X_test]) for c in nat.clients])
    y = np.concatenate([np.concatenate([c.y_train, c.y_val, c.y_test]) for c in nat.clients])
    s, best = seed, None
    for attempt in range(300):  # resample until every client has a usable amount of data
        parts = dirichlet_split(y, num_clients, alpha, seed + 1000 * attempt)
        if best is None or min(len(p) for p in parts) > min(len(p) for p in best[1]):
            best = (seed + 1000 * attempt, parts)
        if min(len(p) for p in parts) >= min_size:
            break
    s, parts = best
    clients = [_client_from_pool(f"C{i+1}", X[p], y[p], seed + i) for i, p in enumerate(parts)]
    return Federation(clients, nat.feature_names, "pooled_dirichlet",
                      {"alpha": alpha, "K": num_clients, "partition_seed": s})


def load_cleveland_dirichlet(seed: int = 0) -> Federation:
    """Conference setting. Partition fixed (seed 42) as in the original code;
    the `seed` argument only changes the within-client train/val split and
    model initialisation. The global test set is the original 61-patient
    hold-out (never used for model selection here)."""
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(os.path.join(DATA_DIR, "cleveland_303.csv"))
    X = df.drop(columns=["target"]).values
    y = df["target"].values.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(X_tr)  # same as conference code (fit on training pool)
    X_tr = sc.transform(X_tr).astype(np.float32); X_te = sc.transform(X_te).astype(np.float32)
    parts = dirichlet_split(y_tr, 5, 0.5, 42)
    clients = []
    for i, p in enumerate(parts):
        Xi, yi = X_tr[p], y_tr[p]
        n = len(yi)
        idx = np.arange(n)
        rng = np.random.RandomState(seed + i)
        rng.shuffle(idx)
        k = int(0.8 * n)
        tr, va = idx[:k], idx[k:]
        clients.append(Client(f"H{i+1}", Xi[tr], yi[tr], Xi[va], yi[va],
                              np.zeros((0, X.shape[1]), np.float32), np.zeros(0, np.float32)))
    fed = Federation(clients, list(df.columns[:-1]), "cleveland_dirichlet")
    fed.meta["global_test"] = (X_te, y_te)
    return fed


def describe(fed: Federation):
    rows = []
    for c in fed.clients:
        rows.append(dict(site=c.name, n_train=c.n, pos=c.pos, neg=c.neg,
                         n_val=len(c.y_val), n_test=len(c.y_test),
                         pos_rate=round(float(np.mean(np.concatenate([c.y_train, c.y_val, c.y_test]))), 3)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(describe(load_natural(0)))
    print(describe(load_cleveland_dirichlet(0)))
    f = load_pooled_dirichlet(10, 0.5, 0)
    print(describe(f), f.meta)
