"""E1 (final utility table), E3 (filter calibration under DP and heterogeneity),
E4 (privacy-utility), E5 (scaling). Usage: python exp_extra.py e1|e3|e4|e5"""
import json, sys, warnings
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings("ignore")
from ctfl.attacks import Attack
from ctfl.data import Client, Federation, load_cleveland_dirichlet, load_natural, load_pooled_dirichlet
from ctfl.fl import Config, evaluate, run_federated

DP_LR, NODP_LR = 0.1, 0.05
CTFL = dict(agg="uniform", filt="cos_mad", norm_clip=True)


def strip(r):
    r = dict(r)
    for k in ("weights", "history"):
        r.pop(k, None)
    return r


# ----------------------------------------------------------------- E1
E1_METHODS = {
    "FedAvg": dict(agg="fedavg"),
    "FedProx": dict(agg="fedavg", local_algo="fedprox", mu=0.1),
    "SCAFFOLD": dict(agg="fedavg", local_algo="scaffold"),
    "Uniform": dict(agg="uniform"),
    "Balance-weighted": dict(agg="balance", weight_cap=1.0),
    "Median": dict(agg="median"),
    "Trimmed mean": dict(agg="trimmed"),
    "Krum": dict(agg="krum"),
    "CardioTrustFL": CTFL,
}


def e1_job(a):
    scen, name, over, dp, seed, rounds = a
    fed = load_natural(seed) if scen == "natural" else load_cleveland_dirichlet(seed)
    d = fed.clients[0].X_train.shape[1]
    cfg = Config(dp=dp, lr=DP_LR if dp else NODP_LR, seed=seed, rounds=rounds, **over)
    if name == "Centralized":
        pooled = Client("pooled", np.concatenate([c.X_train for c in fed.clients]),
                        np.concatenate([c.y_train for c in fed.clients]), *fed.pooled("val"), *fed.pooled("test"))
        rr = run_federated(Federation([pooled], fed.feature_names, scen, dict(fed.meta)), replace(cfg, agg="fedavg"))
        r = evaluate(rr["weights"], fed, d, per_site=True); r["eps_max"] = rr["eps_max"]
    elif name == "Local-only":
        res = []
        for c in fed.clients:
            rr = run_federated(Federation([c], fed.feature_names, scen, dict(fed.meta)), replace(cfg, agg="fedavg"))
            ev = evaluate(rr["weights"], fed, d, per_site=True); ev["eps_max"] = rr["eps_max"]
            res.append(ev)
        r = {k: float(np.mean([x[k] for x in res])) for k in
             ("acc", "bal_acc", "f1", "auc", "val_auc", "eps_max", "f1_05", "bal_acc_05")}
        r["per_local_model"] = [{k: x[k] for k in ("acc", "bal_acc", "f1", "auc")} for x in res]
    else:
        r = strip(run_federated(fed, cfg))
    r.update(scenario=scen, method=name, dp=dp, seed=seed, rounds=rounds)
    return r


def e1():
    jobs = []
    for scen in ["natural", "cleveland"]:
        for dp in [True, False]:
            for seed in range(5):
                for name, over in E1_METHODS.items():
                    jobs.append((scen, name, over, dp, seed, 30))
                jobs.append((scen, "Centralized", {}, dp, seed, 30))
                jobs.append((scen, "Local-only", {}, dp, seed, 30))
        for seed in range(5):  # calibration study at 10 rounds (conference setting)
            for name in ["FedAvg", "Uniform", "Balance-weighted", "CardioTrustFL"]:
                jobs.append((scen, name, E1_METHODS[name], True, seed, 10))
    # loss ablation
    for scen in ["natural", "cleveland"]:
        for seed in range(5):
            for lname, lo in {"BCE": dict(loss="bce"), "Focal a=0.25": dict(loss="focal", focal_alpha=0.25),
                              "Focal a=0.75": dict(loss="focal", focal_alpha=0.75),
                              "Class-bal. BCE": dict(class_balanced_loss=True)}.items():
                for agg in ["fedavg", "balance"]:
                    jobs.append((scen, f"loss:{lname}|agg:{agg}", dict(lo, agg=agg, weight_cap=1.0), True, seed, 30))
    with Pool(2) as p:
        out = p.map(e1_job, jobs, chunksize=1)
    json.dump(out, open("results/e1_final.json", "w"), default=float, indent=1)


# ----------------------------------------------------------------- E3
SIGMAS = [None, 0.5, 1.0, 1.5, 2.0, 3.0]


def e3_job(a):
    kind, scen, sigma, seed, extra = a
    fed = load_natural(seed) if scen == "natural" else load_pooled_dirichlet(10, 1.0, seed)
    dp = sigma is not None
    base = dict(dp=dp, sigma=sigma or 1.0, lr=DP_LR if dp else NODP_LR, seed=seed)
    if kind == "honest":
        r = run_federated(fed, Config(agg="uniform", **base), return_history=True)
        H = r["history"]
        return dict(kind=kind, scenario=scen, sigma=sigma, seed=seed,
                    dist=[h["dist"] for h in H], dist_full=[h["dist_full"] for h in H])
    filt, site = extra
    over = {"cos_fixed": dict(filt="cos_fixed", cos_tau=0.3),
            "cos_fixed_full": dict(filt="cos_fixed_full", cos_tau=0.3),
            "cos_mad": dict(filt="cos_mad")}[filt]
    r = run_federated(fed, Config(agg="uniform", **base, **over), attack=Attack("signflip", [site], "full", len(fed.clients), seed))
    return dict(kind=kind, scenario=scen, sigma=sigma, seed=seed, filt=filt, site=site,
                tpr=r["tpr"], fpr=r["fpr"], auc=r["auc"])


def e3():
    jobs = []
    for scen in ["natural", "pooled"]:
        for s in SIGMAS:
            for seed in range(3):
                jobs.append(("honest", scen, s, seed, None))
    for s in SIGMAS:
        for seed in range(2):
            for site in range(4):
                for f in ["cos_fixed", "cos_fixed_full", "cos_mad"]:
                    jobs.append(("detect", "natural", s, seed, (f, site)))
    with Pool(2) as p:
        out = p.map(e3_job, jobs, chunksize=1)
    json.dump(out, open("results/e3_calibration.json", "w"), default=float)


# ----------------------------------------------------------------- E4
def e4_job(a):
    method, sigma, seed = a
    fed = load_natural(seed)
    over = CTFL if method == "CardioTrustFL" else dict(agg="fedavg")
    dp = sigma is not None
    r = run_federated(fed, Config(dp=dp, sigma=sigma or 1.0, lr=DP_LR if dp else NODP_LR, seed=seed, **over))
    return dict(method=method, sigma=sigma, seed=seed, auc=r["auc"], bal_acc=r["bal_acc"], f1=r["f1"],
                acc=r["acc"], eps=r["eps_per_site"], eps_max=r["eps_max"])


def e4():
    jobs = [(m, s, seed) for m in ["CardioTrustFL", "FedAvg"] for s in [None, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
            for seed in range(5)]
    with Pool(2) as p:
        out = p.map(e4_job, jobs, chunksize=1)
    json.dump(out, open("results/e4_privacy.json", "w"), default=float)


# ----------------------------------------------------------------- E5
def e5_job(a):
    K, alpha, method, seed = a
    fed = load_pooled_dirichlet(K, alpha, seed, min_size=10)
    over = {"FedAvg": dict(agg="fedavg"), "Balance-weighted": dict(agg="balance", weight_cap=1.0),
            "CardioTrustFL": CTFL}[method]
    r = run_federated(fed, Config(seed=seed, **over))
    return dict(K=K, alpha=alpha, method=method, seed=seed, auc=r["auc"], bal_acc=r["bal_acc"],
                f1=r["f1"], eps_max=r["eps_max"], train_s_per_round=r["timing"]["train"] / 30,
                agg_s_per_round=r["timing"]["agg"] / 30, hash_s_per_round=r["timing"]["hash"] / 30)


def e5():
    jobs = [(K, a, m, s) for K in [5, 10, 20] for a in [0.3, 0.5, 1.0]
            for m in ["FedAvg", "Balance-weighted", "CardioTrustFL"] for s in range(3)]
    with Pool(2) as p:
        out = p.map(e5_job, jobs, chunksize=1)
    json.dump(out, open("results/e5_scaling.json", "w"), default=float)


if __name__ == "__main__":
    globals()[sys.argv[1]]()
