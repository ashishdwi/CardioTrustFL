"""E2: poisoning attacks x defences x attacker knowledge (full vs partial).
Scenario A: natural 4-hospital federation, 1 attacker (each site in turn), 3 seeds.
Scenario B: pooled Dirichlet (K=10, alpha=1.0), m in {1,2} attackers, 3 seeds.
All runs use DP-SGD (sigma=1.5) as in the main configuration."""
import json, sys, warnings
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings("ignore")
from ctfl.attacks import Attack
from ctfl.data import load_natural, load_pooled_dirichlet
from ctfl.fl import Config, run_federated

DEFENCES = {
    "FedAvg": dict(agg="fedavg"),
    "Balance-wt.": dict(agg="balance", weight_cap=1.0),
    "Median": dict(agg="median"),
    "Trimmed mean": dict(agg="trimmed"),
    "Krum": dict(agg="krum"),
    "Cos-filter (conf.)": dict(agg="balance", weight_cap=1.0, filt="cos_fixed_full", cos_tau=0.3),
    "Cos-filter MAD+clip": dict(agg="uniform", filt="cos_mad", norm_clip=True),
    "Trimmed+clip": dict(agg="trimmed", norm_clip=True),
}
ATTACKS = [("none", None), ("signflip", "full"), ("gauss", "full"), ("labelflip", "full"),
           ("lie", "full"), ("lie", "partial"), ("minmax", "full"), ("minmax", "partial"),
           ("adaptive", "full"), ("adaptive", "partial")]


def job(a):
    scen, dname, kind, kn, attackers, seed, m = a
    if scen == "natural":
        fed = load_natural(seed)
    else:
        fed = load_pooled_dirichlet(10, 1.0, seed)
    over = dict(DEFENCES[dname])
    if over.get("agg") in ("trimmed",):
        over["trim_f"] = m
    if over.get("agg") == "krum":
        over["krum_f"] = m
    cfg = Config(seed=seed, **over)
    att = None if kind == "none" else Attack(kind, attackers, kn, len(fed.clients), seed)
    r = run_federated(fed, cfg, attack=att)
    return dict(scenario=scen, defence=dname, attack=kind, knowledge=kn, attackers=list(attackers),
                m=m, seed=seed, auc=r["auc"], bal_acc=r["bal_acc"], f1=r["f1"], acc=r["acc"],
                tpr=r.get("tpr", np.nan), fpr=r.get("fpr", np.nan),
                gamma=(float(np.median(att.gamma_log)) if att is not None and att.gamma_log else None))


def main():
    jobs = []
    for seed in range(3):
        for site in range(4):
            for d in DEFENCES:
                for kind, kn in ATTACKS:
                    if kind == "none" and site > 0:
                        continue
                    jobs.append(("natural", d, kind, kn, [site], seed, 1))
        rng = np.random.RandomState(seed)
        for m in (1, 2):
            attackers = sorted(rng.choice(10, m, replace=False).tolist())
            for d in DEFENCES:
                for kind, kn in ATTACKS:
                    if kind == "none" and m > 1:
                        continue
                    jobs.append(("pooled", d, kind, kn, attackers, seed, m))
    print(len(jobs), "jobs", flush=True)
    with Pool(2) as p:
        out = []
        for i, r in enumerate(p.imap_unordered(job, jobs, chunksize=2)):
            out.append(r)
            if i % 50 == 0:
                print(i, flush=True)
                json.dump(out, open("results/e2_attacks.json", "w"), default=float)
    json.dump(out, open("results/e2_attacks.json", "w"), default=float, indent=1)


if __name__ == "__main__":
    main()
