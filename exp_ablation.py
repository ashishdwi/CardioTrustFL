"""Component ablation of the CardioTrustFL default rule under attack
(simulated four-site federation, one attacker rotated over all sites, 3 seeds).
Variants: uniform averaging (no filter, no clipping), clipping only,
MAD filter only; the full rule (filter + clipping) is taken from e2."""
import json, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import ctfl.attacks as A
A.LIE_Z = 1.0
import exp_attack as X
X.DEFENCES.update({
    "Uniform (no filter, no clip)": dict(agg="uniform"),
    "Clip only": dict(agg="uniform", norm_clip=True),
    "MAD filter only": dict(agg="uniform", filt="cos_mad"),
})
VARIANTS = ["Uniform (no filter, no clip)", "Clip only", "MAD filter only"]
if __name__ == "__main__":
    jobs = []
    for seed in range(3):
        for site in range(4):
            for d in VARIANTS:
                for kind, kn in X.ATTACKS:
                    if kind == "none" and site > 0:
                        continue
                    jobs.append(("natural", d, kind, kn, [site], seed, 1))
    print(len(jobs), flush=True)
    with Pool(2) as p:
        out = p.map(X.job, jobs, chunksize=2)
    json.dump(out, open("results/e7_ablation.json", "w"), default=float, indent=1)
    print("done", flush=True)
