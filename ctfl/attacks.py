"""Model-poisoning adversaries with explicit knowledge models.

knowledge = "full"    : the attacker sees every honest update of the CURRENT
                        round before submitting (what an adaptive last mover
                        obtains when updates are revealed without commitment).
knowledge = "partial" : commit-reveal is enforced, so the attacker only has
                        (i) its own benign update(s) of the current round and
                        (ii) the honest updates of the PREVIOUS round, which
                        are public after the reveal phase.

Attacks
-------
signflip  : -5 x own benign update                      (knowledge-free)
gauss     : isotropic noise with 5x own-update norm     (knowledge-free)
labelflip : local training on flipped labels           (knowledge-free)
lie       : "A Little Is Enough" (Baruch et al. 2019)   mu - z * sigma
minmax    : Shejwalkar & Houmansadr (NDSS'21) Min-Max, inverse-std direction
adaptive  : AGR-tailored attack: reversal mu + gamma * (-mu/|mu|) with gamma
            chosen by simulating the defence's own aggregation rule on the
            attacker's knowledge set (Shejwalkar & Houmansadr 2021).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm as gauss

from .fl import apply_filter

LIE_Z = 1.0    # fixed z used in the paper; None gives the analytic z_max of Baruch et al.,
               # which is 0 (a no-op attack) with one attacker and n = 4 or 10

KNOWLEDGE_FREE = {"signflip", "gauss", "labelflip"}


class Attack:
    def __init__(self, kind, attackers, knowledge="full", n_total=None, seed=0):
        self.kind = kind
        self.attackers = set(attackers)
        self.knowledge = knowledge
        self.n_total = n_total
        self.rng = np.random.RandomState(10_000 + seed)
        self.gamma_log = []

    # attackers claim the maximum aggregation weight they can
    def claim(self, claimed, cfg):
        claimed = list(claimed)
        if cfg.agg == "balance":
            for i in self.attackers:
                claimed[i] = 10 ** 9
        return claimed

    def _knowledge(self, honest_now, prev_honest, own):
        if self.knowledge == "full":
            return list(honest_now) + list(own)
        ks = list(own)
        if prev_honest is not None:
            ks += list(prev_honest)
        return ks

    def craft(self, r, w, honest_now, prev_honest, own, cfg, claimed):
        m = len(own)
        if self.kind == "signflip":
            return [-5.0 * o for o in own]
        if self.kind == "gauss":
            return [self.rng.randn(len(o)) * 5.0 * np.linalg.norm(o) / np.sqrt(len(o)) for o in own]
        K = self._knowledge(honest_now, prev_honest, own)
        if len(K) < 2:          # nothing to estimate from yet -> behave honestly
            return list(own)
        K = np.stack(K)
        mu, sd = K.mean(0), K.std(0) + 1e-12
        if self.kind == "lie":
            n = self.n_total
            s = int(np.floor(n / 2 + 1)) - m
            z = gauss.ppf((n - s) / n) if LIE_Z is None else LIE_Z
            return [mu - z * sd] * m
        if self.kind == "minmax":
            p = -sd / np.linalg.norm(sd)
            bound = max(np.linalg.norm(a - b) for a in K for b in K)
            g = self._search(lambda g: max(np.linalg.norm(mu + g * p - k) for k in K) <= bound)
            self.gamma_log.append(g)
            return [mu + g * p] * m
        if self.kind == "adaptive":
            # AGR-tailored attack (Shejwalkar & Houmansadr 2021): the attacker
            # runs the defence's own deterministic aggregation on its knowledge
            # set and picks the reversal magnitude gamma that maximises the
            # predicted damage  -<agg, mu_hat>.
            from .fl import aggregate
            mu_hat = mu / (np.linalg.norm(mu) + 1e-12)
            p = -mu_hat
            scale = float(np.median(np.linalg.norm(K, axis=1)))
            wK = [float(np.median(claimed))] * len(K)
            wm = [claimed[i] for i in sorted(self.attackers)]
            ctx = {"w": w}
            best_g, best = 0.0, -np.inf
            for g in np.concatenate([[0.0], np.logspace(-2, 1.5, 36) * scale]):
                mal = mu + g * p
                agg, _, _ = aggregate(list(K) + [mal] * m, wK + wm, cfg, ctx)
                dmg = -float(agg @ mu_hat)
                if dmg > best + 1e-12:
                    best, best_g = dmg, g
            self.gamma_log.append(best_g / (scale + 1e-12))
            return [mu + best_g * p] * m
        raise ValueError(self.kind)

    @staticmethod
    def _search(ok, hi=None, iters=30):
        lo = 0.0
        if hi is None:
            hi = 1.0
            while ok(hi) and hi < 1e6:
                hi *= 2
        if ok(hi):
            return hi
        for _ in range(iters):
            mid = (lo + hi) / 2
            if ok(mid):
                lo = mid
            else:
                hi = mid
        return lo
