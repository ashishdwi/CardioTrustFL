"""Off-chain cost of one round for one hospital as the model (p) and the
federation (n) grow: download volume, verification + decryption of all n
ciphertexts, the CardioTrustFL default aggregation rule (MAD cosine filter,
median-norm clipping, uniform mean) and the canonical model hash.
On-chain gas does not depend on p (only 32-byte digests go on-chain)."""
import hashlib, json, os, secrets, sys, time
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctfl.fl import Config, aggregate, canonical_hash

MEM_LIMIT = 1.6e8          # n*p limit for a 7 GB host (aggregation holds ~4 copies)
P = [3_713, 100_000, 1_000_000, 5_000_000, 25_000_000]
N = [4, 10, 20, 50]
cfg = Config(agg="uniform", filt="cos_mad", norm_clip=True)
rng = np.random.default_rng(0)
out = []
for p in P:
    for n in N:
        if n * p > MEM_LIMIT:
            out.append(dict(p=p, n=n, skipped="memory")); continue
        # each hospital's update: common signal + DP-scale noise
        base = rng.standard_normal(p)
        ups = [base * 0.1 + rng.standard_normal(p) for _ in range(n)]
        blobs = []
        for u in ups:
            salt = secrets.token_bytes(32)
            key = hashlib.sha3_256(b"CTFL-blob-key" + salt).digest()
            pl = u.astype("<f8").tobytes()
            ct = AESGCM(key).encrypt(b"\x00" * 12, pl, None)
            blobs.append((salt, ct, hashlib.sha3_256(ct).digest(), hashlib.sha3_256(pl).digest()))
        del ups
        reps = 3 if p <= 1_000_000 else 1
        tv, ta, th = [], [], []
        for _ in range(reps):
            t0 = time.perf_counter()
            dec = []
            for salt, ct, uri, uh in blobs:
                assert hashlib.sha3_256(ct).digest() == uri
                key = hashlib.sha3_256(b"CTFL-blob-key" + salt).digest()
                pl = AESGCM(key).decrypt(b"\x00" * 12, ct, None)
                assert hashlib.sha3_256(pl).digest() == uh
                dec.append(np.frombuffer(pl, dtype="<f8"))
            t1 = time.perf_counter()
            agg, keep, _ = aggregate(dec, [1] * n, cfg, {"w": np.zeros(p)})
            t2 = time.perf_counter()
            canonical_hash(agg)
            t3 = time.perf_counter()
            tv.append(t1 - t0); ta.append(t2 - t1); th.append(t3 - t2)
            del dec, agg
        mb = n * (8 * p + 16) / 1e6
        r = dict(p=p, n=n, download_MB=mb, verify_decrypt_s=float(np.median(tv)),
                 aggregate_s=float(np.median(ta)), hash_s=float(np.median(th)),
                 transfer_1gbps_s=mb * 8 / 1000)
        r["compute_s"] = r["verify_decrypt_s"] + r["aggregate_s"] + r["hash_s"]
        print(r, flush=True); out.append(r)
        del blobs
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/e8_offchain.json"), "w"), indent=1)
