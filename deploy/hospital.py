"""One hospital of the CardioTrustFL deployment, run as its own OS process.

It holds only its own site's data and key, talks to the chain over JSON-RPC and
to the content store over HTTP, and runs the full round itself:
DP-SGD -> encrypt + upload -> commit -> reveal -> fetch, verify, decrypt ->
aggregate -> validation counts -> threshold -> vote -> adopt -> (settle).

usage: hospital.py <config.json> <index> [--crash-at ROUND]"""
import json, os, secrets, sys, time, urllib.request, warnings

import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from eth_account import Account
from opacus.accountants import RDPAccountant
from web3 import Web3

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ctfl.data import load_natural
from ctfl.fl import Config, HeartNet, LocalTrainer, aggregate, canonical_hash, flat, predict

import hashlib

PHASE_FINAL, PHASE_FAILED = 4, 5
GRID = np.linspace(0.05, 0.95, 91)


def sha3(b):
    return hashlib.sha3_256(b).digest()


def ser(v):
    return np.asarray(v, dtype="<f8").tobytes()


def deser(b):
    return np.frombuffer(b, dtype="<f8").copy()


def blob_key(salt):
    return sha3(b"CTFL-blob-key" + salt)


class Hospital:
    def __init__(self, cfgfile, i, crash_at=None):
        C = json.load(open(cfgfile))
        self.C, self.i, self.crash_at = C, i, crash_at
        self.N = len(C["hospitals"])
        self.w3 = Web3(Web3.HTTPProvider(C["rpcs"][i % len(C["rpcs"])], request_kwargs={"timeout": 30}))
        self.acct = Account.from_key(C["hospitals"][i])
        self.addr = self.acct.address
        art = json.load(open(C["artifact"]))
        self.c = self.w3.eth.contract(address=C["contract"], abi=art["abi"])
        self.store = C["store"]
        self.nonce = self.w3.eth.get_transaction_count(self.addr, "pending")
        self.chain_id = self.w3.eth.chain_id
        self.log = dict(index=i, address=self.addr, rounds=[], txs=[])
        # local data and DP-SGD state
        fed = load_natural(C["seed"])
        self.cl = fed.clients[i]
        self.d = self.cl.X_train.shape[1]
        self.cfg = Config(agg="uniform", filt="cos_mad", norm_clip=True, seed=C["seed"], rounds=C["rounds"])
        torch.manual_seed(C["seed"])                       # common initial model
        self.w = flat(HeartNet(self.d))
        torch.manual_seed(1000 * C["seed"] + 17 * (i + 1))  # private randomness
        self.rng = np.random.RandomState(1000 * C["seed"] + 17 * (i + 1))
        self.trainer = LocalTrainer(self.d, self.cfg)
        self.acc = RDPAccountant()

    # ---------------------------------------------------------------- chain io
    def block(self):
        return self.w3.eth.block_number

    def wait_block(self, n):
        while self.block() < n:
            time.sleep(0.05)

    def rnd(self, r):
        return self.c.functions.rounds(r).call()

    def send(self, fn, name, value=0):
        t0 = time.time()
        tx = fn.build_transaction({"from": self.addr, "nonce": self.nonce, "gas": 3_000_000,
                                   "gasPrice": self.w3.eth.gas_price * 2, "value": value,
                                   "chainId": self.chain_id})
        h = self.w3.eth.send_raw_transaction(self.acct.sign_transaction(tx).raw_transaction)
        self.nonce += 1
        while True:
            try:
                rc = self.w3.eth.get_transaction_receipt(h)
                if rc is not None:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        t1 = time.time()
        print(f"[h{self.i}] {name} status={rc.status} block={rc.blockNumber} gas={rc.gasUsed} t={t1-t0:.2f}s", flush=True)
        self.log["txs"].append(dict(name=name, gas=rc.gasUsed, ok=rc.status == 1,
                                    latency=t1 - t0, block=rc.blockNumber))
        return rc

    def try_send(self, fn, name, value=0):
        try:
            fn.call({"from": self.addr, "value": value}, block_identifier="pending")
        except Exception:
            return None
        return self.send(fn, name, value)

    def put(self, path, body):
        req = urllib.request.Request(f"{self.store}/{path}", data=body, method="PUT")
        urllib.request.urlopen(req, timeout=10).read()

    def get(self, path):
        return urllib.request.urlopen(f"{self.store}/{path}", timeout=10).read()

    # ---------------------------------------------------------------- round
    def designated(self, r):
        d = (r - 1) % self.N
        ci, ca = self.C.get("crash_index"), self.C.get("crash_at")
        if ci is not None and d == ci and r >= ca:
            d = (d + 1) % self.N
        return d

    def ensure_started(self, r):
        designated = self.designated(r)
        t_wait = time.time()
        while True:
            cur = self.c.functions.currentRound().call()
            if cur >= r:
                return
            prev = self.rnd(cur) if cur > 0 else None
            if prev is not None and prev[3] not in (PHASE_FINAL, PHASE_FAILED):
                if self.block() > prev[2]:
                    self.try_send(self.c.functions.closeExpired(), "closeExpired")
                time.sleep(0.05)
                continue
            # fallback: any hospital starts the round if the designated one is silent
            if self.i == designated or time.time() - t_wait > 3 + self.i:
                self.try_send(self.c.functions.startRound(), "startRound")
            time.sleep(0.05)

    def run_round(self, r):
        L = dict(round=r)
        t = {}
        self.ensure_started(r)
        rd = self.rnd(r)
        commit_end, reveal_end, vote_end = rd[0], rd[1], rd[2]
        print(f"[h{self.i}] round {r} windows commit<={commit_end} reveal<={reveal_end} vote<={vote_end} now={self.block()}", flush=True)
        t["seen"] = time.time()
        L["start_block"] = commit_end - self.C["commit_blocks"]
        # 1. local DP-SGD
        delta, _, _ = self.trainer.run(self.w, self.cl, self.rng, self.acc)
        eps = self.acc.get_epsilon(self.cfg.delta)
        t["trained"] = time.time()
        # 2. encrypt and upload ciphertext under its content address
        plain = ser(delta)
        uh, salt = sha3(plain), secrets.token_bytes(32)
        ct = AESGCM(blob_key(salt)).encrypt(b"\x00" * 12, plain, None)
        uri = sha3(ct)
        self.put(f"blob/{uri.hex()}", ct)
        eps_m, weight = int(round(eps * 1000)), int(self.cl.n)
        t["uploaded"] = time.time()
        # 3. commit
        cm = self.c.functions.commitmentOf(uh, uri, eps_m, weight, salt, self.addr, r).call()
        rc = self.send(self.c.functions.commit(cm), "commit", self.C["deposit"])
        t["committed"] = time.time()
        L["committed_ok"] = rc.status == 1
        # 4. reveal after the commit deadline
        self.wait_block(commit_end)
        rc = self.send(self.c.functions.reveal(uh, uri, eps_m, weight, salt), "reveal")
        t["revealed"] = time.time()
        # 5. after the reveal deadline: read reveal order, fetch, verify, decrypt
        self.wait_block(reveal_end)
        order = self.c.functions.revealersOf(r).call()
        evs = self.c.events.Revealed().get_logs(from_block=L["start_block"], to_block="latest",
                                                argument_filters={"round": r})
        salts = {e["args"]["h"]: e["args"]["salt"] for e in evs}
        updates, weights = [], []
        for h in order:
            rv = self.c.functions.reveals(r, h).call()
            blob = self.get(f"blob/{rv[1].hex()}")
            assert sha3(blob) == rv[1], "ciphertext address mismatch"
            pl = AESGCM(blob_key(salts[h])).decrypt(b"\x00" * 12, blob, None)
            assert sha3(pl) == rv[0], "plaintext digest mismatch"
            updates.append(deser(pl))
            weights.append(rv[3])
        t["fetched"] = time.time()
        if not updates:
            raise RuntimeError(f"round {r}: no revealed updates")
        agg, keep, _ = aggregate(updates, weights, self.cfg, {"w": self.w})
        w_new = self.w + agg
        bitmap = sum(1 << k for k in keep)
        t["aggregated"] = time.time()
        # 6. validation confusion counts -> board; threshold from the summed counts
        p = predict(w_new, self.cl.X_val, self.d)
        y = self.cl.y_val.astype(int)
        cnt = [[int(((p >= g) & (y == 1)).sum()), int(((p < g) & (y == 1)).sum()),
                int(((p < g) & (y == 0)).sum()), int(((p >= g) & (y == 0)).sum())] for g in GRID]
        self.put(f"counts/{r}/{self.addr.lower()}", json.dumps(cnt).encode())
        want = {h.lower() for h in order}
        while True:
            board = json.loads(self.get(f"counts/{r}"))
            if want <= set(board) or self.block() >= vote_end - 1:
                break
            time.sleep(0.05)
        tot = np.sum([json.loads(board[h]) for h in want if h in board], axis=0)
        best, thr = -1, 0.5
        for g, (tp, fn, tn, fp) in zip(GRID, tot):
            b = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
            if b > best:
                best, thr = b, g
        thr_m = int(round(thr * 1000))
        gh = bytes.fromhex(canonical_hash(w_new))
        t["counted"] = time.time()
        # 7. vote (unless the quorum has already finalised the round)
        if self.rnd(r)[3] != PHASE_FINAL:
            self.try_send(self.c.functions.vote(gh, bitmap, thr_m), "vote")
        t["voted"] = time.time()
        # 8. wait for finality or expiry
        while True:
            ph = self.rnd(r)
            if ph[3] in (PHASE_FINAL, PHASE_FAILED):
                break
            if self.block() > vote_end:
                self.try_send(self.c.functions.closeExpired(), "closeExpired")
            time.sleep(0.05)
        t["final"] = time.time()
        L["phase"] = int(ph[3])
        L["agree"] = ph[3] == PHASE_FINAL and ph[9] == gh and ph[10] == bitmap and ph[11] == thr_m
        if L["agree"]:
            self.w = w_new
        fin = self.c.events.Finalized().get_logs(from_block=L["start_block"], to_block="latest",
                                                 argument_filters={"round": r})
        L["final_block"] = fin[0]["blockNumber"] if fin else None
        # 9. settlement by the designated hospital (or the next one alive)
        if self.i == self.designated(r):
            self.try_send(self.c.functions.settleAll(r), "settleAll")
        L.update(eps=eps, n_reveal=len(order), kept=len(keep), threshold=thr,
                 times={k: v - t["seen"] for k, v in t.items()})
        self.log["rounds"].append(L)

    def run(self):
        for r in range(1, self.C["rounds"] + 1):
            if self.crash_at is not None and r >= self.crash_at:
                self.log["crashed_at"] = r
                break
            self.run_round(r)
            if r == self.C["rounds"] and self.i == 0:
                np.save(self.C["out"] + f"/final_model_h{self.i}.npy", self.w)
        json.dump(self.log, open(self.C["out"] + f"/hospital{self.i}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    crash = int(sys.argv[sys.argv.index("--crash-at") + 1]) if "--crash-at" in sys.argv else None
    Hospital(sys.argv[1], int(sys.argv[2]), crash).run()
