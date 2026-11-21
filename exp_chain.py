"""E6: end-to-end protocol execution on a local EVM (py-evm via eth-tester)
using the compiled CardioTrustFL v4 contract.
 (a) 30 FL rounds on the natural 4-hospital federation, every round driven by
     the contract (commit -> reveal -> vote -> finalise -> settle); gas and
     wall-clock per phase are recorded and the finalised global-model hash is
     compared with the plain simulation (protocol must not change the model).
 (b) protocol-only gas scaling for K = 4..50 hospitals.
 (c) adversarial protocol checks: copied commitment, late reveal, wrong
     reveal, non-revealer slashing, conflicting votes / quorum."""
import hashlib, json, secrets, time, warnings
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import numpy as np
import torch
from eth_tester import EthereumTester, PyEVMBackend
from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

warnings.filterwarnings("ignore")
from ctfl.data import load_natural
from ctfl.fl import (Config, HeartNet, LocalTrainer, aggregate, canonical_hash, choose_threshold,
                     flat, predict, run_federated)
from opacus.accountants import RDPAccountant

ART = json.load(open("contracts/CardioTrustFL.json"))
DEPOSIT = Web3.to_wei(0.01, "ether")
COMMIT_P, REVEAL_P, VOTE_P = 20, 20, 20


def new_chain(n_accounts, period=None):
    P = period or COMMIT_P
    backend = PyEVMBackend(
        genesis_parameters=PyEVMBackend.generate_genesis_params(overrides={"gas_limit": 30_000_000}),
        genesis_state=PyEVMBackend.generate_genesis_state(num_accounts=max(n_accounts + 1, 10)))
    tester = EthereumTester(backend)
    w3 = Web3(EthereumTesterProvider(tester))
    accts = w3.eth.accounts
    C = w3.eth.contract(abi=ART["abi"], bytecode=ART["bytecode"])
    tx = C.constructor(DEPOSIT, P, P, P).transact({"from": accts[0]})
    rc = w3.eth.wait_for_transaction_receipt(tx)
    c = w3.eth.contract(address=rc.contractAddress, abi=ART["abi"])
    return w3, tester, c, accts, rc.gasUsed


class Gas:
    def __init__(self):
        self.g = {}

    def add(self, k, rc):
        self.g.setdefault(k, []).append(rc.gasUsed)


def tx(w3, fn, sender, gas, key, value=0):
    h = fn.transact({"from": sender, "value": value, "gas": 3_000_000})
    rc = w3.eth.wait_for_transaction_receipt(h)
    assert rc.status == 1, key
    gas.add(key, rc)
    return rc


def mine_to(w3, tester, block):
    cur = w3.eth.block_number
    if block > cur:
        tester.mine_blocks(block - cur)


def b32(hexstr):
    return bytes.fromhex(hexstr)


# ---- canonical serialisation and encrypted content store -------------------
def ser(vec):
    """Canonical serialisation: fixed parameter order, little-endian float64."""
    return np.asarray(vec, dtype="<f8").tobytes()


def deser(b):
    return np.frombuffer(b, dtype="<f8").copy()


def sha3(b):
    return hashlib.sha3_256(b).digest()


def blob_key(salt):
    # the commitment salt seeds the blob key (domain-separated), so the key is
    # disclosed exactly when, and only when, the commitment is opened
    return sha3(b"CTFL-blob-key" + salt)


def encrypt_blob(plain, salt):
    return AESGCM(blob_key(salt)).encrypt(b"\x00" * 12, plain, None)   # fresh key per update


def decrypt_blob(ct, salt):
    return AESGCM(blob_key(salt)).decrypt(b"\x00" * 12, ct, None)


def run_protocol_fl(seed=0, rounds=30):
    fed = load_natural(seed)
    K = len(fed.clients)
    cfg = Config(agg="uniform", filt="cos_mad", norm_clip=True, seed=seed, rounds=rounds)
    w3, tester, c, accts, deploy_gas = new_chain(K)
    gov, hosp = accts[0], accts[1:K + 1]
    gas = Gas()
    for i, h in enumerate(hosp):
        tx(w3, c.functions.registerHospital(h, fed.clients[i].name), gov, gas, "registerHospital")
    # identical RNG discipline to ctfl.fl.run_federated
    rng = np.random.RandomState(cfg.seed); torch.manual_seed(cfg.seed)
    d = fed.clients[0].X_train.shape[1]
    w = flat(HeartNet(d))
    trainer = LocalTrainer(d, cfg)
    accs = [RDPAccountant() for _ in range(K)]
    claimed = [cl.n for cl in fed.clients]
    T = dict(train=0.0, chain=0.0, aggregate=0.0, verify=0.0, encrypt=0.0, decrypt=0.0)
    hashes_match = True
    leak_free = True
    store = {}                     # content-addressed store holding ciphertexts only
    for r in range(rounds):
        t0 = time.perf_counter()
        tx(w3, c.functions.startRound(), hosp[r % K], gas, "startRound")
        rnd = c.functions.currentRound().call()
        T["chain"] += time.perf_counter() - t0
        # local training
        t0 = time.perf_counter()
        deltas = []
        for i, cl in enumerate(fed.clients):
            dl, _, _ = trainer.run(w, cl, rng, accs[i])
            deltas.append(dl)
        T["train"] += time.perf_counter() - t0
        # commit
        t0 = time.perf_counter()
        payload = []
        for i, h in enumerate(hosp):
            te = time.perf_counter()
            plain = ser(deltas[i])
            uh = sha3(plain)                                  # digest of the plaintext update
            salt = secrets.token_bytes(32)
            ct = encrypt_blob(plain, salt)
            uri = sha3(ct)                                    # content address of the CIPHERTEXT
            store[uri] = ct                                   # uploaded before the commit deadline
            T["encrypt"] += time.perf_counter() - te
            eps_m = int(round(accs[i].get_epsilon(cfg.delta) * 1000))
            payload.append((uh, uri, eps_m, claimed[i], salt))
            cm = c.functions.commitmentOf(uh, uri, eps_m, claimed[i], salt, h, rnd).call()
            tx(w3, c.functions.commit(cm), h, gas, "commit", DEPOSIT)
        # before T_c an observer sees only commitments and ciphertexts: check that
        # nothing in the store equals or reveals a plaintext digest
        leak_free &= all(sha3(store[p[1]]) == p[1] and p[0] not in store for p in payload)
        mine_to(w3, tester, c.functions.rounds(rnd).call()[0] + 1)
        events = {}
        for i, h in enumerate(hosp):
            rc = tx(w3, c.functions.reveal(*payload[i]), h, gas, "reveal")
            ev = c.events.Revealed().process_receipt(rc)[0]["args"]
            events[h] = ev
        T["chain"] += time.perf_counter() - t0
        # every hospital independently fetches the revealed updates, checks them
        # against the on-chain hashes, and recomputes the deterministic aggregate
        t0 = time.perf_counter()
        order = c.functions.revealersOf(rnd).call()
        idx = [hosp.index(a) for a in order]
        decisions = []
        for j in range(K):
            # hospital j: fetch each ciphertext by its on-chain URI, check it, decrypt
            # with the key derived from the revealed salt, check the plaintext digest
            recovered = []
            for i in idx:
                ev = events[hosp[i]]
                ct = store[bytes(ev["uri"])]
                td = time.perf_counter()
                ok = sha3(ct) == bytes(ev["uri"])
                plain = decrypt_blob(ct, bytes(ev["salt"]))
                ok &= sha3(plain) == bytes(ev["updateHash"])
                ok &= c.functions.reveals(rnd, hosp[i]).call()[0] == bytes(ev["updateHash"])
                T["decrypt"] += time.perf_counter() - td
                hashes_match &= bool(ok)
                recovered.append(deser(plain))
            ta = time.perf_counter()
            agg, keep, _ = aggregate(recovered, [claimed[i] for i in idx], cfg, {"w": w})
            T["aggregate"] += time.perf_counter() - ta
            w_new = w + agg
            Xv, yv = fed.pooled("val")
            thr = choose_threshold(predict(w_new, Xv, d), yv)
            bitmap = sum(1 << k for k in keep)
            decisions.append((b32(canonical_hash(w_new)), bitmap, int(round(thr * 1000))))
        assert len(set(decisions)) == 1, "non-deterministic aggregation"
        T["verify"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        mine_to(w3, tester, c.functions.rounds(rnd).call()[1] + 1)
        for j, h in enumerate(hosp):
            if c.functions.rounds(rnd).call()[3] == 4:   # already finalised
                break
            tx(w3, c.functions.vote(*decisions[j]), h, gas, "vote")
        assert c.functions.rounds(rnd).call()[3] == 4
        tx(w3, c.functions.settleAll(rnd), hosp[0], gas, "settleAll")
        T["chain"] += time.perf_counter() - t0
        w = w_new
    sim = run_federated(fed, cfg)
    same = canonical_hash(sim["weights"]) == canonical_hash(w)
    reps = [c.functions.hospitals(h).call()[3] for h in hosp]
    return dict(deploy_gas=deploy_gas, gas={k: dict(mean=float(np.mean(v)), n=len(v)) for k, v in gas.g.items()},
                time=T, rounds=rounds, final_hash_equals_simulation=bool(same),
                store_holds_only_ciphertext_before_reveal=bool(leak_free),
                blob_bytes=len(ser(w)), ciphertext_bytes=len(encrypt_blob(ser(w), b"\x00" * 32)),
                onchain_hashes_verified=bool(hashes_match), reputations=reps,
                latest_global=c.functions.latestGlobal().call().hex(),
                sim_auc=sim["auc"])


def gas_scaling(Ks=(4, 10, 20, 50), rounds=3):
    out = {}
    for K in Ks:
        w3, tester, c, accts, dg = new_chain(K, period=K + 10)
        gov, hosp = accts[0], accts[1:K + 1]
        gas = Gas()
        for h in hosp:
            tx(w3, c.functions.registerHospital(h, "h"), gov, gas, "registerHospital")
        t_start = time.perf_counter()
        for r in range(rounds):
            tx(w3, c.functions.startRound(), hosp[0], gas, "startRound")
            rnd = c.functions.currentRound().call()
            pl = []
            for h in hosp:
                uh = secrets.token_bytes(32); salt = secrets.token_bytes(32)
                pl.append((uh, uh, 5000, 50, salt))
                tx(w3, c.functions.commit(c.functions.commitmentOf(uh, uh, 5000, 50, salt, h, rnd).call()),
                   h, gas, "commit", DEPOSIT)
            mine_to(w3, tester, c.functions.rounds(rnd).call()[0] + 1)
            for h, p in zip(hosp, pl):
                tx(w3, c.functions.reveal(*p), h, gas, "reveal")
            mine_to(w3, tester, c.functions.rounds(rnd).call()[1] + 1)
            gh = secrets.token_bytes(32); bm = (1 << K) - 1
            for h in hosp:
                if c.functions.rounds(rnd).call()[3] == 4:
                    break
                tx(w3, c.functions.vote(gh, bm, 500), h, gas, "vote")
            tx(w3, c.functions.settleAll(rnd), hosp[0], gas, "settleAll")
        g = {k: float(np.mean(v)) for k, v in gas.g.items()}
        n_votes = len(gas.g["vote"]) / rounds
        g["per_round_total"] = g["startRound"] + K * (g["commit"] + g["reveal"]) + n_votes * g["vote"] + g["settleAll"]
        g["votes_needed"] = n_votes
        g["deploy"] = dg
        g["local_wallclock_per_round_s"] = (time.perf_counter() - t_start) / rounds
        out[K] = g
    return out


def adversarial_checks():
    """Each check returns True if the contract behaves as specified (N = 4, q = 3)."""
    w3, tester, c, accts, _ = new_chain(4)
    gov, A, B, Cc, D = accts[0], *accts[1:5]
    gas = Gas()
    for h in (A, B, Cc, D):
        tx(w3, c.functions.registerHospital(h, "h"), gov, gas, "reg")
    res = {}

    def reverts(fn, sender, value=0):
        try:
            h = fn.transact({"from": sender, "value": value, "gas": 3_000_000})
            return w3.eth.wait_for_transaction_receipt(h).status == 0
        except Exception:
            return True

    def commit_all(rnd, who):
        pl = {}
        for h in who:
            uh = secrets.token_bytes(32); salt = secrets.token_bytes(32)
            pl[h] = (uh, uh, 1000, 10, salt)
            tx(w3, c.functions.commit(c.functions.commitmentOf(*pl[h], h, rnd).call()), h, gas, "commit", DEPOSIT)
        return pl

    # ---------------- round 1: D withholds, votes split -> round fails ----------
    tx(w3, c.functions.startRound(), A, gas, "start")
    rnd = 1
    res["quorum_is_N_minus_floor_N_minus_1_over_3"] = c.functions.rounds(rnd).call()[6] == 3
    pl1 = commit_all(rnd, (A, B, Cc, D))
    res["early_reveal_rejected"] = reverts(c.functions.reveal(*pl1[A]), A)
    mine_to(w3, tester, c.functions.rounds(rnd).call()[0] + 1)
    res["copied_opening_rejected"] = reverts(c.functions.reveal(*pl1[A]), B)
    bad = (secrets.token_bytes(32),) + pl1[A][1:]
    res["equivocating_opening_rejected"] = reverts(c.functions.reveal(*bad), A)
    for h in (A, B, Cc):                       # D withholds (selective abort)
        tx(w3, c.functions.reveal(*pl1[h]), h, gas, "reveal")
    mine_to(w3, tester, c.functions.rounds(rnd).call()[1] + 1)
    res["late_reveal_rejected"] = reverts(c.functions.reveal(*pl1[D]), D)
    res["non_revealer_cannot_vote"] = reverts(c.functions.vote(b"\x01" * 32, 7, 500), D)
    g1, g2 = b"\x11" * 32, b"\x22" * 32
    tx(w3, c.functions.vote(g1, 0b111, 500), A, gas, "vote")
    tx(w3, c.functions.vote(g2, 0b111, 500), Cc, gas, "vote")
    tx(w3, c.functions.vote(g1, 0b111, 500), B, gas, "vote")
    res["two_votes_below_quorum_do_not_finalise"] = c.functions.rounds(rnd).call()[3] != 4
    res["double_vote_rejected"] = reverts(c.functions.vote(g1, 0b111, 500), A)
    mine_to(w3, tester, c.functions.rounds(rnd).call()[2] + 1)
    tx(w3, c.functions.closeExpired(), A, gas, "close")
    res["expired_round_closes_as_failed"] = c.functions.rounds(rnd).call()[3] == 5
    bal_before = w3.eth.get_balance(D)
    tx(w3, c.functions.settleAll(rnd), A, gas, "settle")
    hs = {h: c.functions.hospitals(h).call() for h in (A, B, Cc, D)}
    res["withholder_slashed_and_penalised"] = (hs[D][3] == 85 and w3.eth.get_balance(D) == bal_before
                                              and c.functions.slashedPool().call() == DEPOSIT)
    res["failed_round_no_reward_for_revealers"] = all(hs[h][3] == 100 for h in (A, B, Cc))

    # ---------------- round 2: all reveal; C (Byzantine) deviates ----------------
    tx(w3, c.functions.startRound(), A, gas, "start")
    rnd = 2
    pl2 = commit_all(rnd, (A, B, Cc, D))
    mine_to(w3, tester, c.functions.rounds(rnd).call()[0] + 1)
    for h in (A, B, Cc, D):
        tx(w3, c.functions.reveal(*pl2[h]), h, gas, "reveal")
    mine_to(w3, tester, c.functions.rounds(rnd).call()[1] + 1)
    keep = 0b1011                              # reveal order A,B,C,D: C excluded
    tx(w3, c.functions.vote(g2, 0b1111, 500), Cc, gas, "vote")
    res["single_byzantine_vote_cannot_finalise"] = c.functions.rounds(rnd).call()[3] != 4
    tx(w3, c.functions.vote(g1, keep, 500), A, gas, "vote")
    tx(w3, c.functions.vote(g1, keep, 500), B, gas, "vote")
    res["not_final_with_q_minus_1_votes"] = c.functions.rounds(rnd).call()[3] != 4
    tx(w3, c.functions.vote(g1, keep, 500), D, gas, "vote")
    st = c.functions.rounds(rnd).call()
    res["finalised_at_quorum_on_honest_decision"] = st[3] == 4 and st[9] == g1
    tx(w3, c.functions.settleAll(rnd), A, gas, "settle")
    hs = {h: c.functions.hospitals(h).call() for h in (A, B, Cc, D)}
    res["excluded_update_penalised"] = hs[Cc][3] == 95
    res["kept_updates_rewarded_capped"] = hs[A][3] == 100 and hs[B][3] == 100 and hs[D][3] == 90

    # ---------------- round 3: replaying a round-1 opening ----------------------
    tx(w3, c.functions.startRound(), A, gas, "start")
    tx(w3, c.functions.commit(c.functions.commitmentOf(*pl1[A], A, 1).call()), A, gas, "commit", DEPOSIT)
    mine_to(w3, tester, c.functions.rounds(3).call()[0] + 1)
    res["replay_previous_round_opening_rejected"] = reverts(c.functions.reveal(*pl1[A]), A)
    return res


if __name__ == "__main__":
    out = {}
    out["adversarial_checks"] = adversarial_checks()
    print(out["adversarial_checks"], flush=True)
    out["gas_scaling"] = gas_scaling()
    print(json.dumps(out["gas_scaling"], indent=1), flush=True)
    out["end_to_end"] = run_protocol_fl()
    print(json.dumps(out["end_to_end"], indent=1, default=str), flush=True)
    json.dump(out, open("results/e6_chain.json", "w"), indent=1, default=str)
