"""Orchestrates one deployment run: a live EVM node with real block times
(anvil, separate process), the content store (separate process), the
operator (deploys and registers), and N hospital processes.
usage: run_deploy.py <outdir> [--block-time S] [--rounds R] [--crash I@ROUND]"""
import json, os, subprocess, sys, time, signal
import numpy as np
from eth_account import Account
from web3 import Web3

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import shutil
# anvil binary: $ANVIL, then ./node_modules/.bin/anvil (npm install), then PATH (foundryup)
ANVIL = (os.environ.get("ANVIL")
         or next((p for p in [os.path.join(ROOT, "node_modules", ".bin", "anvil")] if os.path.exists(p)), None)
         or shutil.which("anvil"))
if not ANVIL:
    sys.exit("anvil not found: run `npm install` in the repository root or install Foundry, or set $ANVIL")
out = os.path.abspath(sys.argv[1]); os.makedirs(out, exist_ok=True)
arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
BT, R = arg("--block-time", 2), arg("--rounds", 30)
crash = arg("--crash", "")
CB, RB, VB = arg("--commit", 2), arg("--reveal", 2), arg("--vote", 3)
KEYS = f"{HERE}/keys_local_only.json"
if not os.path.exists(KEYS):   # fresh throw-away keys for a local test chain; never commit them
    json.dump({"hospitals": [Account.create().key.hex() for _ in range(5)]}, open(KEYS, "w"))
keys = json.load(open(KEYS))["hospitals"]
hk, opk = keys[:4], keys[4]
port = 8545
procs = []
import atexit
def _cleanup():
    for p in procs:
        try: os.killpg(p.pid, signal.SIGTERM)
        except Exception: pass
atexit.register(_cleanup)
procs.append(subprocess.Popen([ANVIL, "--port", str(port), "--block-time", str(BT), "--chain-id", "31337",
                               "--hardfork", "cancun", "--gas-limit", "30000000", "--silent"],
                              start_new_session=True))
procs.append(subprocess.Popen([sys.executable, f"{HERE}/store.py", "8700"], start_new_session=True))
rpc = f"http://127.0.0.1:{port}"
w3 = Web3(Web3.HTTPProvider(rpc))
for _ in range(100):
    try:
        w3.eth.block_number; break
    except Exception:
        time.sleep(0.1)
for k in keys:
    w3.provider.make_request("anvil_setBalance", [Account.from_key(k).address, hex(10**24)])
art_path = f"{ROOT}/contracts/CardioTrustFL.json"
art = json.load(open(art_path))
op = Account.from_key(opk)
deposit = Web3.to_wei(0.01, "ether")
nonce = w3.eth.get_transaction_count(op.address)
def send(tx):
    global nonce
    tx.pop('maxFeePerGas', None); tx.pop('maxPriorityFeePerGas', None)
    tx.update(nonce=nonce, gas=6_000_000, gasPrice=w3.eth.gas_price * 2, chainId=31337); nonce += 1
    h = w3.eth.send_raw_transaction(op.sign_transaction(tx).raw_transaction)
    return w3.eth.wait_for_transaction_receipt(h, poll_latency=0.05)
C = w3.eth.contract(abi=art["abi"], bytecode=art["bytecode"])
rc = send(C.constructor(deposit, CB, RB, VB).build_transaction({"from": op.address}))
addr = rc.contractAddress
c = w3.eth.contract(address=addr, abi=art["abi"])
names = ["Hungary", "Switzerland", "VA Long Beach", "Cleveland"]
for k, nm in zip(hk, names):
    send(c.functions.registerHospital(Account.from_key(k).address, nm).build_transaction({"from": op.address}))
cfg = dict(rpcs=[rpc], hospitals=hk, contract=addr, artifact=art_path, store="http://127.0.0.1:8700",
           seed=0, rounds=R, deposit=deposit, commit_blocks=CB, reveal_blocks=RB, vote_blocks=VB,
           block_time=BT, out=out, deploy_gas=rc.gasUsed)
ci = ca = None
if crash:
    ci, ca = map(int, crash.split("@")); cfg.update(crash_index=ci, crash_at=ca)
json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)
t0 = time.time()
hosp = []
for i in range(4):
    a = [sys.executable, f"{HERE}/hospital.py", f"{out}/config.json", str(i)]
    if ci == i: a += ["--crash-at", str(ca)]
    hosp.append(subprocess.Popen(a, stdout=open(f"{out}/h{i}.log", "w"), stderr=subprocess.STDOUT))
codes = [p.wait() for p in hosp]
cfg.pop("hospitals", None)          # do not leave keys in the results folder
json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)
wall = time.time() - t0
# chain-side facts
fin = c.events.Finalized().get_logs(from_block=0, to_block="latest")
failed = c.events.RoundFailed().get_logs(from_block=0, to_block="latest")
started = c.events.RoundStarted().get_logs(from_block=0, to_block="latest")
reps = {Account.from_key(k).address: c.functions.hospitals(Account.from_key(k).address).call()[3] for k in hk}
summary = dict(exit_codes=codes, wall_s=wall, finalized=len(fin), failed=len(failed), started=len(started),
               blocks=w3.eth.block_number, reputation=reps)
# pooled test AUC of the deployed final model
sys.path.insert(0, ROOT)
from ctfl.data import load_natural
from ctfl.fl import evaluate
fed = load_natural(0)
wf = np.load(f"{out}/final_model_h0.npy")
ev = evaluate(wf, fed, fed.clients[0].X_train.shape[1])
summary.update(test_auc=ev["auc"], test_bal_acc=ev["bal_acc"])
json.dump(summary, open(f"{out}/summary.json", "w"), indent=1, default=float)
print(json.dumps(summary, default=float))
for p in procs:
    try: os.killpg(p.pid, signal.SIGTERM)
    except Exception: pass
