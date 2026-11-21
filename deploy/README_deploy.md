# Multi-process deployment (Section 7.2)

Four hospital processes, a content store and a live EVM node, each a separate OS process.

```
npm install        # in the repository root; provides the anvil node (or install Foundry, or set $ANVIL)
cd deploy
python3 run_deploy.py ../results/deploy/bt2 --block-time 2 --rounds 30
python3 run_deploy.py ../results/deploy/bt2_crash --block-time 2 --rounds 30 --crash 3@11
python3 run_deploy.py ../results/deploy/bt1 --block-time 1 --rounds 10
python3 run_deploy.py ../results/deploy/bt5 --block-time 5 --rounds 10
python3 report_deploy.py                # tables, macros and figure
```
(`run_all.sh` runs all four.)

* `hospital.py` – one hospital: own data, key, DP-SGD randomness and accountant; commit, reveal,
  fetch/verify/decrypt, aggregate, validation counts, vote, adopt, settle.
* `store.py` – content-addressed ciphertext store and validation-count board (HTTP).
* `run_deploy.py` – starts node and store, deploys and registers, launches the hospitals.

Throw-away keys for the local chain are generated into `keys_local_only.json` on first run.
Never reuse them, and never commit real keys.

## Running against Hyperledger Besu (QBFT)

The hospitals use only standard JSON-RPC and locally signed transactions. To measure a real
multi-validator network, start a 4-validator Besu QBFT network (Besu quick-start, `blockperiodseconds`
1–5), fund the five accounts in `keys_local_only.json` in its genesis, and in `run_deploy.py` replace
the anvil start-up by the network's RPC URL (`rpc = "http://<node>:8545"`, `chain-id` of the network).
Everything else is unchanged.
