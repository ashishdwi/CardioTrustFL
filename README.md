# CardioTrustFL

Code, smart contract, results and supplementary material for the paper

> **A Decentralised Byzantine-Tolerant Aggregation Protocol for Verifiable,
> Differentially Private Federated Learning across Hospitals**

CardioTrustFL removes the trusted aggregation server from federated learning. On a smart
contract, each hospital commits to its encrypted, differentially private model update, reveals
it before a deadline under a deposit, recomputes a deterministic robust aggregate itself, and
the round is finalised when a Byzantine quorum of hospitals agrees on the model hash.

This repository reproduces every number, table and figure in the paper.

* **Online Resource** (supplementary material of the paper): [[`Online_Resource.pdf`]((https://github.com/ashishdwi/CardioTrustFL/blob/main/Online%20Resource.pdf))](https://github.com/ashishdwi/CardioTrustFL/blob/main/Online%20Resource.pdf)

---

## Repository layout

| Path | Contents |
|---|---|
| `ctfl/data.py` | Recovers the four institutional sites from the combined UCI file, federated standardisation, and the three scenarios (simulated four-site federation, Cleveland–Dirichlet, pooled Dirichlet) |
| `ctfl/fl.py` | Model, DP-SGD local training (per-example clipping, Poisson sampling, cumulative Rényi-DP accounting), FedAvg / FedProx / SCAFFOLD, all aggregation rules including the CardioTrustFL default (MAD-calibrated cosine filter, median-norm clipping, uniform mean), evaluation with a validation-calibrated threshold |
| `ctfl/attacks.py` | Sign-flip, Gaussian, label-flip, LIE, Min-Max and AGR-tailored attacks, each with full or partial (commit–reveal) knowledge |
| `contracts/CardioTrustFL.sol` | The smart contract (Solidity 0.8.26) |
| `contracts/CardioTrustFL.json` | Compiled ABI and bytecode (optimiser on, `viaIR`, Cancun EVM) |
| `contracts/compile.js` | Recompiles the contract with solc-js |
| `exp_chain.py` | In-process EVM: end-to-end run with an encrypted, content-addressed store (bit-identity check), gas scaling to 50 hospitals, 17 adversarial contract checks |
| `deploy/` | Multi-process deployment: four hospital processes, a content store and a live EVM node; latency, throughput and crash-fault runs (see [`deploy/README_deploy.md`](deploy/README_deploy.md)) |
| `exp_extra.py` | Utility, calibration and loss ablation (`e1`), filter calibration (`e3`), privacy–utility (`e4`), scaling the number of clients (`e5`) |
| `exp_attack.py` | Attack × aggregation rule × attacker knowledge study |
| `exp_ablation.py` | Component ablation of the default aggregation rule |
| `exp_offchain.py`, `report_offchain.py` | Off-chain cost per hospital for models of up to 25 M parameters and up to 50 hospitals |
| `make_report.py` | Turns `results/*.json` into the LaTeX tables, number macros and figures of the paper |
| `results/` | The raw results behind the paper (JSON) |
| `data/` | UCI Heart Disease data (CC BY 4.0, see [`data/README.md`](data/README.md)) |
| `docs/Online_Resource.pdf` | Supplementary material of the paper |

## Installation

```bash
git clone https://github.com/ashishdwi/CardioTrustFL.git
cd CardioTrustFL
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
npm install                 # solc 0.8.26 and the anvil EVM node (only for compile and deploy/)
```

Tested on Linux with a 2-core CPU and no GPU, with Python 3.13, PyTorch 2.14 (CPU),
Opacus 1.6.0 (accountant only), scikit-learn 1.9.1, NumPy 2.5.3, SciPy 1.18.1, web3.py 8.0.0,
eth-tester 0.14.0b1, py-evm 0.12.1b1, cryptography 50.0.1, solc-js 0.8.26 and Anvil 1.7.1.

## Reproducing the results

Run each command from the repository root; results are written to `results/`.

| Command | Paper |
|---|---|
| `python3 exp_extra.py e1` | Table 7; Online Resource 1, Tables S1–S3 |
| `python3 exp_extra.py e3` | Fig. 3; Table S5 |
| `python3 exp_extra.py e4` | Section 7.3; Fig. S1, Table S4 |
| `python3 exp_extra.py e5` | Section 7.6; Table S12 |
| `python3 exp_attack.py` | Table 8, Fig. 4; Table S6 (about 1.5 h on 2 cores) |
| `python3 exp_ablation.py` | Section 7.5; Table S7 |
| `python3 exp_chain.py` | Table 5, Section 7.1; Table S11 |
| `bash deploy/run_all.sh`, then `python3 deploy/report_deploy.py` | Table 6, Fig. 2; Table S8 |
| `python3 exp_offchain.py` | Table 9 |
| `python3 make_report.py` | Writes all tables and figures to `paper/` |




## Licence

Code: MIT (see [`LICENSE`](LICENSE)). Data: CC BY 4.0, from the UCI Machine Learning Repository
(see [`data/README.md`](data/README.md)).
