"""Summarise the multi-process deployment runs into JSON, LaTeX macros,
a LaTeX table and a latency-breakdown figure."""
import glob, json, os, statistics as st
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = f"{ROOT}/results/deploy"
OUT_TEX = f"{ROOT}/results/deploy_tex"
os.makedirs(OUT_TEX, exist_ok=True)


def load(run):
    d = f"{R}/{run}"
    cfg = json.load(open(f"{d}/config.json"))
    summ = json.load(open(f"{d}/summary.json"))
    hs = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/hospital*.json"))]
    return cfg, summ, hs


def analyse(run):
    cfg, summ, hs = load(run)
    bt = cfg["block_time"]
    rounds = {}
    for h in hs:
        for L in h["rounds"]:
            rounds.setdefault(L["round"], []).append(L)
    starts = sorted({(r, Ls[0]["start_block"]) for r, Ls in rounds.items()})
    finals = {r: next((L["final_block"] for L in Ls if L.get("final_block")), None) for r, Ls in rounds.items()}
    lat_blocks = [finals[r] - sb for r, sb in starts if finals[r] is not None]
    cycle_blocks = [starts[k + 1][1] - starts[k][1] for k in range(len(starts) - 1)]
    agree = all(L["agree"] for Ls in rounds.values() for L in Ls)
    # observed wall-clock latency from round visible to finality, per hospital-round
    obs = [L["times"]["final"] for Ls in rounds.values() for L in Ls]
    # phase breakdown (means over hospital-rounds)
    keys = ["trained", "uploaded", "committed", "revealed", "fetched", "aggregated", "counted", "voted", "final"]
    br = {}
    for Ls in rounds.values():
        for L in Ls:
            prev = 0.0
            for k in keys:
                v = L["times"][k]
                br.setdefault(k, []).append(v - prev)
                prev = v
    br = {k: float(np.mean(v)) for k, v in br.items()}
    txs = {}
    for h in hs:
        for t in h["txs"]:
            txs.setdefault(t["name"], []).append(t)
    tx = {k: dict(n=len(v), gas=float(np.mean([t["gas"] for t in v if t["ok"]] or [0])),
                  lat=float(np.mean([t["latency"] for t in v])),
                  lat_p95=float(np.percentile([t["latency"] for t in v], 95)),
                  ok=sum(t["ok"] for t in v)) for k, v in txs.items()}
    nrev = [Ls[0]["n_reveal"] for r, Ls in sorted(rounds.items())]
    return dict(run=run, block_time=bt, rounds=len(starts), finalized=summ["finalized"], failed=summ["failed"],
                agree=agree, lat_blocks_mean=float(np.mean(lat_blocks)), lat_blocks_max=int(max(lat_blocks)),
                lat_s_mean=float(np.mean(lat_blocks)) * bt,
                obs_s_mean=float(np.mean(obs)), obs_s_p95=float(np.percentile(obs, 95)),
                cycle_blocks_mean=float(np.mean(cycle_blocks)) if cycle_blocks else None,
                cycle_s_mean=float(np.mean(cycle_blocks)) * bt if cycle_blocks else None,
                rounds_per_min=60.0 / (float(np.mean(cycle_blocks)) * bt) if cycle_blocks else None,
                breakdown=br, tx=tx, n_reveal=nrev, wall_s=summ["wall_s"],
                test_auc=summ["test_auc"], reputation=summ["reputation"],
                exit_codes=summ["exit_codes"], crash=cfg.get("crash_at"))


def main():
    runs = [r for r in ["bt1", "bt2", "bt5", "bt2_crash"] if os.path.exists(f"{R}/{r}/summary.json")]
    A = {r: analyse(r) for r in runs}
    json.dump(A, open(f"{R}/deploy_summary.json", "w"), indent=1)
    m = []
    def mac(name, val):
        m.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    b = A.get("bt2")
    if b:
        mac("depRounds", b["rounds"]); mac("depFinal", b["finalized"])
        mac("depLatBlocks", f"{b['lat_blocks_mean']:.1f}"); mac("depLatS", f"{b['obs_s_mean']:.1f}")
        mac("depLatPns", f"{b['obs_s_p95']:.1f}"); mac("depCycleS", f"{b['cycle_s_mean']:.1f}")
        mac("depRpm", f"{b['rounds_per_min']:.1f}"); mac("depAUC", f"{b['test_auc']:.3f}")
        mac("depCommitLat", f"{b['tx']['commit']['lat']:.2f}"); mac("depRevealLat", f"{b['tx']['reveal']['lat']:.2f}")
        mac("depVoteLat", f"{b['tx']['vote']['lat']:.2f}")
        mac("depWallMin", f"{b['wall_s']/60:.1f}")
    c = A.get("bt2_crash")
    if c:
        mac("crashFinal", c["finalized"]); mac("crashRounds", c["rounds"]); mac("crashFailed", c["failed"])
        mac("crashAt", c["crash"]); mac("crashLatS", f"{c['obs_s_mean']:.1f}"); mac("crashAUC", f"{c['test_auc']:.3f}")
    open(f"{OUT_TEX}/deploy_numbers.tex", "w").write("\n".join(m) + "\n")
    # table rows: one per run
    rows = []
    lab = {"bt1": "1\\,s blocks", "bt2": "2\\,s blocks", "bt5": "5\\,s blocks", "bt2_crash": "2\\,s, crash"}
    for r in ["bt1", "bt2", "bt5", "bt2_crash"]:
        if r not in A:
            continue
        a = A[r]
        rows.append(f"{lab[r]} & {a['rounds']} & {a['finalized']}/{a['rounds']} & {a['lat_blocks_mean']:.1f} & "
                    f"{a['obs_s_mean']:.1f} & {a['obs_s_p95']:.1f} & {a['cycle_s_mean']:.1f} & "
                    f"{a['tx']['commit']['lat']:.2f} & {a['tx']['reveal']['lat']:.2f} & "
                    f"{'yes' if a['agree'] else 'no'} \\\\")
    open(f"{OUT_TEX}/tab_deploy.tex", "w").write("\n".join(rows) + "\n")
    if "bt2" in A:
        tr = []
        nm = {"startRound": "\\textsc{startRound}", "commit": "\\textsc{commit}", "reveal": "\\textsc{reveal}",
              "vote": "\\textsc{vote}", "settleAll": "\\textsc{settleAll}", "closeExpired": "\\textsc{closeExpired}"}
        for k in ["startRound", "commit", "reveal", "vote", "settleAll", "closeExpired"]:
            if k in A["bt2"]["tx"]:
                t = A["bt2"]["tx"][k]
                tr.append(f"{nm[k]} & {t['n']} & {t['ok']} & {t['gas']/1000:.1f} & {t['lat']:.2f} & {t['lat_p95']:.2f} \\\\")
        open(f"{OUT_TEX}/tab_deploy_tx.tex", "w").write("\n".join(tr) + "\n")
    # figure: breakdown of the observed round latency per block time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["trained", "uploaded", "committed", "revealed", "fetched", "aggregated", "counted", "voted", "final"]
    groups = [("Train + encrypt + upload", ["trained", "uploaded"]), ("Commit (to inclusion)", ["committed"]),
              ("Wait $T_c$ + reveal", ["revealed"]), ("Wait $T_r$ + fetch/verify", ["fetched"]),
              ("Aggregate + counts", ["aggregated", "counted"]), ("Vote + finality", ["voted", "final"])]
    cols = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#b07aa1"]
    runs_f = [r for r in ["bt1", "bt2", "bt5", "bt2_crash"] if r in A]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    for y, r in enumerate(runs_f):
        left = 0
        for (g, ks), c in zip(groups, cols):
            v = sum(A[r]["breakdown"][k] for k in ks)
            ax.barh(y, v, left=left, color=c, label=g if y == 0 else None, edgecolor="white", linewidth=0.5)
            left += v
    ax.set_yticks(range(len(runs_f)))
    ax.set_yticklabels([lab[r].replace("\\,", " ") for r in runs_f])
    ax.invert_yaxis()
    ax.set_xlabel("Seconds from round start to finality (mean over hospitals and rounds)", fontsize=8)
    ax.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT_TEX}/deploy_latency.pdf")
    print(json.dumps({k: {kk: v[kk] for kk in ["rounds", "finalized", "failed", "agree", "lat_blocks_mean",
                                                 "obs_s_mean", "obs_s_p95", "cycle_s_mean", "test_auc"]}
                      for k, v in A.items()}, indent=1))


if __name__ == "__main__":
    main()
