"""Turns results/*.json into LaTeX tables, number macros and figures for the paper."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
OUT = os.environ.get("CTFL_TEX_OUT", "paper/generated")   # LaTeX tables and number macros
FIG = os.environ.get("CTFL_FIG_OUT", "paper/figures")      # figures
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
MAC = {}

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                          "#e87ba4", "#008300", "#6250d6", "#e34948")
GRAY = "#8a8985"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.6, "legend.frameon": False, "pdf.fonttype": 42})


def ms(x, d=3):
    x = np.asarray(x, float)
    return f"{x.mean():.{d}f}$\\pm${x.std():.{d}f}"


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def write(name, s):
    open(os.path.join(OUT, name), "w").write(s)


def macro(k, v):
    MAC[k] = v


# ------------------------------------------------------------------ E1
e1 = load("e1_final.json")
if e1:
    d = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in e1])
    order = ["Local-only", "Centralized", "FedAvg", "FedProx", "SCAFFOLD", "Uniform", "Balance-weighted",
             "Median", "Trimmed mean", "Krum", "CardioTrustFL"]
    names = {"Centralized": "Centralised", "Local-only": "Local only",
             "CardioTrustFL": "\\textbf{CTFL default}"}
    rows = []
    for m in order:
        cells = [names.get(m, m)]
        for scen in ["natural", "cleveland"]:
            for dp in [True, False]:
                x = d[(d.scenario == scen) & (d.method == m) & (d.dp == dp) & (d.rounds == 30)]
                if dp:
                    cells += [ms(x.auc), ms(x.bal_acc)]
                else:
                    cells += [ms(x.auc)]
        rows.append(" & ".join(cells) + " \\\\")
        if m in ("Centralized",):
            rows.append("\\midrule")
    write("tab_utility.tex", "\n".join(rows))
    # four-site columns only (main text); the full table goes to the supplementary material
    nat = []
    for r in rows:
        if "&" in r:
            f = [x.strip() for x in r.rstrip("\\ ").split("&")]
            nat.append(" & ".join(f[:4]) + " \\\\")
        else:
            nat.append(r)
    write("tab_utility_nat.tex", "\n".join(nat))
    g = lambda scen, m, dp=True, col="auc": d[(d.scenario == scen) & (d.method == m) & (d.dp == dp) & (d.rounds == 30)][col].mean()
    for scen, tag in [("natural", "Nat"), ("cleveland", "Cle")]:
        for m, mt in [("FedAvg", "FedAvg"), ("CardioTrustFL", "Ctfl"), ("Centralized", "Cent"), ("Local-only", "Local"),
                      ("Balance-weighted", "Bal"), ("Uniform", "Unif"), ("SCAFFOLD", "Scaf"), ("Krum", "Krum")]:
            macro(f"auc{tag}{mt}", f"{g(scen, m):.3f}")
            macro(f"bal{tag}{mt}", f"{g(scen, m, col='bal_acc'):.3f}")
            macro(f"aucNoDP{tag}{mt}", f"{g(scen, m, False):.3f}")
        e = d[(d.scenario == scen) & (d.method == "FedAvg") & (d.dp)]
    # epsilon per site (natural, sigma 1.5)
    nat = [r for r in e1 if r["scenario"] == "natural" and r["method"] == "FedAvg" and r["dp"] and r["rounds"] == 30][0]
    eps = nat["eps_per_site"]
    macro("EPSMIN", f"{min(eps):.2f}"); macro("EPSMAX", f"{max(eps):.2f}")
    for site, e in zip(["Hun", "Swi", "VA", "Cle"], eps):
        macro(f"eps{site}", f"{e:.2f}")
    cen = [r["eps_max"] for r in e1 if r["scenario"] == "natural" and r["method"] == "Centralized" and r["dp"]]
    macro("EPSCENT", f"{np.mean(cen):.2f}")
    cle = [r for r in e1 if r["scenario"] == "cleveland" and r["method"] == "FedAvg" and r["dp"]][0]
    macro("EPSMAXCle", f"{cle['eps_max']:.1f}")
    # calibration table
    rows = []
    for scen, sn in [("cleveland", "Clev.--Dir."), ("natural", "Four-site")]:
        for R_ in [10, 30]:
            for m in ["FedAvg", "Uniform", "Balance-weighted", "CardioTrustFL"]:
                x = d[(d.scenario == scen) & (d.method == m) & (d.dp) & (d.rounds == R_)]
                if len(x) == 0:
                    continue
                rows.append(" & ".join([sn if (R_ == 10 and m == "FedAvg") else "", str(R_) if m == "FedAvg" else "",
                                        names.get(m, m).replace("\\textbf{", "").replace("}", ""),
                                        f"{x.mean_p.mean():.2f}", ms(x.f1_05, 2), ms(x.bal_acc_05, 2),
                                        ms(x.auc, 3), ms(x.f1, 2), ms(x.bal_acc, 2)]) + " \\\\")
            rows.append("\\addlinespace")
    write("tab_calibration.tex", "\n".join(rows[:-1]))
    for m, mt in [("FedAvg", "FedAvg"), ("Balance-weighted", "Bal")]:
        x = d[(d.scenario == "cleveland") & (d.method == m) & (d.dp) & (d.rounds == 10)]
        macro(f"calF{mt}", f"{x.f1_05.mean():.2f}"); macro(f"calAUC{mt}", f"{x.auc.mean():.3f}")
        macro(f"calFcal{mt}", f"{x.f1.mean():.2f}"); macro(f"calP{mt}", f"{x.mean_p.mean():.2f}")
        x = d[(d.scenario == "cleveland") & (d.method == m) & (d.dp) & (d.rounds == 30)]
        macro(f"calPthirty{mt}", f"{x.mean_p.mean():.2f}"); macro(f"calBalfivethirty{mt}", f"{x.bal_acc_05.mean():.2f}")
    # loss ablation
    rows = []
    for lname in ["BCE", "Focal a=0.25", "Focal a=0.75"]:
        cells = [lname.replace("a=", "$\\alpha$=")]
        for scen in ["natural", "cleveland"]:
            for agg in ["fedavg", "balance"]:
                x = d[(d.scenario == scen) & (d.method == f"loss:{lname}|agg:{agg}")]
                cells += [ms(x.auc)]
        rows.append(" & ".join(cells) + " \\\\")
    write("tab_loss.tex", "\n".join(rows))
    for scen, tag in [("natural", "Nat"), ("cleveland", "Cle")]:
        macro(f"focal{tag}", f"{d[(d.scenario==scen)&(d.method=='loss:Focal a=0.25|agg:balance')].auc.mean():.3f}")
        macro(f"bce{tag}", f"{d[(d.scenario==scen)&(d.method=='loss:BCE|agg:balance')].auc.mean():.3f}")
    # per-site
    rows = []
    for m in ["Local-only", "FedAvg", "Balance-weighted", "CardioTrustFL"]:
        rr = [r for r in e1 if r["scenario"] == "natural" and r["method"] == m and r["dp"] and r["rounds"] == 30]
        if m == "Local-only":
            continue
        cells = [names.get(m, m)]
        for site in ["Hungary", "Switzerland", "VA", "Cleveland"]:
            sens = [r["sites"][site]["rec"] for r in rr]; spec = [r["sites"][site]["spec"] for r in rr]
            cells += [f"{np.mean(sens):.2f}/{np.mean(spec):.2f}"]
        rows.append(" & ".join(cells) + " \\\\")
    write("tab_sites.tex", "\n".join(rows))

# ------------------------------------------------------------------ E3
e3 = load("e3_calibration.json")
if e3:
    H = [r for r in e3 if r["kind"] == "honest"]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3), sharey=True)
    for ax, scen, title in [(axes[0], "natural", "Simulated four-site federation"), (axes[1], "pooled", "Pooled, K=10, Dirichlet(1.0)")]:
        xs, med_d, q1_d, q3_d, med_f, q1_f, q3_f = [], [], [], [], [], [], []
        for s in [None, 0.5, 1.0, 1.5, 2.0, 3.0]:
            rr = [r for r in H if r["scenario"] == scen and r["sigma"] == s]
            D = np.concatenate([np.array(r["dist"]).ravel() for r in rr])
            F = np.concatenate([np.array(r["dist_full"]).ravel() for r in rr])
            # offline application of the MAD rule per round
            mad_flag = []
            for r in rr:
                for row in r["dist"]:
                    row = np.array(row); m = np.median(row); mad = np.median(np.abs(row - m)) * 1.4826 + 1e-6
                    mad_flag += list(row > m + 3 * mad)
            xs.append(0 if s is None else s)
            med_d.append(np.median(D)); q1_d.append(np.quantile(D, .25)); q3_d.append(np.quantile(D, .75))
            med_f.append(np.median(F)); q1_f.append(np.quantile(F, .25)); q3_f.append(np.quantile(F, .75))
            if scen == "natural":
                rows.append([("no DP" if s is None else f"{s}"), f"{np.mean(D):.2f}", f"{np.mean(D > 0.3)*100:.0f}\\%",
                             f"{np.mean(F):.4f}", f"{np.mean(F > 0.3)*100:.0f}\\%", f"{np.mean(mad_flag)*100:.1f}\\%"])
            if scen == "natural" and s is None:
                macro("distNoDP", f"{np.mean(D):.2f}"); macro("fprFixedNoDP", f"{np.mean(D > 0.3)*100:.0f}")
            if scen == "natural" and s == 1.5:
                macro("distDP", f"{np.mean(D):.2f}"); macro("fprFixedDP", f"{np.mean(D > 0.3)*100:.0f}")
                macro("distFullDP", f"{np.mean(F):.4f}"); macro("fprMadDP", f"{np.mean(mad_flag)*100:.1f}")
        ax.fill_between(xs, q1_d, q3_d, color=BLUE, alpha=0.15, lw=0)
        ax.plot(xs, med_d, "-o", color=BLUE, ms=3.5, label="updates $\\Delta_i$ (median, IQR)")
        ax.fill_between(xs, q1_f, q3_f, color=ORANGE, alpha=0.15, lw=0)
        ax.plot(xs, med_f, "-s", color=ORANGE, ms=3.5, label="full weights $w_i$")
        ax.axhline(0.3, color=GRAY, ls="--", lw=1)
        ax.text(3.0, 0.32, "$\\tau=0.3$", ha="right", va="bottom", color="#52514e", fontsize=7)
        ax.set_xticks(xs); ax.set_xticklabels(["none" if x == 0 else str(x) for x in xs])
        ax.set_xlabel("DP noise multiplier $\\sigma$"); ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("Honest cosine distance to median")
    axes[1].legend(loc="center right", fontsize=7, bbox_to_anchor=(1.0, 0.42))
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "filter_calibration.pdf")); plt.close(fig)
    # detection
    Dd = pd.DataFrame([r for r in e3 if r["kind"] == "detect"])
    det = Dd.groupby(["sigma", "filt"], dropna=False)[["tpr", "fpr", "auc"]].mean()
    for row in rows:
        s = None if row[0] == "no DP" else float(row[0])
        for f in ["cos_fixed", "cos_fixed_full", "cos_mad"]:
            sel = Dd[(Dd.filt == f) & ((Dd.sigma.isna()) if s is None else (Dd.sigma == s))]
            row.append(f"{sel.tpr.mean()*100:.0f}/{sel.fpr.mean()*100:.0f}")
    write("tab_filter.tex", "\n".join(" & ".join(r) + " \\\\" for r in rows))

# ------------------------------------------------------------------ E2
e2 = load("e2_attacks.json")
DEF_ORDER = ["FedAvg", "Balance-wt.", "Median", "Trimmed mean", "Krum", "Cos-filter (conf.)", "Trimmed+clip", "Cos-filter MAD+clip"]
DEF_NAMES = {"Cos-filter (conf.)": "Cos $\\tau$=0.3", "Cos-filter MAD+clip": "\\textbf{CTFL}", "Trimmed+clip": "Trim.+clip",
             "Trimmed mean": "Trim.", "Balance-wt.": "Bal.", "Krum": "Krum", "Median": "Median", "FedAvg": "FedAvg"}
ATT_ORDER = [("none", None, "No attack"), ("signflip", "full", "Sign-flip ($-5\\times$)"), ("gauss", "full", "Gaussian"),
             ("labelflip", "full", "Label-flip"), ("lie", "full", "LIE, full"), ("lie", "partial", "LIE, partial"),
             ("minmax", "full", "Min-Max, full"), ("minmax", "partial", "Min-Max, partial"),
             ("adaptive", "full", "AGR-tailored, full"), ("adaptive", "partial", "AGR-tailored, partial")]
if e2:
    d = pd.DataFrame(e2)
    write("attack_header.tex", " & ".join(["Attack / knowledge"] + [DEF_NAMES[x] for x in DEF_ORDER]) + " \\\\")

    def attack_table(sel, fname):
        rows, worst = [], {k: 1.0 for k in DEF_ORDER}
        for kind, kn, label in ATT_ORDER:
            cells = [label]
            vals = []
            for df_ in DEF_ORDER:
                x = sel[(sel.attack == kind) & (sel.defence == df_) & ((sel.knowledge == kn) if kn else sel.knowledge.isna())]
                v = x.auc.mean(); vals.append(v)
                if kind != "none":
                    worst[df_] = min(worst[df_], v)
            best = np.nanmax(vals)
            cells += [(f"\\textbf{{{v:.2f}}}" if abs(v - best) < 0.0051 else f"{v:.2f}") for v in vals]
            rows.append(" & ".join(cells) + " \\\\")
            if kind == "none":
                rows.append("\\midrule")
        rows.append("\\midrule")
        wb = max(worst.values())
        rows.append(" & ".join(["Worst case"] + [(f"\\textbf{{{worst[k]:.2f}}}" if abs(worst[k]-wb) < 0.0051 else f"{worst[k]:.2f}") for k in DEF_ORDER]) + " \\\\")
        write(fname, "\n".join(rows))
        return worst
    wn = attack_table(d[d.scenario == "natural"], "tab_attack_natural.tex")
    p = d[d.scenario == "pooled"]
    w1 = attack_table(p[(p.m == 1) | (p.attack == "none")], "tab_attack_pooled1.tex")
    w2 = attack_table(p[(p.m == 2) | (p.attack == "none")], "tab_attack_pooled2.tex")
    macro("worstCtflNat", f"{wn['Cos-filter MAD+clip']:.2f}"); macro("worstCtflPoolOne", f"{w1['Cos-filter MAD+clip']:.2f}")
    macro("worstTrimNat", f"{wn['Trimmed mean']:.2f}"); macro("worstKrumNat", f"{wn['Krum']:.2f}")
    macro("worstFedAvgNat", f"{wn['FedAvg']:.2f}")
    # knowledge effects on CTFL
    c = d[d.defence == "Cos-filter MAD+clip"]
    for scen, tag in [("natural", "Nat"), ("pooled", "Pool")]:
        for kind, kt in [("adaptive", "Adapt"), ("lie", "Lie"), ("minmax", "Minmax")]:
            for kn in ["full", "partial"]:
                x = c[(c.scenario == scen) & (c.attack == kind) & (c.knowledge == kn) & ((c.m == 1) if scen == "pooled" else True)]
                macro(f"tpr{tag}{kt}{kn.capitalize()}", f"{x.tpr.mean()*100:.0f}")
                macro(f"auc{tag}{kt}{kn.capitalize()}", f"{x.auc.mean():.2f}")
    # figure: knowledge gap
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    defs = ["Median", "Trimmed+clip", "Cos-filter MAD+clip"]
    labs = ["Median", "Trimmed+clip", "CTFL default"]
    sel = d[(d.scenario == "natural")]
    width = 0.38
    for ax, kind, title in [(axes[0], "minmax", "Min-Max (Shejwalkar & Houmansadr)"), (axes[1], "adaptive", "AGR-tailored")]:
        xs = np.arange(len(defs))
        for j, (kn, col) in enumerate([("full", ORANGE), ("partial", BLUE)]):
            v = [sel[(sel.defence == df_) & (sel.attack == kind) & (sel.knowledge == kn)].auc.mean() for df_ in defs]
            ax.bar(xs + (j - .5) * width, v, width * 0.92, color=col, label=("full knowledge (no commitment)" if kn == "full" else "partial knowledge (commit–reveal)"))
            for xx, vv in zip(xs + (j - .5) * width, v):
                ax.text(xx, vv - 0.015, f"{vv:.2f}", ha="center", va="top", fontsize=6.5, color="white")
        clean = [sel[(sel.defence == df_) & (sel.attack == "none")].auc.mean() for df_ in defs]
        for xx, cv in zip(xs, clean):
            ax.plot([xx - width, xx + width], [cv, cv], color="#0b0b0b", lw=1, ls=":")
        ax.set_xticks(xs); ax.set_xticklabels(labs); ax.set_ylim(0.3, 1.02); ax.set_title(title, fontsize=8)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Test AUC (simulated four-site federation)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(os.path.join(FIG, "knowledge_gap.pdf")); plt.close(fig)

# ------------------------------------------------------------------ E4
e4 = load("e4_privacy.json")
if e4:
    d = pd.DataFrame(e4)
    rows = []
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for m, col, lab in [("CardioTrustFL", BLUE, "CTFL default"), ("FedAvg", ORANGE, "FedAvg")]:
        xs, ys, sd = [], [], []
        for s in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
            x = d[(d.method == m) & (d.sigma == s)]
            xs.append(np.mean([max(e) for e in x.eps])); ys.append(x.auc.mean()); sd.append(x.auc.std())
        ax.errorbar(xs, ys, yerr=sd, fmt="-o", ms=3.5, color=col, label=lab, capsize=2, lw=1.6)
        nodp = d[(d.method == m) & (d.sigma.isna())].auc.mean()
        ax.axhline(nodp, color=col, ls=":", lw=1)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
    tk = [3, 5, 7, 14, 25, 64]
    ax.xaxis.set_major_locator(FixedLocator(tk)); ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in tk]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Cumulative $\\varepsilon$ of the least protected site (30 rounds)")
    ax.set_ylabel("Test AUC"); ax.legend(fontsize=7, loc="lower right")
    ax.text(0.02, 0.97, "dotted: no DP", transform=ax.transAxes, fontsize=6.5, va="top", color="#52514e")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "privacy_utility.pdf")); plt.close(fig)
    for s in [None, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        x = d[(d.method == "CardioTrustFL") & ((d.sigma.isna()) if s is None else (d.sigma == s))]
        y = d[(d.method == "FedAvg") & ((d.sigma.isna()) if s is None else (d.sigma == s))]
        if s is None:
            epsc = ["$\\infty$"] * 4
        else:
            E = np.array(list(x.eps)); epsc = [f"{v:.2f}" for v in E.mean(0)]
        rows.append(" & ".join(["no DP" if s is None else f"{s}"] + epsc + [ms(x.auc), ms(x.bal_acc), ms(y.auc)]) + " \\\\")
    write("tab_privacy.tex", "\n".join(rows))
    for s, t in [(0.5, "Low"), (3.0, "High")]:
        x = d[(d.method == "CardioTrustFL") & (d.sigma == s)]
        macro(f"privAuc{t}", f"{x.auc.mean():.3f}"); macro(f"privEps{t}", f"{np.mean([max(e) for e in x.eps]):.1f}")
    macro("privAucNoDP", f"{d[(d.method=='CardioTrustFL') & d.sigma.isna()].auc.mean():.3f}")

# ------------------------------------------------------------------ E5
e5 = load("e5_scaling.json")
if e5:
    d = pd.DataFrame(e5)
    rows = []
    for K in [5, 10, 20]:
        for a in [0.3, 0.5, 1.0]:
            cells = [str(K) if a == 0.3 else "", f"{a}"]
            for m in ["FedAvg", "Balance-weighted", "CardioTrustFL"]:
                x = d[(d.K == K) & (d.alpha == a) & (d.method == m)]
                cells.append(ms(x.auc))
            x = d[(d.K == K) & (d.alpha == a)]
            cells.append(f"{x.eps_max.mean():.1f}")
            cells.append(f"{(x.train_s_per_round / K).mean()*1000:.0f}")
            rows.append(" & ".join(cells) + " \\\\")
        rows.append("\\addlinespace")
    write("tab_scaling.tex", "\n".join(rows[:-1]))

# ------------------------------------------------------------------ E6
e6 = load("e6_chain.json")
if e6:
    gs = e6["gas_scaling"]; ee = e6["end_to_end"]
    rows = []
    for op, lab in [("commit", "\\textsc{commit} (per hospital)"), ("reveal", "\\textsc{reveal} (per hospital)"),
                    ("vote", "\\textsc{vote} (per voter)"), ("startRound", "\\textsc{startRound}"),
                    ("settleAll", "\\textsc{settleAll}"), ("per_round_total", "\\textbf{Total per round}")]:
        rows.append(" & ".join([lab] + [f"{gs[k][op]/1000:,.1f}" for k in ["4", "10", "20", "50"]]) + " \\\\")
        if op == "settleAll":
            rows.append("\\midrule")
    rows.append(" & ".join(["Votes needed to finalise"] + [f"{gs[k]['votes_needed']:.0f}" for k in ["4", "10", "20", "50"]]) + " \\\\")
    rows.append(" & ".join(["Total per hospital per round"] + [f"{gs[k]['per_round_total']/int(k)/1000:,.1f}" for k in ["4", "10", "20", "50"]]) + " \\\\")
    write("tab_gas.tex", "\n".join(rows))
    macro("gasDeploy", f"{gs['4']['deploy']/1e6:.2f}")
    macro("gasRoundFour", f"{gs['4']['per_round_total']/1e6:.2f}")
    macro("gasRoundFifty", f"{gs['50']['per_round_total']/1e6:.1f}")
    macro("gasCommit", f"{gs['4']['commit']/1000:.1f}"); macro("gasReveal", f"{gs['4']['reveal']/1000:.1f}")
    macro("gasPerHospFifty", f"{gs['50']['per_round_total']/50/1000:.0f}")
    macro("gasPerHospFour", f"{gs['4']['per_round_total']/4/1000:.0f}")
    t = ee["time"]
    macro("timeTrain", f"{t['train']/ee['rounds']*1000:.0f}"); macro("timeAgg", f"{t['aggregate']/ee['rounds']/4*1000:.2f}")
    macro("checksPassed", str(sum(e6["adversarial_checks"].values()))); macro("checksTotal", str(len(e6["adversarial_checks"])))
    macro("eteHashEqual", "identical" if ee["final_hash_equals_simulation"] else "DIFFERENT")
    macro("eteAUC", f"{ee['sim_auc']:.3f}")
    macro("eteRepSwiss", str(ee["reputations"][1]))
    macro("blobKB", f"{ee['blob_bytes']/1000:.1f}")
    macro("encMs", f"{ee['time']['encrypt']/ (ee['rounds']*4) *1000:.2f}")
    chk = e6["adversarial_checks"]
    labels = {"quorum_is_N_minus_floor_N_minus_1_over_3": "Round quorum set to $N-\\lfloor (N-1)/3\\rfloor$ ($=3$ for $N=4$)",
              "early_reveal_rejected": "Reveal during the commit phase",
              "copied_opening_rejected": "Opening another hospital's commitment (copy / front-run)",
              "equivocating_opening_rejected": "Opening own commitment to a different update",
              "late_reveal_rejected": "Reveal after $T_r$",
              "non_revealer_cannot_vote": "Vote by a hospital that did not reveal",
              "two_votes_below_quorum_do_not_finalise": "Split votes (2 vs.\\ 1) below the quorum",
              "double_vote_rejected": "Second vote by the same hospital",
              "expired_round_closes_as_failed": "Round without quorum closes as failed after $T_v$",
              "withholder_slashed_and_penalised": "Withholder loses deposit and 15 reputation",
              "failed_round_no_reward_for_revealers": "No reward for revealers of a failed round",
              "single_byzantine_vote_cannot_finalise": "A single deviating vote cannot finalise",
              "not_final_with_q_minus_1_votes": "$q-1$ matching votes do not finalise",
              "finalised_at_quorum_on_honest_decision": "Finalisation at $q$ matching votes, on that decision",
              "excluded_update_penalised": "Excluded update costs 5 reputation",
              "kept_updates_rewarded_capped": "Kept update earns 5 reputation, capped at 100",
              "replay_previous_round_opening_rejected": "Replaying a previous round's opening"}
    write("tab_checks.tex", "\n".join(f"{labels[k]} & {'pass' if v else 'FAIL'} \\\\" for k, v in chk.items()))
    # gas figure
    Ks = [4, 10, 20, 50]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    tot = [gs[str(k)]["per_round_total"] / 1e6 for k in Ks]
    per = [gs[str(k)]["per_round_total"] / k / 1e3 for k in Ks]
    ax.plot(Ks, tot, "-o", color=BLUE, ms=3.5)
    for k, v in zip(Ks, tot):
        ax.text(k, v + 0.4, f"{v:.1f}M", ha="center", fontsize=6.5)
    ax.set_xlabel("Hospitals $n$"); ax.set_ylabel("Gas per round (millions)")
    ax.set_xticks(Ks)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "gas_scaling.pdf")); plt.close(fig)

with open(os.path.join(OUT, "numbers.tex"), "w") as f:
    for k, v in sorted(MAC.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(MAC), "macros;", sorted(os.listdir(OUT)))
