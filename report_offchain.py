"""LaTeX table for the off-chain scaling experiment (results/e8_offchain.json)."""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f"{H}/results/e8_offchain.json"))
N = [4, 10, 20, 50]
P = sorted({r["p"] for r in R})
cell = {(r["p"], r["n"]): r for r in R}
def fmt_mb(x):
    return f"{x/1000:.1f}\\,GB" if x >= 1000 else (f"{x:.0f}\\,MB" if x >= 10 else f"{x:.1f}\\,MB")
def fmt_s(x):
    return f"{x*1000:.0f}\\,ms" if x < 1 else f"{x:.1f}\\,s"
rows = []
for p in P:
    lab = {3713: "3,713 (ours)", 100000: "$10^5$", 1000000: "$10^6$", 5000000: "$5\\cdot10^6$", 25000000: "$2.5\\cdot10^7$"}[p]
    cs = []
    for n in N:
        r = cell.get((p, n))
        cs.append("--" if r is None or "skipped" in r else f"{fmt_mb(r['download_MB'])} / {fmt_s(r['compute_s'])}")
    rows.append(lab + " & " + " & ".join(cs) + " \\\\")
out = os.environ.get("OUT", f"{H}/results/tab_offchain.tex")
open(out, "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
