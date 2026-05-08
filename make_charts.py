"""
Regenerate eval_charts.png from the completed eval_results.csv
Clean, presentation-ready 4-panel figure.
"""
import csv, collections, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "eval_results.csv")
OUT_PATH   = os.path.join(SCRIPT_DIR, "eval_charts.png")

rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = "#0d1117"
PANEL = "#161b22"
BORD  = "#30363d"
GREEN = "#2ea44f"
ORANGE= "#e36209"
BLUE  = "#0969da"
RED   = "#f85149"
GOLD  = "#e3b341"
TEXT  = "#c9d1d9"
TITLE = "#e6edf3"

# ── Compute stats ─────────────────────────────────────────────────────────────
def stats(cfg_key):
    r = [x for x in rows if x["config"] == cfg_key]
    n = len(r)
    passes = [x for x in r if x["outcome"] == "pass"]
    p1  = sum(1 for x in r if x["pass_attempt"] == "1")
    p2  = sum(1 for x in r if x["pass_attempt"] == "2")
    p3  = sum(1 for x in r if x["pass_attempt"] == "3")
    p4p = sum(1 for x in r if int(x["pass_attempt"]) >= 4 and x["outcome"] == "pass")
    return {
        "n": n, "passes": len(passes),
        "rate": len(passes)/n*100,
        "p1_pct": p1/n*100, "p2_pct": p2/n*100,
        "p3_pct": p3/n*100, "p4p_pct": p4p/n*100,
        "fail_pct": (n - len(passes))/n*100,
    }

full_s    = stats("full")
nocorr_s  = stats("no_corr")
onecorr_s = stats("one_corr")

# By design type
full_rows = [x for x in rows if x["config"] == "full"]
by_type   = collections.defaultdict(list)
for x in full_rows:
    by_type[x["type"]].append(x["outcome"] == "pass")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.patch.set_facecolor(BG)

for ax in axes.flat:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(BORD)

# ────────────────────────────────────────────────────────
# Chart 1 — Ablation: Overall Pass Rate
# ────────────────────────────────────────────────────────
ax1 = axes[0, 0]
cfg_labels = ["No Corrections\n(1 attempt)", "1 Correction\n(2 attempts)", "Full System\n(3 corrections)"]
rates      = [nocorr_s["rate"], onecorr_s["rate"], full_s["rate"]]
colors     = [RED, ORANGE, GREEN]
xs = [0, 1, 2]
bars = ax1.bar(xs, rates, color=colors, width=0.5, edgecolor=BORD, linewidth=0.8, zorder=3)
ax1.set_xticks(xs)
ax1.set_xticklabels(cfg_labels, fontsize=10, color=TEXT)
ax1.set_ylim(0, 100)
ax1.set_ylabel("Pass Rate (%)", color=TEXT, fontsize=11)
ax1.set_title("Ablation Study — Correction Loop Impact", color=TITLE, fontsize=12, pad=12, fontweight="bold")
ax1.axhline(y=100, color=BORD, linestyle="--", linewidth=0.6, zorder=1)
ax1.grid(axis="y", color=BORD, linewidth=0.4, zorder=0)

for bar, rate, n_pass in zip(bars, rates, [nocorr_s["passes"], onecorr_s["passes"], full_s["passes"]]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f"{rate:.1f}%\n({n_pass}/{full_s['n']})",
             ha="center", va="bottom", color=TITLE, fontsize=11, fontweight="bold")

# Improvement arrow
ax1.annotate("", xy=(2, full_s["rate"]-2), xytext=(0, nocorr_s["rate"]+2),
             arrowprops=dict(arrowstyle="->", color=GOLD, lw=2))
ax1.text(0.95, (nocorr_s["rate"] + full_s["rate"]) / 2,
         f"+{full_s['rate']-nocorr_s['rate']:.1f}%",
         color=GOLD, fontsize=12, fontweight="bold", transform=ax1.transData,
         ha="center")

# ────────────────────────────────────────────────────────
# Chart 2 — Full system breakdown by attempt
# ────────────────────────────────────────────────────────
ax2 = axes[0, 1]
cats   = ["Pass@1", "Pass@2", "Pass@3", "Pass@4+", "Fail"]
vals   = [full_s["p1_pct"], full_s["p2_pct"], full_s["p3_pct"],
          full_s["p4p_pct"], full_s["fail_pct"]]
colors2 = [GREEN, "#56d364", "#94d82d", GOLD, RED]
bot = 0
for cat, val, col in zip(cats, vals, colors2):
    if val > 0:
        ax2.bar([0], [val], bottom=bot, color=col, width=0.55,
                edgecolor=BORD, linewidth=0.8, label=cat, zorder=3)
        if val >= 4:
            raw_count = round(val / 100 * full_s["n"])
            ax2.text(0, bot + val/2, f"{val:.0f}%  ({raw_count})",
                     ha="center", va="center", color="white",
                     fontsize=12, fontweight="bold")
        bot += val

ax2.set_xlim(-0.5, 0.5)
ax2.set_ylim(0, 108)
ax2.set_xticks([])
ax2.set_ylabel("% of 28 Prompts", color=TEXT, fontsize=11)
ax2.set_title("Full System — Pass Attempt Breakdown", color=TITLE, fontsize=12, pad=12, fontweight="bold")
ax2.legend(loc="upper right", facecolor="#21262d", edgecolor=BORD,
           labelcolor=TEXT, fontsize=10, framealpha=0.9)

# ────────────────────────────────────────────────────────
# Chart 3 — By design type (Full system)
# ────────────────────────────────────────────────────────
ax3 = axes[1, 0]
type_order = sorted(by_type.keys(), key=lambda t: -(sum(by_type[t])/len(by_type[t])))
type_labels  = type_order
type_rates   = [sum(by_type[t])/len(by_type[t])*100 for t in type_order]
bar_colors3  = [GREEN if r==100 else (ORANGE if r>=50 else RED) for r in type_rates]
ys = range(len(type_labels))

bars3 = ax3.barh(list(ys), type_rates, color=bar_colors3, edgecolor=BORD, linewidth=0.7, zorder=3)
ax3.set_yticks(list(ys))
ax3.set_yticklabels(type_labels, fontsize=10, color=TEXT)
ax3.set_xlim(0, 125)
ax3.set_xlabel("Pass Rate (%)", color=TEXT, fontsize=11)
ax3.set_title("Pass Rate by Design Type (Full System)", color=TITLE, fontsize=12, pad=12, fontweight="bold")
ax3.axvline(x=100, color=BORD, linestyle="--", linewidth=0.6, zorder=1)
ax3.grid(axis="x", color=BORD, linewidth=0.4, zorder=0)

for bar, pct, t in zip(bars3, type_rates, type_labels):
    n_items = len(by_type[t])
    n_pass  = sum(by_type[t])
    ax3.text(pct + 1.5, bar.get_y() + bar.get_height()/2,
             f"{pct:.0f}%  ({n_pass}/{n_items})",
             va="center", color=TITLE, fontsize=9)

# ────────────────────────────────────────────────────────
# Chart 4 — Complexity breakdown: combinational vs sequential
# ────────────────────────────────────────────────────────
ax4 = axes[1, 1]

COMBO_TYPES = {"gate", "mux", "adder", "decoder", "encoder", "alu", "combinational"}
SEQ_SIMPLE  = {"dff", "tff", "jkff", "counter", "shift_reg"}
SEQ_COMPLEX = {"fsm", "lfsr", "johnson", "fifo", "uart"}

def group_rate(type_set):
    items = []
    for t in type_set:
        items.extend(by_type.get(t, []))
    return (sum(items)/len(items)*100) if items else 0

cats4  = ["Combinational\n(gates, adders,\nmux, ALU)", "Simple\nSequential\n(DFF, counters,\nshift regs)", "Complex\nSequential\n(FSM, FIFO,\nLFSR, UART)"]
rates4 = [group_rate(COMBO_TYPES), group_rate(SEQ_SIMPLE), group_rate(SEQ_COMPLEX)]
col4   = [GREEN, ORANGE, RED]

for i, (cat, rate, col) in enumerate(zip(cats4, rates4, col4)):
    ax4.bar([i], [rate], color=col, width=0.55, edgecolor=BORD, linewidth=0.8, zorder=3)
    ax4.text(i, rate + 1.5, f"{rate:.0f}%", ha="center", va="bottom",
             color=TITLE, fontsize=14, fontweight="bold")

ax4.set_xticks([0, 1, 2])
ax4.set_xticklabels(cats4, fontsize=10, color=TEXT)
ax4.set_ylim(0, 120)
ax4.set_ylabel("Pass Rate (%)", color=TEXT, fontsize=11)
ax4.set_title("Pass Rate by Circuit Complexity", color=TITLE, fontsize=12, pad=12, fontweight="bold")
ax4.axhline(y=100, color=BORD, linestyle="--", linewidth=0.6, zorder=1)
ax4.grid(axis="y", color=BORD, linewidth=0.4, zorder=0)

# ── Final ─────────────────────────────────────────────────────────────────────
fig.suptitle("HDL Sentinel — Evaluation Results  |  28 Benchmark Prompts  |  CPSC 597",
             color=TITLE, fontsize=13, fontweight="bold", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3.5, w_pad=2.5)
plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"Chart saved: {OUT_PATH}")
plt.close()
