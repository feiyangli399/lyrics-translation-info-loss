# -*- coding: utf-8 -*-
"""Generate Fig 1 (radar) and Fig 3 (PCA scatter) for the paper from real results.
Run: python pipeline/make_paper_figs.py
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DIMS = ["l_sem", "l_csi", "l_rhyme", "l_meter", "l_emo"]
DIM_LABELS = ["sem", "csi", "rhyme", "meter", "emo"]

# ---------- load 5-seed means (ilv_across_seeds.csv: "mean,sd,n" strings) ----------
def parse_cell(v):
    return float(v.split(",")[0]) if v else np.nan

cells = {}  # (system, direction, style) -> {dim: mean}
with open(os.path.join(RES, "ilv_across_seeds.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        key = (row["system"], row["direction"], row["style"])
        cells[key] = {d: parse_cell(row[f"{d}_mean"]) for d in DIMS}

# ---------- Fig 1: radar, 2 systems x 2 directions, mean over 3 styles ----------
styles = ["guofeng", "pop", "folk"]
agg = {}  # (system, direction) -> dim -> list of means
for (sys_, dir_, sty), dv in cells.items():
    agg.setdefault((sys_, dir_), {d: [] for d in DIMS})
    for d in DIMS:
        agg[(sys_, dir_)][d].append(dv[d])

series = {}
for k, dv in agg.items():
    series[k] = [np.nanmean(dv[d]) for d in DIMS]

angles = np.linspace(0, 2 * np.pi, len(DIMS), endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), subplot_kw=dict(polar=True))
colors = {"NMT": "#4878cf", "LLM": "#d65f5f"}
for ax, dir_ in zip(axes, ["zh2en", "en2zh"]):
    for sys_ in ["NMT", "LLM"]:
        vals = series[(sys_, dir_)]
        vals = vals + vals[:1]
        ax.plot(angles, vals, color=colors[sys_], linewidth=2, label=sys_)
        ax.fill(angles, vals, color=colors[sys_], alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIM_LABELS, fontsize=11)
    ax.set_ylim(0, 0.45)
    ax.set_title(f"{dir_}  (style-mean)", fontsize=12)
    ax.grid(alpha=0.3)
axes[0].legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
fig.suptitle("Fig. 1  ILV five-dimension radar (system x direction, 5-seed mean)", y=1.03, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(RES, "fig_radar_ilv.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_radar_ilv.png done")

# ---------- Fig 3: PCA scatter of per-song ILV, seed=42 ----------
songs = {}  # (song_id, system, direction, style) -> dim -> list
with open(os.path.join(RES, "seed42", "ilv_per_pair.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        key = (row["song_id"], row["system"], row["direction"], row["style"])
        rec = songs.setdefault(key, {d: [] for d in DIMS})
        for d in DIMS:
            v = row[d]
            if v != "":
                rec[d].append(float(v))

X, meta = [], []
for (sid, sys_, dir_, sty), dv in songs.items():
    vec = [np.mean(dv[d]) if dv[d] else np.nan for d in DIMS]
    if np.isnan(vec).any():
        # fill missing l_emo with cell mean later; simplest: impute with dim median after collecting
        X.append(vec)
    else:
        X.append(vec)
    meta.append((sys_, dir_, sty))

X = np.array(X, dtype=float)
# impute NaN (l_emo undefined) with column median
for j in range(X.shape[1]):
    col = X[:, j]
    med = np.nanmedian(col)
    col[np.isnan(col)] = med
    X[:, j] = col

pca = PCA(n_components=2, random_state=42)
Z = pca.fit_transform(X)
var = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(8, 6))
mk = {"NMT": "o", "LLM": "s"}
col = {"zh2en": "#4878cf", "en2zh": "#d65f5f"}
for sys_ in ["NMT", "LLM"]:
    for dir_ in ["zh2en", "en2zh"]:
        idx = [i for i, (s, d, _) in enumerate(meta) if s == sys_ and d == dir_]
        ax.scatter(Z[idx, 0], Z[idx, 1], marker=mk[sys_], c=col[dir_],
                   s=18, alpha=0.55, edgecolors="none",
                   label=f"{sys_} {dir_}")
ax.set_xlabel(f"PC1 ({var[0]:.1f}% var)")
ax.set_ylabel(f"PC2 ({var[1]:.1f}% var)")
ax.set_title("Fig. 3  Per-song ILV projected to 2-D PCA (seed 42)")
ax.legend(fontsize=9, framealpha=0.8)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(RES, "fig_pca_ilv.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"fig_pca_ilv.png done, PC1={var[0]:.1f}% PC2={var[1]:.1f}%, n={len(X)}")
