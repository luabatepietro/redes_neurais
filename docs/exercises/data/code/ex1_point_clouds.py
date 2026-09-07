"""
Exercise 1 - Point Clouds: Geometry and Spread in 2D
Generates Figures 1-3 and the numbers used in the report / results summary.
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
import json

rng = np.random.default_rng(42)

FIG_DIR = "../figures"

# ----- A: Generate the clouds -------------------------------------------------
params = {
    0: {"mean": np.array([2.0, 3.0]), "std": np.array([0.8, 2.5])},
    1: {"mean": np.array([5.0, 6.0]), "std": np.array([1.2, 1.9])},
    2: {"mean": np.array([8.0, 1.0]), "std": np.array([0.9, 0.9])},
    3: {"mean": np.array([15.0, 4.0]), "std": np.array([0.5, 2.0])},
}
N_PER_CLASS = 100
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def make_clouds(scale=1.0):
    """Return dict {class: (N,2) array} using std * scale, same rng stream."""
    clouds = {}
    for c, p in params.items():
        pts = rng.normal(loc=p["mean"], scale=p["std"] * scale, size=(N_PER_CLASS, 2))
        clouds[c] = pts
    return clouds


clouds_s1 = make_clouds(1.0)

fig, ax = plt.subplots(figsize=(7, 6))

# Sketch of the decision boundaries a trained network might learn: nearest-centroid
# (Voronoi) regions over the 4 class means, shown as shaded background + contour lines.
means_grid = np.array([params[c]["mean"] for c in range(4)])
gx = np.linspace(-2, 18, 400)
gy = np.linspace(-4, 12, 400)
GX, GY = np.meshgrid(gx, gy)
grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=1)
d = np.linalg.norm(grid_pts[:, None, :] - means_grid[None, :, :], axis=2)
nearest = d.argmin(axis=1).reshape(GX.shape)
from matplotlib.colors import ListedColormap
ax.contourf(GX, GY, nearest, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
            colors=colors, alpha=0.12)
ax.contour(GX, GY, nearest, levels=[0.5, 1.5, 2.5, 3.5], colors="black",
           linewidths=1.2, linestyles="--")

for c, pts in clouds_s1.items():
    ax.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.7, color=colors[c], label=f"Class {c}")
    ax.scatter(*params[c]["mean"], marker="X", s=180, color=colors[c],
               edgecolor="black", linewidth=1.2, zorder=5)
ax.set_xlim(-2, 18)
ax.set_ylim(-4, 12)
ax.set_title("Figure 1 — Point clouds (s = 1.0), centers, and sketched decision boundaries")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig1_point_clouds.png", dpi=150)
plt.close(fig)

# ----- B: More or less spread out ---------------------------------------------
scales = [0.5, 1.0, 2.0, 4.0]
all_datasets = {s: make_clouds(s) for s in scales}

# Figure 2: 4 subplots, shared axis limits
all_pts = np.vstack([pts for ds in all_datasets.values() for pts in ds.values()])
xlim = (all_pts[:, 0].min() - 1, all_pts[:, 0].max() + 1)
ylim = (all_pts[:, 1].min() - 1, all_pts[:, 1].max() + 1)

fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
for ax, s in zip(axes, scales):
    for c, pts in all_datasets[s].items():
        ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.7, color=colors[c])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"s = {s}")
    ax.set_xlabel("x1")
axes[0].set_ylabel("x2")
fig.suptitle("Figure 2 — Same 4 classes at increasing spread (shared axes)")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig2_spread_subplots.png", dpi=150)
plt.close(fig)

# Separation ratio r_ij at s = 1 (uses the *base* stds, not the s=1 dataset stds,
# since r_ij is defined from the generating parameters)
sigma_bar = {c: (p["std"][0] + p["std"][1]) / 2 for c, p in params.items()}
pairs = list(itertools.combinations(range(4), 2))
r_ij = {}
for (i, j) in pairs:
    dist = np.linalg.norm(params[i]["mean"] - params[j]["mean"])
    r_ij[(i, j)] = dist / (sigma_bar[i] + sigma_bar[j])

smallest_pair = min(r_ij, key=r_ij.get)
smallest_r = r_ij[smallest_pair]
smallest_r_at_s2 = smallest_r / 2.0  # r scales as 1/s

# Mixing rate per scale: fraction of points whose nearest class MEAN is not their own class
mixing_rates = {}
means_arr = np.array([params[c]["mean"] for c in range(4)])
for s in scales:
    ds = all_datasets[s]
    mismatches = 0
    total = 0
    for c, pts in ds.items():
        d = np.linalg.norm(pts[:, None, :] - means_arr[None, :, :], axis=2)  # (N,4)
        nearest = d.argmin(axis=1)
        mismatches += np.sum(nearest != c)
        total += len(pts)
    mixing_rates[s] = mismatches / total

# Figure 3: mixing rate x s
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(scales, [mixing_rates[s] for s in scales], marker="o", color="#333333")
ax.set_xlabel("scale factor s")
ax.set_ylabel("mixing rate")
ax.set_title("Figure 3 — Mixing rate vs. spread scale s")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig3_mixing_rate.png", dpi=150)
plt.close(fig)

results = {
    "r_ij": {f"{i}-{j}": float(v) for (i, j), v in r_ij.items()},
    "smallest_pair": smallest_pair,
    "smallest_r_s1": float(smallest_r),
    "smallest_r_s2": float(smallest_r_at_s2),
    "mixing_rates": {str(s): float(v) for s, v in mixing_rates.items()},
}
with open("ex1_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
