"""
Exercise 2 - Non-Linearity in Higher Dimensions
Generates Figures 4-5 and the numbers used in the report / results summary.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json

rng = np.random.default_rng(42)

FIG_DIR = "../figures"
N = 500
D = 5

# ----- A: Dataset I - shifted Gaussians ---------------------------------------
mu_A = np.zeros(D)
Sigma_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])
mu_B = np.full(D, 1.5)
Sigma_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

X_A = rng.multivariate_normal(mu_A, Sigma_A, size=N)
X_B = rng.multivariate_normal(mu_B, Sigma_B, size=N)
dataset1 = np.vstack([X_A, X_B])
labels1 = np.array([0] * N + [1] * N)

# ----- B: Dataset II - concentric shells ---------------------------------------
def sample_shell(n, radius_mean, radius_std, dim, rng):
    v = rng.normal(size=(n, dim))
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    rho = rng.normal(radius_mean, radius_std, size=(n, 1))
    return rho * u

X_C = sample_shell(N, 2.0, 0.4, D, rng)   # core
X_D = sample_shell(N, 5.0, 0.4, D, rng)   # shell
dataset2 = np.vstack([X_C, X_D])
labels2 = np.array([0] * N + [1] * N)

# ----- C: Visualize and compare -------------------------------------------------
pca1 = PCA(n_components=2).fit(dataset1)
proj1 = pca1.transform(dataset1)
ev1 = pca1.explained_variance_ratio_

pca2 = PCA(n_components=2).fit(dataset2)
proj2 = pca2.transform(dataset2)
ev2 = pca2.explained_variance_ratio_

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for lab, name, col in [(0, "Class A", "#1f77b4"), (1, "Class B", "#ff7f0e")]:
    m = labels1 == lab
    axes[0].scatter(proj1[m, 0], proj1[m, 1], s=14, alpha=0.6, color=col, label=name)
axes[0].set_title(f"Dataset I (shifted Gaussians)\nPC1+PC2 explained var = {ev1[:2].sum():.3f}")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].legend()

for lab, name, col in [(0, "Class C (core)", "#2ca02c"), (1, "Class D (shell)", "#d62728")]:
    m = labels2 == lab
    axes[1].scatter(proj2[m, 0], proj2[m, 1], s=14, alpha=0.6, color=col, label=name)
axes[1].set_title(f"Dataset II (concentric shells)\nPC1+PC2 explained var = {ev2[:2].sum():.3f}")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].legend()

fig.suptitle("Figure 4 — PCA projection to 2D")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig4_pca_projection.png", dpi=150)
plt.close(fig)

# distance between class centers (5D)
dist1 = np.linalg.norm(X_A.mean(axis=0) - X_B.mean(axis=0))
dist2 = np.linalg.norm(X_C.mean(axis=0) - X_D.mean(axis=0))

# Figure 5: radius histograms
radius1_A = np.linalg.norm(X_A, axis=1)
radius1_B = np.linalg.norm(X_B, axis=1)
radius2_C = np.linalg.norm(X_C, axis=1)
radius2_D = np.linalg.norm(X_D, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(radius1_A, bins=30, alpha=0.6, label="Class A", color="#1f77b4")
axes[0].hist(radius1_B, bins=30, alpha=0.6, label="Class B", color="#ff7f0e")
axes[0].set_title("Dataset I — ||x|| per class")
axes[0].set_xlabel("||x||")
axes[0].set_ylabel("count")
axes[0].legend()

axes[1].hist(radius2_C, bins=30, alpha=0.6, label="Class C (core)", color="#2ca02c")
axes[1].hist(radius2_D, bins=30, alpha=0.6, label="Class D (shell)", color="#d62728")
axes[1].set_title("Dataset II — ||x|| per class")
axes[1].set_xlabel("||x||")
axes[1].set_ylabel("count")
axes[1].legend()

fig.suptitle("Figure 5 — Radius histograms (5D)")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig5_radius_histograms.png", dpi=150)
plt.close(fig)

results = {
    "dist_centers_dataset1": float(dist1),
    "dist_centers_dataset2": float(dist2),
    "explained_var_pc1pc2_dataset1": float(ev1[:2].sum()),
    "explained_var_pc1pc2_dataset2": float(ev2[:2].sum()),
}
with open("ex2_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
