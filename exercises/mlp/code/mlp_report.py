"""
3. MLP — Exercises 1 a 4.

Um único script, com UM único gerador aleatório (rng = default_rng(42)) usado em
todo o relatório, na ordem do enunciado. Rodar de dentro de code/:

    python mlp_report.py

Gera as Figuras 1-6 em ../figures/ e imprime todos os números citados no relatório
(também salvos em results.json).

Bibliotecas: numpy e matplotlib. O scikit-learn aparece SÓ onde o enunciado permite:
geração dos dados (make_blobs, make_classification), split (train_test_split),
matriz de confusão e o baseline linear (LogisticRegression). O MLP — ativações,
perda, forward, gradientes e atualização — é escrito à mão com numpy.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Seed fixa: o MESMO rng é usado do começo ao fim do relatório
# (inicialização dos pesos e embaralhamento dos mini-batches dos Ex. 2, 3 e 4).
rng = np.random.default_rng(42)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
RESULTS = {}


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=130)
    plt.close(fig)


# =============================================================================
# Exercise 1 — conferência numérica da conta feita à mão no relatório.
# (A derivação está no index.md; isto só confere os números passo a passo.)
# =============================================================================
# --8<-- [start:ex1]
def exercise1():
    x = np.array([0.5, -0.2])
    y = 1.0
    W1 = np.array([[0.3, -0.1],
                   [0.2, 0.4]])          # W^(1): linha i = pesos do neurônio oculto i
    b1 = np.array([0.1, -0.2])
    W2 = np.array([0.5, -0.3])           # W^(2): 1 neurônio de saída
    b2 = 0.2
    eta = 0.3

    # A — forward
    z1 = W1 @ x + b1
    h1 = np.tanh(z1)
    u2 = W2 @ h1 + b2
    y_hat = np.tanh(u2)
    L = (y - y_hat) ** 2

    # B — backward (regra da cadeia, tanh'(u) = 1 - tanh(u)^2)
    dL_dyhat = -2.0 * (y - y_hat)
    dL_du2 = dL_dyhat * (1.0 - y_hat ** 2)
    dL_dW2 = dL_du2 * h1
    dL_db2 = dL_du2
    dL_dh1 = dL_du2 * W2
    dL_dz1 = dL_dh1 * (1.0 - h1 ** 2)
    dL_dW1 = np.outer(dL_dz1, x)
    dL_db1 = dL_dz1

    # C — update
    W2_new = W2 - eta * dL_dW2
    b2_new = b2 - eta * dL_db2
    W1_new = W1 - eta * dL_dW1
    b1_new = b1 - eta * dL_db1

    # Segundo forward, para conferir a direção da perda
    h1_2 = np.tanh(W1_new @ x + b1_new)
    L_2 = (y - np.tanh(W2_new @ h1_2 + b2_new)) ** 2

    return dict(z1=z1, h1=h1, u2=u2, y_hat=y_hat, L=L,
                dL_dyhat=dL_dyhat, dL_du2=dL_du2, dL_dW2=dL_dW2, dL_db2=dL_db2,
                dL_dh1=dL_dh1, dL_dz1=dL_dz1, dL_dW1=dL_dW1, dL_db1=dL_db1,
                W2_new=W2_new, b2_new=b2_new, W1_new=W1_new, b1_new=b1_new,
                L_second_pass=L_2)
# --8<-- [end:ex1]


# =============================================================================
# O MLP — escrito do zero. A MESMA classe serve os Exercises 2, 3 e 4:
# só mudam os argumentos (layer_sizes, loss, lr), nunca o código.
# =============================================================================
# --8<-- [start:mlp]
def sigmoid(u):
    # Forma numericamente estável: evita exp de números grandes positivos.
    out = np.empty_like(u)
    pos = u >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-u[pos]))
    e = np.exp(u[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def softmax(u):
    # Subtrair o máximo da linha não muda o resultado e evita overflow.
    e = np.exp(u - u.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def binary_cross_entropy(P, Y, eps=1e-12):
    P = np.clip(P, eps, 1 - eps)
    return float(-np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P)))


def categorical_cross_entropy(P, Y, eps=1e-12):
    return float(-np.mean(np.sum(Y * np.log(np.clip(P, eps, 1.0)), axis=1)))


# Cada perda vem com a ativação de saída que a "casa": sigmoid+BCE e softmax+CCE.
# Nos dois pares, dL/du da saída é (P - Y)/N — por isso o backward é um só.
LOSSES = {
    "bce": (sigmoid, binary_cross_entropy),
    "cce": (softmax, categorical_cross_entropy),
}


class MLP:
    """
    MLP totalmente conectado, tanh nas camadas ocultas.

    layer_sizes: [n_entradas, n_oculta_1, ..., n_oculta_k, n_saidas]
    loss:        "bce" (saída sigmoid, 1 unidade) ou "cce" (saída softmax, C unidades)

    Convenção de formas: X é (N, d); W[l] é (n_in, n_out), então a camada faz
    U = A_prev @ W + b  — é o W^(l) do enunciado transposto, aplicado a N amostras.
    """

    def __init__(self, layer_sizes, loss, lr, rng):
        self.sizes = list(layer_sizes)
        self.out_act, self.loss_fn = LOSSES[loss]
        self.lr = lr
        self.rng = rng
        # Inicialização Xavier/Glorot (normal), adequada a tanh.
        self.W, self.b = [], []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            std = np.sqrt(2.0 / (n_in + n_out))
            self.W.append(rng.normal(0.0, std, size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))

    def n_params(self):
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))

    # ---- forward ------------------------------------------------------------
    def forward(self, X):
        """Guarda as ativações de cada camada (A[0] = X) para o backward."""
        self.A = [X]
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            U = self.A[-1] @ W + b
            last = l == len(self.W) - 1
            self.A.append(self.out_act(U) if last else np.tanh(U))
        return self.A[-1]

    # ---- backward -----------------------------------------------------------
    def backward(self, Y):
        """
        Gradientes de L (média sobre o batch) em relação a todos os W e b.
        delta = dL/dU da camada atual.
          saída:   delta = (P - Y) / N            (sigmoid+BCE ou softmax+CCE)
          camada:  dW = A_prev^T delta,  db = soma de delta nas amostras
          descer:  delta_prev = (delta W^T) ⊙ (1 - A_prev^2)   (derivada da tanh)
        """
        N = Y.shape[0]
        delta = (self.A[-1] - Y) / N
        self.dW = [None] * len(self.W)
        self.db = [None] * len(self.b)
        for l in range(len(self.W) - 1, -1, -1):
            A_prev = self.A[l]
            self.dW[l] = A_prev.T @ delta
            self.db[l] = delta.sum(axis=0)
            if l > 0:
                delta = (delta @ self.W[l].T) * (1.0 - A_prev ** 2)

    # ---- update -------------------------------------------------------------
    def step(self):
        """Gradiente descendente: θ ← θ - η ∂L/∂θ."""
        for l in range(len(self.W)):
            self.W[l] -= self.lr * self.dW[l]
            self.b[l] -= self.lr * self.db[l]

    # ---- treino / predição --------------------------------------------------
    def fit(self, X, Y, epochs, batch_size=32):
        """Mini-batch SGD; embaralha com o rng do relatório a cada época.
        Registra a perda no treino inteiro ao fim de cada época."""
        history = []
        N = X.shape[0]
        for _ in range(epochs):
            idx = self.rng.permutation(N)
            for start in range(0, N, batch_size):
                batch = idx[start:start + batch_size]
                self.forward(X[batch])
                self.backward(Y[batch])
                self.step()
            history.append(self.loss_fn(self.forward(X), Y))
        return np.array(history)

    def predict(self, X):
        P = self.forward(X)
        if P.shape[1] == 1:
            return (P[:, 0] >= 0.5).astype(int)
        return P.argmax(axis=1)
# --8<-- [end:mlp]


def one_hot(y, n_classes):
    return np.eye(n_classes)[y]


# =============================================================================
# Exercise 2 — binário, não-linearmente separável
# =============================================================================
# --8<-- [start:ex2_data]
centers = np.array([[0., 0.], [3., 3.], [-3., -3.]])
X2, c2 = make_blobs(n_samples=[500, 250, 250], centers=centers,
                    cluster_std=1.2, random_state=42)
y2 = (c2 > 0).astype(int)      # classe 0: 1 cluster | classe 1: 2 clusters

X2_tr, X2_te, y2_tr, y2_te = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2)
# --8<-- [end:ex2_data]

# Figura 1 — os 1000 pontos coloridos por y
fig, ax = plt.subplots(figsize=(6.5, 5.5))
for k in (0, 1):
    m = y2 == k
    ax.scatter(X2[m, 0], X2[m, 1], s=12, alpha=0.7, c=COLORS[k],
               label=f"Classe {k} ({m.sum()} pontos)")
ax.set_title("Figura 1 — Dataset do Exercise 2 (make_blobs, 1000 pontos)")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend(title="Classe")
ax.set_aspect("equal")
savefig(fig, "fig1_ex2_data.png")

# --8<-- [start:ex2_baseline]
baseline2 = LogisticRegression().fit(X2_tr, y2_tr)
base2_acc = baseline2.score(X2_te, y2_te)
# --8<-- [end:ex2_baseline]

# Para o item D: o que a reta faz com cada um dos 3 blobs do teste?
# (refaz o mesmo split carregando junto o rótulo de blob c2; a partição é idêntica)
_, _, _, _, _, c2_te = train_test_split(X2, y2, c2, test_size=0.2,
                                        random_state=42, stratify=y2)
pred_base2 = baseline2.predict(X2_te)
base2_by_blob = {int(k): dict(n=int(np.sum(c2_te == k)),
                              correct=int(np.sum((pred_base2 == y2_te) & (c2_te == k))))
                 for k in range(3)}
base2_proba = baseline2.predict_proba(X2_te)[:, 1]

# --8<-- [start:ex2_train]
ARCH2, LR2, EPOCHS2 = [2, 8, 1], 0.05, 300
mlp2 = MLP(ARCH2, loss="bce", lr=LR2, rng=rng)
hist2 = mlp2.fit(X2_tr, y2_tr.reshape(-1, 1).astype(float), epochs=EPOCHS2)
acc2 = float(np.mean(mlp2.predict(X2_te) == y2_te))
# --8<-- [end:ex2_train]
train_acc2 = float(np.mean(mlp2.predict(X2_tr) == y2_tr))

# Figura 2 — perda de treino × época
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(np.arange(1, EPOCHS2 + 1), hist2, color=COLORS[0],
        label=f"MLP {ARCH2}, η={LR2}")
ax.set_title("Figura 2 — Perda de treino (BCE) por época — Exercise 2")
ax.set_xlabel("Época")
ax.set_ylabel("Binary cross-entropy (treino)")
ax.legend()
ax.grid(alpha=0.3)
savefig(fig, "fig2_ex2_loss.png")

# Figura 3 — fronteira de decisão sobre os pontos de teste
lo, hi = X2.min(axis=0) - 1.0, X2.max(axis=0) + 1.0
gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 400), np.linspace(lo[1], hi[1], 400))
grid = np.c_[gx.ravel(), gy.ravel()]
Zm = mlp2.predict(grid).reshape(gx.shape)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.contourf(gx, gy, Zm, levels=[-0.5, 0.5, 1.5],
            cmap=ListedColormap(["#c6dbef", "#fdd0a2"]), alpha=0.8)
ax.contour(gx, gy, Zm, levels=[0.5], colors="k", linewidths=1)
for k in (0, 1):
    m = y2_te == k
    ax.scatter(X2_te[m, 0], X2_te[m, 1], s=16, c=COLORS[k], edgecolors="k",
               linewidths=0.3, label=f"Classe {k} (teste)")
ax.set_title(f"Figura 3 — Fronteira de decisão do MLP {ARCH2} (teste: {acc2:.1%})")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend(title="Classe", loc="upper left")
ax.set_aspect("equal")
savefig(fig, "fig3_ex2_boundary.png")


# =============================================================================
# Exercise 3 — multiclasse, MESMA classe MLP (só muda loss e tamanhos)
# =============================================================================
# --8<-- [start:ex3_data]
X3, y3 = make_classification(n_samples=1500, n_features=4, n_informative=4,
                             n_redundant=0, n_repeated=0, n_classes=3,
                             n_clusters_per_class=2, class_sep=1.0,
                             random_state=42)
X3_tr, X3_te, y3_tr, y3_te = train_test_split(
    X3, y3, test_size=0.2, random_state=42, stratify=y3)
Y3_tr = one_hot(y3_tr, 3)
# --8<-- [end:ex3_data]

# --8<-- [start:ex3_train]
ARCH3, LR3, EPOCHS3 = [4, 16, 3], 0.05, 300
mlp3 = MLP(ARCH3, loss="cce", lr=LR3, rng=rng)
hist3 = mlp3.fit(X3_tr, Y3_tr, epochs=EPOCHS3)
acc3 = float(np.mean(mlp3.predict(X3_te) == y3_te))
cm3 = confusion_matrix(y3_te, mlp3.predict(X3_te))
# --8<-- [end:ex3_train]
train_acc3 = float(np.mean(mlp3.predict(X3_tr) == y3_tr))

baseline3 = LogisticRegression(max_iter=1000).fit(X3_tr, y3_tr)
base3_acc = baseline3.score(X3_te, y3_te)
cm3_base = confusion_matrix(y3_te, baseline3.predict(X3_te))

# Par mais confundido: soma dos dois sentidos fora da diagonal
pairs = {(i, j): int(cm3[i, j] + cm3[j, i]) for i in range(3) for j in range(i + 1, 3)}
worst_pair = max(pairs, key=pairs.get)

# --8<-- [start:ex3_clusters]
# Análise do item D: de qual dos 6 clusters veio cada amostra?
# make_classification gera os clusters em blocos de 250 (cluster k -> classe k % 3)
# e SÓ DEPOIS embaralha linhas e colunas. Regerando com shuffle=False (mesmo
# random_state) obtemos os mesmos pontos em ordem de cluster; basta achar a
# permutação de colunas e casar as linhas.
X3_ord, _ = make_classification(n_samples=1500, n_features=4, n_informative=4,
                                n_redundant=0, n_repeated=0, n_classes=3,
                                n_clusters_per_class=2, class_sep=1.0,
                                random_state=42, shuffle=False)
from itertools import permutations
perm = next(list(p) for p in permutations(range(4))
            if np.allclose(np.sort(X3[:, list(p)], axis=0), np.sort(X3_ord, axis=0)))
row = np.array([np.argmin(np.abs(X3_ord - r).sum(axis=1)) for r in X3[:, perm]])
cluster3 = row // 250                     # cluster gerador de cada amostra (0..5)
n_flipped = int(np.sum(y3 != cluster3 % 3))  # flip_y=0.01: rótulos trocados ao acaso

centroids = np.array([X3_ord[k * 250:(k + 1) * 250].mean(axis=0) for k in range(6)])
cdist = np.linalg.norm(centroids[:, None] - centroids[None], axis=2)

# Mesmo split do enunciado, agora sobre os índices, para saber o cluster do teste
_, idx_te = train_test_split(np.arange(len(y3)), test_size=0.2,
                             random_state=42, stratify=y3)
pred3_te = mlp3.predict(X3_te)
err = pred3_te != y3_te
errors_by_cluster = {int(k): int(np.sum(err & (cluster3[idx_te] == k))) for k in range(6)}
# --8<-- [end:ex3_clusters]
assert np.allclose(X3[idx_te], X3_te)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(np.arange(1, EPOCHS3 + 1), hist3, color=COLORS[0], label=f"MLP {ARCH3}, η={LR3}")
ax.set_title("Figura 4 — Perda de treino (CCE) por época — Exercise 3")
ax.set_xlabel("Época")
ax.set_ylabel("Categorical cross-entropy (treino)")
ax.legend()
ax.grid(alpha=0.3)
savefig(fig, "fig4_ex3_loss.png")

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm3, cmap="Blues")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm3[i, j], ha="center", va="center",
                color="white" if cm3[i, j] > cm3.max() / 2 else "black", fontsize=13)
ax.set_xticks(range(3), [f"Classe {k}" for k in range(3)])
ax.set_yticks(range(3), [f"Classe {k}" for k in range(3)])
ax.set_xlabel("Classe prevista")
ax.set_ylabel("Classe verdadeira")
ax.set_title(f"Figura 5 — Matriz de confusão (teste) — Ex. 3, acc {acc3:.1%}")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Nº de amostras")
savefig(fig, "fig5_ex3_confusion.png")


# =============================================================================
# Exercise 4 — mais profundo: mesmos dados, épocas, η e batch do Ex. 3
# =============================================================================
# --8<-- [start:ex4_train]
ARCH4 = [4, 16, 16, 3]          # 2 camadas ocultas; resto idêntico ao Ex. 3
mlp4 = MLP(ARCH4, loss="cce", lr=LR3, rng=rng)
hist4 = mlp4.fit(X3_tr, Y3_tr, epochs=EPOCHS3)
acc4 = float(np.mean(mlp4.predict(X3_te) == y3_te))
# --8<-- [end:ex4_train]
train_acc4 = float(np.mean(mlp4.predict(X3_tr) == y3_tr))

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(np.arange(1, EPOCHS3 + 1), hist3, color=COLORS[0],
        label=f"Ex. 3 — {ARCH3} (1 oculta)")
ax.plot(np.arange(1, EPOCHS3 + 1), hist4, color=COLORS[1],
        label=f"Ex. 4 — {ARCH4} (2 ocultas)")
ax.set_title("Figura 6 — Perda de treino: 1 vs 2 camadas ocultas (rng do relatório)")
ax.set_xlabel("Época")
ax.set_ylabel("Categorical cross-entropy (treino)")
ax.legend(title="Rede")
ax.grid(alpha=0.3)
savefig(fig, "fig6_ex3_vs_ex4_loss.png")

# --8<-- [start:ex4_seeds]
# Comparação controlada em várias seeds: para cada seed, AS DUAS redes
# recebem um gerador novo com a mesma seed (mesma sorte de partida/embaralhamento).
SEEDS = [0, 1, 2, 3, 4]
rows = {"ex3": [], "ex4": []}
for s in SEEDS:
    for key, arch in (("ex3", ARCH3), ("ex4", ARCH4)):
        net = MLP(arch, loss="cce", lr=LR3, rng=np.random.default_rng(s))
        h = net.fit(X3_tr, Y3_tr, epochs=EPOCHS3)
        rows[key].append(dict(seed=s, final_loss=float(h[-1]),
                              test_acc=float(np.mean(net.predict(X3_te) == y3_te)),
                              train_acc=float(np.mean(net.predict(X3_tr) == y3_tr))))
# --8<-- [end:ex4_seeds]


def summary(key):
    r = rows[key]
    return dict(loss_mean=float(np.mean([x["final_loss"] for x in r])),
                loss_std=float(np.std([x["final_loss"] for x in r])),
                acc_mean=float(np.mean([x["test_acc"] for x in r])),
                acc_std=float(np.std([x["test_acc"] for x in r])),
                train_acc_mean=float(np.mean([x["train_acc"] for x in r])))


# =============================================================================
# Saída: todos os números citados no relatório
# =============================================================================
ex1 = exercise1()
RESULTS["ex1"] = {k: np.round(np.asarray(v), 6).tolist() for k, v in ex1.items()}
RESULTS["ex2"] = dict(baseline_acc=base2_acc,
                      baseline_coef=baseline2.coef_.ravel().tolist(),
                      baseline_intercept=float(baseline2.intercept_[0]),
                      baseline_by_blob=base2_by_blob,
                      baseline_proba_range=[float(base2_proba.min()), float(base2_proba.max())],
                      arch=ARCH2, lr=LR2, epochs=EPOCHS2,
                      params=mlp2.n_params(), test_acc=acc2, train_acc=train_acc2,
                      final_loss=float(hist2[-1]))
RESULTS["ex3"] = dict(baseline_acc=base3_acc, arch=ARCH3, lr=LR3, epochs=EPOCHS3,
                      params=mlp3.n_params(), test_acc=acc3, train_acc=train_acc3,
                      final_loss=float(hist3[-1]), confusion=cm3.tolist(),
                      confusion_baseline=cm3_base.tolist(),
                      pair_errors={f"{i}-{j}": v for (i, j), v in pairs.items()},
                      worst_pair=list(worst_pair),
                      cluster_class=[k % 3 for k in range(6)],
                      centroids=np.round(centroids, 3).tolist(),
                      centroid_dist=np.round(cdist, 3).tolist(),
                      n_flipped_labels=n_flipped,
                      test_errors_by_cluster=errors_by_cluster,
                      errors_detail=[dict(cluster=int(cluster3[i]), true=int(t), pred=int(p))
                                     for i, t, p in zip(idx_te[err], y3_te[err], pred3_te[err])])
RESULTS["ex4"] = dict(arch=ARCH4, params=mlp4.n_params(), test_acc=acc4,
                      train_acc=train_acc4, final_loss=float(hist4[-1]))
RESULTS["seeds"] = dict(seeds=SEEDS, runs=rows, ex3=summary("ex3"), ex4=summary("ex4"),
                        acc_diff_mean=summary("ex4")["acc_mean"] - summary("ex3")["acc_mean"])

with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(RESULTS, f, indent=2)
print(json.dumps(RESULTS, indent=2))
