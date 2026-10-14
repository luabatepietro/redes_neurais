"""
2. Perceptron — Exercises 1 and 2.

Um único script, com UM único gerador aleatório (rng = default_rng(42)) usado em
todo o relatório, na ordem do enunciado. Rodar de dentro de code/:

    python perceptron_report.py

Gera as Figuras 1-6 em ../figures/ e imprime todos os números citados no relatório.
Bibliotecas: apenas numpy e matplotlib. O perceptron (ativação, predição, regra de
atualização e loop de treino) é implementado à mão — nada de scikit-learn.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Seed fixa: o MESMO rng é usado do começo ao fim do relatório.
rng = np.random.default_rng(42)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
COLORS = {0: "#1f77b4", 1: "#ff7f0e"}


# --8<-- [start:perceptron]
def step(z):
    """Função degrau: 1 se z >= 0, senão 0 (funciona para escalar ou vetor)."""
    return (np.asarray(z) >= 0).astype(int)


class Perceptron:
    """Perceptron de camada única, escrito do zero, com rótulos y ∈ {0, 1}."""

    def __init__(self, w_init, b_init=0.0, lr=0.01):
        # Pesos iniciais recebidos de fora (sorteados com o rng do relatório),
        # para que duas execuções possam partir exatamente do mesmo ponto.
        self.w = np.array(w_init, dtype=float)
        self.b = float(b_init)
        self.lr = lr

    def predict(self, X):
        """ŷ = step(w·x + b) para cada linha de X."""
        return step(X @ self.w + self.b)

    def accuracy(self, X, y):
        """Acurácia no dataset inteiro com os pesos ATUAIS."""
        return float(np.mean(self.predict(X) == y))

    def fit(self, X, y, max_epochs=100, pocket=False):
        """
        Treino online, amostra por amostra, na ordem de X.
        Para quando uma época inteira não produz nenhuma atualização,
        ou após max_epochs épocas.

        pocket=True liga o algoritmo pocket: após cada atualização, se a acurácia
        no dataset inteiro superar a melhor já vista, copia (w, b) para o "bolso".
        É a única coisa acrescentada ao loop — a regra de aprendizado é a mesma.
        """
        history = {"acc": [], "pocket_acc": [], "updates": []}
        best_acc = self.accuracy(X, y)
        self.pocket_w, self.pocket_b = self.w.copy(), self.b
        self.pocket_epoch = 0

        for epoch in range(1, max_epochs + 1):
            n_updates = 0
            for xi, yi in zip(X, y):
                y_hat = step(self.w @ xi + self.b)   # predição
                error = yi - y_hat                   # 0 (acerto), +1 ou -1 (erro)
                if error != 0:
                    # Regra de atualização para rótulos {0,1}:
                    #   w <- w + η (y - ŷ) x ,   b <- b + η (y - ŷ)
                    self.w += self.lr * error * xi
                    self.b += self.lr * error
                    n_updates += 1
                    if pocket:
                        acc = self.accuracy(X, y)
                        if acc > best_acc:           # melhor até agora -> bolso
                            best_acc = acc
                            self.pocket_w, self.pocket_b = self.w.copy(), self.b
                            self.pocket_epoch = epoch
            history["acc"].append(self.accuracy(X, y))   # acurácia ao fim da época
            history["pocket_acc"].append(best_acc)
            history["updates"].append(n_updates)
            if n_updates == 0:                           # passada limpa: convergiu
                break
        self.epochs_run = epoch
        return history
# --8<-- [end:perceptron]


# ---------------------------------------------------------------- helpers de plot
def scatter_classes(ax, X, y, alpha=0.45):
    for c in (0, 1):
        ax.scatter(*X[y == c].T, s=10, alpha=alpha, color=COLORS[c], label=f"Classe {c}")


def draw_boundary(ax, w, b, X, **kw):
    """Desenha a reta w·x + b = 0 dentro da janela dos dados."""
    xs = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
    ax.plot(xs, -(w[0] * xs + b) / w[1], **kw)


def mark_errors(ax, X, y, y_pred, **kw):
    wrong = y_pred != y
    ax.scatter(*X[wrong].T, s=40, facecolors="none", linewidths=1.2, **kw)
    return int(wrong.sum())


def finish(ax, title, X, loc="upper left"):
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax.legend(loc=loc, fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.3)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=130)
    plt.close(fig)


# --8<-- [start:data]
def make_dataset(mean0, mean1, cov, n=1000):
    """Duas gaussianas 2D, n pontos por classe, geradas com o rng do relatório.
    Sem embaralhar: a ordem é a de geração (1000 da Classe 0, depois 1000 da
    Classe 1), fixa em todas as épocas e em todas as execuções."""
    X0 = rng.multivariate_normal(mean0, cov, size=n)
    X1 = rng.multivariate_normal(mean1, cov, size=n)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return X, y
# --8<-- [end:data]


results = {}

# =============================================================== EXERCISE 1
# --8<-- [start:ex1]
# A — dados separáveis
X1, y1 = make_dataset([1.5, 1.5], [5, 5], [[0.5, 0], [0, 0.5]])

# B/C — inicialização não nula, sorteada UMA vez e reaproveitada no item D
w0_ex1 = rng.normal(0, 0.01, size=2)
p1 = Perceptron(w0_ex1, b_init=0.0, lr=0.01)
h1 = p1.fit(X1, y1, max_epochs=100)

# D — mesma execução, trocando SÓ η (mesmos dados, mesma ordem, mesmo w inicial)
p1_big = Perceptron(w0_ex1, b_init=0.0, lr=1.0)
h1_big = p1_big.fit(X1, y1, max_epochs=100)

# D — verificação numérica do argumento algébrico: partindo de w = 0, b = 0
pz_small = Perceptron([0.0, 0.0], 0.0, lr=0.01); hz_small = pz_small.fit(X1, y1)
pz_big = Perceptron([0.0, 0.0], 0.0, lr=1.0);    hz_big = pz_big.fit(X1, y1)
# --8<-- [end:ex1]

unit = lambda v: v / np.linalg.norm(v)
results["ex1"] = {
    "w0": w0_ex1.tolist(),
    "w": p1.w.tolist(), "b": p1.b, "epochs": p1.epochs_run,
    "acc": h1["acc"][-1], "updates_per_epoch": h1["updates"],
    "acc_per_epoch": h1["acc"],
    "eta1": {"w": p1_big.w.tolist(), "b": p1_big.b, "epochs": p1_big.epochs_run,
             "acc": h1_big["acc"][-1], "updates_per_epoch": h1_big["updates"]},
    "dir_eta001": unit(p1.w).tolist(), "dir_eta1": unit(p1_big.w).tolist(),
    "angle_deg": float(np.degrees(np.arccos(np.clip(unit(p1.w) @ unit(p1_big.w), -1, 1)))),
    "boundary_eta001": {"slope": float(-p1.w[0] / p1.w[1]), "intercept": float(-p1.b / p1.w[1])},
    "boundary_eta1": {"slope": float(-p1_big.w[0] / p1_big.w[1]), "intercept": float(-p1_big.b / p1_big.w[1])},
    "zero_start": {
        "eta001": {"w": pz_small.w.tolist(), "b": pz_small.b, "epochs": pz_small.epochs_run},
        "eta1": {"w": pz_big.w.tolist(), "b": pz_big.b, "epochs": pz_big.epochs_run},
        "ratio_w": (pz_big.w / pz_small.w).tolist(), "ratio_b": pz_big.b / pz_small.b,
    },
}

# Figura 1
fig, ax = plt.subplots(figsize=(6.5, 5.5))
scatter_classes(ax, X1, y1)
finish(ax, "Figura 1 — Exercise 1: dados separáveis (1000 pontos por classe)", X1)
savefig(fig, "fig1_separable_data.png")

# Figura 2
fig, ax = plt.subplots(figsize=(6.5, 5.5))
scatter_classes(ax, X1, y1)
draw_boundary(ax, p1.w, p1.b, X1, color="k", lw=2, label="Fronteira $w\\cdot x+b=0$ (η=0.01)")
n_err1 = int((p1.predict(X1) != y1).sum())
mark_errors(ax, X1, y1, p1.predict(X1), edgecolors="red", label=f"Classificado errado ({n_err1})")
finish(ax, "Figura 2 — Exercise 1: fronteira de decisão final", X1)
savefig(fig, "fig2_separable_boundary.png")

# Figura 3
fig, ax = plt.subplots(figsize=(6.5, 4))
ep = np.arange(1, len(h1["acc"]) + 1)
ax.plot(ep, h1["acc"], "o-", label="η = 0.01")
ax.plot(np.arange(1, len(h1_big["acc"]) + 1), h1_big["acc"], "s--", label="η = 1.0")
ax.set_title("Figura 3 — Exercise 1: acurácia × época")
ax.set_xlabel("Época"); ax.set_ylabel("Acurácia no dataset inteiro")
ax.set_xticks(np.arange(1, max(len(h1["acc"]), len(h1_big["acc"])) + 1))
ax.grid(alpha=0.3); ax.legend()
savefig(fig, "fig3_separable_accuracy.png")

# =============================================================== EXERCISE 2
# --8<-- [start:ex2]
# A — dados sobrepostos (médias próximas, variância 3x maior)
X2, y2 = make_dataset([3, 3], [4, 4], [[1.5, 0], [0, 1.5]])

# B — MESMA classe Perceptron, mesmo η = 0.01 e limite de 100 épocas; só liga o pocket
w0_ex2 = rng.normal(0, 0.01, size=2)
p2 = Perceptron(w0_ex2, b_init=0.0, lr=0.01)
h2 = p2.fit(X2, y2, max_epochs=100, pocket=True)

acc_final = p2.accuracy(X2, y2)
acc_pocket = float(np.mean(step(X2 @ p2.pocket_w + p2.pocket_b) == y2))
# --8<-- [end:ex2]

def best_line_acc(X, y):
    """Referência (não é modelo treinado): melhor limiar sobre a projeção x1+x2,
    direção que liga as duas médias — só para comparar com o ~73% do enunciado."""
    s = X.sum(axis=1)
    return max(float(np.mean((s >= t) == y)) for t in np.sort(s))


pred_final = p2.predict(X2)
pocket_pred = step(X2 @ p2.pocket_w + p2.pocket_b)
dist_center = lambda w, b: float((w @ np.array([3.5, 3.5]) + b) / np.linalg.norm(w))
results["ex2"] = {
    "w0": w0_ex2.tolist(),
    "final": {"w": p2.w.tolist(), "b": p2.b, "acc": acc_final,
              "frac_pred_1": float(pred_final.mean()),
              "signed_dist_from_center": dist_center(p2.w, p2.b),
              "intercept_on_diag": float(-p2.b / (p2.w[0] + p2.w[1]))},
    "pocket": {"w": p2.pocket_w.tolist(), "b": p2.pocket_b, "acc": acc_pocket,
               "epoch": p2.pocket_epoch, "frac_pred_1": float(pocket_pred.mean()),
               "signed_dist_from_center": dist_center(p2.pocket_w, p2.pocket_b),
               "intercept_on_diag": float(-p2.pocket_b / (p2.pocket_w[0] + p2.pocket_w[1]))},
    "epochs_run": p2.epochs_run,
    "acc_per_epoch": h2["acc"], "pocket_per_epoch": h2["pocket_acc"],
    "updates_per_epoch": h2["updates"],
    "acc_stats_last50": {"min": float(np.min(h2["acc"][50:])), "max": float(np.max(h2["acc"][50:])),
                         "mean": float(np.mean(h2["acc"][50:]))},
    "mean_norm_x": float(np.linalg.norm(X2, axis=1).mean()),
    "best_line_along_diag": best_line_acc(X2, y2),
}

# Figura 4
fig, ax = plt.subplots(figsize=(6.5, 5.5))
scatter_classes(ax, X2, y2)
finish(ax, "Figura 4 — Exercise 2: dados sobrepostos (1000 pontos por classe)", X2)
savefig(fig, "fig4_overlap_data.png")

# Figura 5 — dois painéis: erros de cada conjunto de pesos
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
for ax, (name, w, b, pred, acc) in zip(axes, [
        ("final", p2.w, p2.b, pred_final, acc_final),
        ("pocket", p2.pocket_w, p2.pocket_b, pocket_pred, acc_pocket)]):
    scatter_classes(ax, X2, y2, alpha=0.35)
    draw_boundary(ax, p2.w, p2.b, X2, color="crimson", lw=2,
                  label=f"Final (acc = {acc_final:.3f})")
    draw_boundary(ax, p2.pocket_w, p2.pocket_b, X2, color="green", lw=2, ls="--",
                  label=f"Pocket (acc = {acc_pocket:.3f})")
    n = mark_errors(ax, X2, y2, pred, edgecolors="k", label=f"Errado pelos pesos {name} ({int((pred != y2).sum())})")
    finish(ax, f"Erros marcados: pesos {name}", X2, loc="lower right")
fig.suptitle("Figura 5 — Exercise 2: fronteiras final e pocket")
savefig(fig, "fig5_overlap_boundaries.png")

# Figura 6
fig, ax = plt.subplots(figsize=(7, 4))
ep2 = np.arange(1, len(h2["acc"]) + 1)
ax.plot(ep2, h2["acc"], lw=1.2, label="Pesos atuais (fim de cada época)")
ax.plot(ep2, h2["pocket_acc"], lw=2, color="green", label="Melhor até agora (pocket)")
ax.axhline(0.5, color="gray", ls=":", lw=1, label="Chute (50%)")
ax.set_title("Figura 6 — Exercise 2: acurácia × época")
ax.set_xlabel("Época"); ax.set_ylabel("Acurácia no dataset inteiro")
ax.set_ylim(0.3, 0.8); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8)
savefig(fig, "fig6_overlap_accuracy.png")

print(json.dumps(results, indent=1, default=float))
