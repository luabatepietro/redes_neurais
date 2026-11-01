"""Exercise 4 — VAE no MNIST.

Roda os dois exercícios em sequência e gera as Figuras 1–6 em ../figures/ e todos os
números citados no relatório em results.json.

    cd docs/exercises/vae/code && python vae_report.py
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)
os.chdir(HERE)  # o "data" do enunciado fica relativo a code/

BATCH, EPOCHS, LR = 128, 20, 1e-3
results = {}

# --8<-- [start:data]
# Exercise 1A — dados. ToTensor() já deixa os pixels em [0, 1]; o achatamento para 784
# acontece em flat(), na hora de entrar no modelo.
train_full = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
test = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
train, val = random_split(train_full, [50_000, 10_000],
                          generator=torch.Generator().manual_seed(42))


def flat(x):
    return x.view(x.size(0), -1)  # (B, 1, 28, 28) -> (B, 784)
# --8<-- [end:data]


# --8<-- [start:model]
class VAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        # encoder 784 -> 512 -> 256, ReLU depois de cada camada
        self.enc = nn.Sequential(nn.Linear(784, 512), nn.ReLU(),
                                 nn.Linear(512, 256), nn.ReLU())
        # duas cabeças lineares: uma para mu, outra para log(sigma^2)
        self.mu = nn.Linear(256, d)
        self.logvar = nn.Linear(256, d)
        # decoder d -> 256 -> 512 -> 784, sigmoid na saída (pixel = probabilidade de Bernoulli)
        self.dec = nn.Sequential(nn.Linear(d, 256), nn.ReLU(),
                                 nn.Linear(256, 512), nn.ReLU(),
                                 nn.Linear(512, 784), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        # z = mu + sigma * eps, com eps ~ N(0, I). A aleatoriedade fica toda em eps,
        # então o gradiente passa por mu e sigma normalmente.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(x_hat, x, mu, logvar):
    """Devolve (reconstrução, KL), ambos em nats por imagem (média no batch)."""
    B = x.size(0)
    # BCE somada nos 784 pixels, dividida pelo tamanho do batch (nada de reduction="mean")
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / B
    # KL( N(mu, sigma^2) || N(0, I) ) em forma fechada, somada nas d dimensões latentes:
    # 0.5 * sum( mu^2 + sigma^2 - 1 - log sigma^2 )
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar) / B
    return recon, kl
# --8<-- [end:model]


# --8<-- [start:train]
@torch.no_grad()
def evaluate(model, dataset):
    """Média por imagem de reconstrução e KL num conjunto inteiro."""
    model.eval()
    tot_r = tot_k = 0.0
    for x, _ in DataLoader(dataset, batch_size=1000):
        x = flat(x)
        x_hat, mu, logvar = model(x)
        r, k = vae_loss(x_hat, x, mu, logvar)
        tot_r += r.item() * x.size(0)
        tot_k += k.item() * x.size(0)
    return tot_r / len(dataset), tot_k / len(dataset)


def train_vae(d):
    torch.manual_seed(42)  # antes de construir cada modelo
    model = VAE(d)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(train, batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    hist = {k: [] for k in ("train_recon", "train_kl", "val_recon", "val_kl")}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_r = tot_k = 0.0
        for x, _ in loader:
            x = flat(x)
            x_hat, mu, logvar = model(x)
            r, k = vae_loss(x_hat, x, mu, logvar)
            loss = r + k  # beta = 1 (ELBO negativo)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_r += r.item() * x.size(0)
            tot_k += k.item() * x.size(0)
        hist["train_recon"].append(tot_r / len(train))
        hist["train_kl"].append(tot_k / len(train))
        vr, vk = evaluate(model, val)  # decisões de treino só olham a validação
        hist["val_recon"].append(vr)
        hist["val_kl"].append(vk)
        print(f"d={d:2d} ep {epoch:2d} | treino rec {hist['train_recon'][-1]:6.2f} "
              f"kl {hist['train_kl'][-1]:5.2f} | val rec {vr:6.2f} kl {vk:5.2f}")
    return model, hist
# --8<-- [end:train]


def tile(imgs, nrows, ncols):
    """Junta nrows*ncols imagens 28x28 numa única matriz."""
    imgs = np.asarray(imgs).reshape(nrows, ncols, 28, 28)
    return imgs.transpose(0, 2, 1, 3).reshape(nrows * 28, ncols * 28)


def show_grid(canvas, path, title, xlabel, ylabel, xticks=None, yticks=None, size=(8, 2.4)):
    fig, ax = plt.subplots(figsize=size)
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([] if xticks is None else xticks[0], [] if xticks is None else xticks[1])
    ax.set_yticks([] if yticks is None else yticks[0], [] if yticks is None else yticks[1])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def centers(n):
    return np.arange(n) * 28 + 13.5


# ============================== Exercise 1 ==============================
model16, hist16 = train_vae(16)

# --8<-- [start:ex1_test]
# Exercise 1C — conjunto de teste usado só aqui, depois do treino terminado
rec16, kl16 = evaluate(model16, test)
print(f"[teste d=16] rec {rec16:.2f} | KL {kl16:.2f} | -ELBO {rec16 + kl16:.2f}")
# --8<-- [end:ex1_test]
results["d16"] = {"test_recon": rec16, "test_kl": kl16, "test_neg_elbo": rec16 + kl16,
                  "val_kl_first": hist16["val_kl"][0], "val_kl_last": hist16["val_kl"][-1],
                  "history": hist16}

# Figura 1 — curvas de treino e validação, um painel por termo
ep = np.arange(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, term, name in [(axes[0], "recon", "Reconstrução (BCE)"), (axes[1], "kl", "KL")]:
    ax.plot(ep, hist16[f"train_{term}"], "o-", ms=3, label="treino")
    ax.plot(ep, hist16[f"val_{term}"], "s--", ms=3, label="validação")
    ax.set_title(f"{name} por época (d = 16)")
    ax.set_xlabel("época")
    ax.set_ylabel("nats por imagem")
    ax.set_xticks(ep[::2] if EPOCHS > 10 else ep)
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
fig.savefig(FIG / "fig1_loss_terms.png", dpi=150)
plt.close(fig)

# --8<-- [start:ex1_figs]
# Figura 2 — primeira imagem de teste de cada dígito e a reconstrução decodificando mu
model16.eval()
labels = test.targets
first_idx = [int((labels == c).nonzero()[0]) for c in range(10)]
x_first = flat(torch.stack([test[i][0] for i in first_idx]))
with torch.no_grad():
    mu_first, _ = model16.encode(x_first)
    rec_first = model16.decode(mu_first)  # mu, não um z amostrado

# Figura 3 — 64 amostras do prior N(0, I)
torch.manual_seed(0)
with torch.no_grad():
    samples = model16.decode(torch.randn(64, 16))
# --8<-- [end:ex1_figs]
results["fig2_indices"] = first_idx

show_grid(tile(torch.cat([x_first, rec_first]).numpy(), 2, 10), FIG / "fig2_reconstructions.png",
          "Originais do teste e reconstruções a partir de μ (d = 16)", "dígito", "",
          xticks=(centers(10), range(10)), yticks=(centers(2), ["original", "reconstrução"]))
show_grid(tile(samples.numpy(), 8, 8), FIG / "fig3_prior_samples.png",
          "64 amostras z ~ N(0, I) decodificadas (d = 16)", "amostra (coluna)",
          "amostra (linha)", size=(5.2, 5.6))

# ============================== Exercise 2 ==============================
model2, hist2 = train_vae(2)
rec2, kl2 = evaluate(model2, test)
print(f"[teste d=2] rec {rec2:.2f} | KL {kl2:.2f} | -ELBO {rec2 + kl2:.2f}")
results["d2"] = {"test_recon": rec2, "test_kl": kl2, "test_neg_elbo": rec2 + kl2,
                 "history": hist2}

# Figura 4 — mu das 10.000 imagens de teste
model2.eval()
x_test = flat(test.data.float() / 255.0)  # mesmo que ToTensor(), só que de uma vez
with torch.no_grad():
    mu_test, _ = model2.encode(x_test)
mu_test = mu_test.numpy()
fig, ax = plt.subplots(figsize=(7, 6))
cmap = plt.get_cmap("tab10")
for c in range(10):
    m = labels.numpy() == c
    ax.scatter(mu_test[m, 0], mu_test[m, 1], s=3, alpha=0.5, color=cmap(c), label=str(c))
ax.set_title("Média μ do encoder para as 10.000 imagens de teste (d = 2)")
ax.set_xlabel("$\\mu_1$")
ax.set_ylabel("$\\mu_2$")
ax.legend(title="dígito", markerscale=4, loc="center left", bbox_to_anchor=(1.0, 0.5))
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig4_latent_scatter.png", dpi=150)
plt.close(fig)

# --8<-- [start:ex2_grid]
# Figura 5 — plano latente decodificado numa grade de quantis da normal padrão
q = np.linspace(0.05, 0.95, 15)
zs = norm.ppf(q)  # Phi^{-1}(q)
# linha 0 do canvas fica em cima, então z2 vai do maior (em cima) para o menor (embaixo),
# para a figura ficar com a mesma orientação do scatter da Figura 4
grid = torch.tensor([[z1, z2] for z2 in zs[::-1] for z1 in zs], dtype=torch.float32)
with torch.no_grad():
    plane = model2.decode(grid)
# --8<-- [end:ex2_grid]
lab = [f"{z:.2f}" for z in zs]
show_grid(tile(plane.numpy(), 15, 15), FIG / "fig5_latent_plane.png",
          "Plano latente decodificado em z = Φ⁻¹(q), q ∈ [0.05, 0.95] (d = 2)",
          "$z_1$", "$z_2$", xticks=(centers(15), lab), yticks=(centers(15), lab[::-1]),
          size=(9, 9.4))
plt.close("all")

# --8<-- [start:ex2_interp]
# Figura 6 — interpolação entre duas imagens de teste de dígitos diferentes
ia, ib = 0, 1  # test[0] é um 7, test[1] é um 2
xa, xb = flat(test[ia][0].unsqueeze(0)), flat(test[ib][0].unsqueeze(0))
t = torch.linspace(0, 1, 10).unsqueeze(1)
pixel_row = (1 - t) * xa + t * xb  # interpolação direto nos pixels
with torch.no_grad():
    mu_a, _ = model16.encode(xa)  # modelo d = 16 do Exercise 1
    mu_b, _ = model16.encode(xb)
    latent_row = model16.decode((1 - t) * mu_a + t * mu_b)
# --8<-- [end:ex2_interp]
results["interp"] = {"index_a": ia, "label_a": int(labels[ia]),
                     "index_b": ib, "label_b": int(labels[ib])}
show_grid(tile(torch.cat([pixel_row, latent_row]).numpy(), 2, 10), FIG / "fig6_interpolation.png",
          f"Interpolação teste[{ia}] ({int(labels[ia])}) → teste[{ib}] ({int(labels[ib])})",
          "t", "", xticks=(centers(10), [f"{v:.2f}" for v in t.squeeze().tolist()]),
          yticks=(centers(2), ["pixels", "latente\n(d = 16)"]))

(HERE / "results.json").write_text(json.dumps(results, indent=2))
print("ok — figuras em", FIG)
