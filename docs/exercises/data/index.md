---
exercise: data
ai_use: "Claude assisted in writing the data-generation, analysis and preprocessing code and in drafting this report; TODO - edit this line to accurately describe your own use of AI tools before submitting."
---

# 1. Data

## Exercise 1

### Point Clouds: Geometry and Spread in 2D

For this one I generated four 2D Gaussian clouds with `rng = np.random.default_rng(42)` and kept that same rng for everything downstream. Then I regenerated the same four classes at four different spread levels and looked at how quickly they start running into each other, using a separation ratio and a simple nearest-centroid mixing rate. No model gets trained anywhere in this exercise — it's all just geometry.

### A — Generate the clouds

400 points total, 100 per class, straight from the means and standard deviations given in the prompt.

![Figure 1](figures/fig1_point_clouds.png)

*Figure 1 — the four clouds at s = 1.0. Centers marked with an X, and I also added the nearest-centroid regions (dashed lines) since that's basically the boundary a small trained network would end up learning — I reuse this in part C.*

```python
--8<-- "docs/exercises/data/code/ex1_point_clouds.py"
```

### B — More or less spread out

Same 4 classes, regenerated four times with the std devs multiplied by `s ∈ {0.5, 1.0, 2.0, 4.0}` (means stay put).

![Figure 2](figures/fig2_spread_subplots.png)

*Figure 2 — all four versions on shared axes so the comparison is fair.*

**Separation ratio r_ij at s = 1.0**, all 6 pairs:

| Pair (i, j) | r_ij |
|---|---|
| 0–1 | 1.326 |
| 0–2 | 2.480 |
| 0–3 | 4.496 |
| 1–2 | 2.380 |
| 1–3 | 3.642 |
| 2–3 | 3.542 |

Smallest one is pair (0, 1) at 1.326. Since the means never move, r_ij just scales as 1/s, so at s = 2 that same pair drops to 1.326 / 2 = **0.663** — no need to regenerate anything, just divide.

**Mixing rate per scale** (fraction of points whose nearest mean isn't their own class):

| s | mixing rate |
|---|---|
| 0.5 | 0.0000 |
| 1.0 | 0.0675 |
| 2.0 | 0.2250 |
| 4.0 | 0.4175 |

![Figure 3](figures/fig3_mixing_rate.png)

*Figure 3 — mixing rate climbing steadily with s.*

So: where does linear separability actually break down? Somewhere between s = 1 and s = 2. At s = 1 the worst-case r_ij is still 1.326, i.e. the two closest centers are further apart than their combined spread. By s = 2 that same ratio has fallen to 0.663 — under 1, meaning the spread now beats the distance between centers. And that's exactly where the mixing rate takes its biggest jump too, from 6.75% up to 22.5%. The two measures agree.

### C — Analysis

1. At s = 1, classes 0 and 1 sit right next to each other and blend a bit (makes sense, they have the smallest r_ij). Class 2 is below them, reasonably separated, and class 3 is way off to the right on its own — nothing to worry about there. Could one straight line separate all four? No — it's a 4-class problem and a single hyperplane only gives you two sides. A handful of linear boundaries stitched together does a decent job though, as you can see in Figure 1: three of the four regions come out clean, and the only messy one is the 0/1 boundary, right where the actual overlap is.
2. The dashed lines in Figure 1 are that sketch — nearest-centroid regions computed over the four means. It's a reasonable stand-in for what a small network would learn here, since the clusters are all roughly the same rough shape.
3. Tying it back to part B: as s grows the clouds swell and start eating into each other's territory, and the strip around each boundary line gets fatter. That's literally what the mixing rate is counting. At s = 0.5 there's barely any overlap zone (mixing rate ~0), but by s = 4 almost half the points (41.75%) land on the wrong side of their own centroid, because at that point the boundary just can't keep up with how spread out everything is.

---

## Exercise 2

### Non-Linearity in Higher Dimensions

Two 5D datasets here, 500 points per class each. Dataset I is two shifted multivariate Gaussians with different covariance structure. Dataset II is two concentric "shells" — pick a random direction on the unit sphere, then a random radius. I looked at both through a PCA projection and also directly in 5D.

### A — Dataset I: shifted Gaussians

Straightforward — `rng.multivariate_normal` with the given `μ_A, Σ_A, μ_B, Σ_B`.

### B — Dataset II: concentric shells

Direction vectors come from `𝒩(0, I₅)`, normalized to unit length. Class C (the core) gets radius `𝒩(2.0, 0.4)`, Class D (the shell around it) gets `𝒩(5.0, 0.4)`.

```python
--8<-- "docs/exercises/data/code/ex2_nonlinearity.py"
```

### C — Visualize and compare

![Figure 4](figures/fig4_pca_projection.png)

*Figure 4 — both datasets projected down to 2D with PCA.*

| | Explained variance (PC1 + PC2) |
|---|---|
| Dataset I | 0.660 |
| Dataset II | 0.429 |

Dataset I keeps a lot more information in the first two components (66% vs 43%) and the classes are still visibly apart after projecting — makes sense, PCA is chasing the direction of highest variance and that direction happens to line up with the shift between A and B. Dataset II doesn't have that luck: its two classes look like one big blob once flattened to 2D, because what actually separates them (radius from the origin) isn't the direction PCA cares about.

**In 5D, no projection involved:**

| | Distance between centers ‖μ₁ − μ₂‖ |
|---|---|
| Dataset I | 3.228 |
| Dataset II | 0.266 |

![Figure 5](figures/fig5_radius_histograms.png)

*Figure 5 — ‖x‖ histograms per class, both datasets.*

### D — Analysis

1. This is the interesting part of Dataset II: the centers are almost on top of each other (0.266 apart, basically zero by construction — both shells are centered at the origin), and yet Figure 5 shows the radius histograms don't overlap at all. That combination is a dead giveaway for radial structure. A hyperplane works by picking a direction and thresholding along it, and when both classes share a center, there's no direction where one class systematically sits further along than the other — the "far from origin vs close to origin" signal just isn't something a linear projection can pick up.
2. And that's exactly why no hyperplane can ever solve this, no matter how much data you throw at it. The two classes are literally nested shells around the same center. A hyperplane cuts space into two half-spaces, but each half-space still contains points at every radius — near and far — so it always slices straight through both shells. It's a geometry problem, not a data problem, so more samples won't fix it.
3. Does a mixed-looking PCA projection mean the classes are truly inseparable? No, and this dataset is a good counterexample. PCA only optimizes for retained variance, not for keeping classes apart, so it's entirely possible for it to throw away the one direction (or in this case, one nonlinear quantity) that actually does the separating. Here that quantity is `f(x) = ‖x‖² = Σᵢxᵢ²`. Computing it directly: the core class averages ‖x‖² ≈ 4.04 (max 10.54), the shell class averages ≈ 25.21 (min 14.08) — the two ranges don't even touch. A threshold around 12.3 separates all 1000 points with zero mistakes, even though the same points looked hopelessly mixed after PCA.

---

## Exercise 3

### Preparing Real-World Data for a Neural Network

Using the Spaceship Titanic `train.csv` (8,693 rows). I split first, before touching any statistics, then did imputation, encoding, a bit of feature engineering, log-transforming the skewed columns, and finally scaling — with every fitted step trained only on the training split — to get something a `tanh` hidden layer can actually work with.

### A — Get to know the data

`Transported` is the target: did the passenger get pulled into another dimension or not. It's close to a coin flip — **50.36% True / 49.64% False** — so no class imbalance to worry about here.

**Feature types**

- Numerical: `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`
- Categorical: `HomePlanet`, `CryoSleep`, `Destination`, `VIP` (`Cabin`, `Name`, and `PassengerId` get dropped or turned into something else — see part C)

**Missing values**

| Column | Missing count | Missing % |
|---|---|---|
| CryoSleep | 217 | 2.50% |
| ShoppingMall | 208 | 2.39% |
| VIP | 203 | 2.34% |
| HomePlanet | 201 | 2.31% |
| Name | 200 | 2.30% |
| Cabin | 199 | 2.29% |
| VRDeck | 188 | 2.16% |
| FoodCourt | 183 | 2.11% |
| Spa | 183 | 2.11% |
| Destination | 182 | 2.09% |
| RoomService | 181 | 2.08% |
| Age | 179 | 2.06% |

Nothing stands out — every column sits around 2 to 2.5% missing. Looks like random dropout across the board rather than one field being systematically broken.

**Spending columns**

| Column | Mean | Median | Max |
|---|---|---|---|
| RoomService | 224.69 | 0.0 | 14,327 |
| FoodCourt | 458.08 | 0.0 | 29,813 |
| ShoppingMall | 173.73 | 0.0 | 23,492 |
| Spa | 311.14 | 0.0 | 22,408 |
| VRDeck | 304.85 | 0.0 | 24,133 |

Every single one of these has a median of exactly 0 while the mean sits in the hundreds. Most passengers just don't spend anything, and a small chunk spend a ton (the maxes run into the tens of thousands). Mean way above median is the classic sign of a skewed, heavy-tailed distribution — that's the reason for the log transform coming up in part C.

### B — Split before you transform

80/20 split, stratified on `Transported`, `random_state=42`.

Why before imputation and scaling? Because both of those steps compute something from the data — the median for filling gaps, the mean/std for scaling — and if that "something" is computed using rows the model will later be tested on, the model has effectively already peeked at the test set before it even started training. The split has to happen first so the test set stays genuinely unseen.

### C — Preprocess

```python
--8<-- "docs/exercises/data/code/ex3_preprocessing.py"
```

1. **Missing data.** Median imputation for the numerical columns — robust against those long-tailed spending values and any odd ages — and most-frequent-category imputation for the categorical ones. Both imputers are fit on the training split only, then applied to test.
2. **Categorical encoding.** One-hot encoding for `HomePlanet`, `CryoSleep`, `Destination`, `VIP`, using `OneHotEncoder(handle_unknown="ignore")` fit on the training categories. If the test set has a category the encoder never saw during training, it just gets an all-zero row for that feature instead of throwing an error — the model gets no signal from it rather than the whole thing crashing.
3. **Feature engineering.** `TotalSpend` is just the row-wise sum across the five spending columns (before the log transform, `skipna=True` so missing values don't wipe out the whole sum). `Cabin`, `Name`, and `PassengerId` get dropped — they're identifiers or free text, not really usable as-is.
4. **Heavy tails.** `log1p` on the five spending columns plus the new `TotalSpend`. Figure 6 shows what this does to `FoodCourt`.
5. **Scaling.** Standardized the whole numerical block (the 6 original numeric columns plus `TotalSpend`) to mean 0, std 1, fit on train only. I went with standardization instead of squeezing everything into `[-1, 1]`, since even after the log transform the spending columns still have a bit of a tail, and a hard min/max rescale would crush the genuinely large (rare) values right up against the boundary.

### D — Verify and visualize

![Figure 6](figures/fig6_foodcourt_before_after.png)

*Figure 6 — `FoodCourt` on the training set, before and after `log1p`. Before, it's basically a spike at 0 with a long thin tail stretching out past 25,000. After, the same data is spread out much more evenly across a usable range — something a `tanh` unit can actually work with instead of just ignoring almost everything or saturating on the rare huge values.*

**Final checks**

- NaNs remaining: **0**, both train and test.
- Final training feature matrix shape: **(6954, 17)** — that's 7 standardized numeric columns (the 6 original + `TotalSpend`) plus 10 one-hot columns (3 for `HomePlanet`, 2 for `CryoSleep`, 3 for `Destination`, 2 for `VIP`). Test matrix comes out to (1739, 17).
- Value range: the standardized numeric columns run from **-2.00 to 3.51** on train (**-2.00 to 3.37** on test); one-hot columns are just 0/1. Not hard-clamped to [-1, 1], but centered around 0 with most values within a couple of units of it — should sit comfortably in the non-saturated part of `tanh`.

**Reflection.** If I had to pick the one preprocessing choice that matters most for training, it's the `log1p` on the spending columns. Without it, those raw values (mostly 0, occasionally in the tens of thousands) would completely dominate the scale after standardization — you'd end up with a huge pile of near-identical values at one end and a handful of extreme outliers way out in `tanh`'s saturated region. That's a much bigger deal for how well gradient descent behaves than, say, whether I'd used median vs. mean imputation.

---

## Results summary

| # | Item | Your value |
|---|---|---|
| 1 | Mixing rate at s = 0.5 | 0.0000 |
| 2 | Mixing rate at s = 1.0 | 0.0675 |
| 3 | Mixing rate at s = 2.0 | 0.2250 |
| 4 | Mixing rate at s = 4.0 | 0.4175 |
| 5 | Smallest r_ij at s = 1.0, and which pair | 1.326, pair (0, 1) |
| 6 | Distance between centers — Dataset I | 3.228 |
| 7 | Distance between centers — Dataset II | 0.266 |
| 8 | Explained variance PC1 + PC2 — Dataset I | 0.660 |
| 9 | Explained variance PC1 + PC2 — Dataset II | 0.429 |
| 10 | Share of the positive class in Transported | 50.36% (True) |
| 11 | Mean and median of FoodCourt on the training set, before transforming | mean 452.61, median 0.00 |
| 12 | Final shape of the training feature matrix | (6954, 17) |
| 13 | Minimum and maximum of the training and test sets after scaling | train [-2.00, 3.51], test [-2.00, 3.37] |
