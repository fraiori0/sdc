# Sparsity-Inducing Contrastive Hashing — Implementation Handout

Target repo: `kamwoh/sdc` (BMVC 2023). Verified against the uploaded snapshot.

This document specifies **what to build and why**, keyed to real file paths.
Low-level coding decisions are left to the implementer. The parts that are
non-negotiable are marked **MUST** — they are places where a wrong choice
produces plausible-looking but meaningless numbers rather than a crash.

---

## 0. Background: the method

We train an encoder whose output `z ∈ [0,1]^d` under an InfoNCE objective with a
**logarithmic dot-product critic**

```
g(z_a, z_b) = log( <z_a, z_b> + eps )
```

### 0.1 The denominator collapses

`exp(g)` is *linear* in `z_b`, so the InfoNCE denominator collapses to a single
mean vector:

```
(1/K) Σ_j exp(g(z_i, z_j)) = (1/K) Σ_j ( <z_i, z_j> + eps ) = <z_i, z̄> + eps
```

with `z̄ = (1/K) Σ_j z_j`. The loss becomes

```
L_i = log( <z_i^1, z̄> + eps ) − log( <z_i^1, z_i^2> + eps )
```

**Cost: O(Kd) instead of O(K²d).** CIBHash's `NtXentLoss` builds a full `(2K, 2K)`
similarity matrix; we do not. Worth reporting as a scalability claim.

**MUST NOT detach `z̄`.** Sample `i` contributes to `z̄`, and that self-contribution
term accounts for ~49% of the effective gradient (measured). A stop-gradient there
silently halves the regularization strength. Expose `detach_mean` as an *ablation
flag only*, defaulting to `False`.

### 0.2 Why sparsity and binarization emerge

Two mechanisms, both verified numerically on free codes.

**Absorbing zeros.** The self-positive gradient is multiplicative in the code:

```
∂/∂z_iu [ −log(‖z_i‖² + eps) ] = −2 z_iu / (‖z_i‖² + eps)   → 0  as z_iu → 0
```

An inactive unit therefore receives a strictly non-negative gradient (the negative
term always pushes down) and **cannot be recruited**. Zero is absorbing; the box
constraint makes one absorbing too. The result is bistable winner-take-all
dynamics: codes binarize as a *fixed-point property of a smooth objective*, with
no straight-through estimator, no Gumbel, no temperature schedule, no
binarization loss.

This is the core differentiator. Every method in the deep-hashing literature
imposes binarization (probabilistic Bernoulli layers, sign+STE, explicit
quantization losses). Here it is emergent.

**Radial cancellation.** Contracting the loss with the code:

```
<z_i, ∇L_i> = <z_i,z̄>/t_i − <z_i,z_i^2>/s_i  →  1 − 1 = 0
```

Both terms saturate identically, so scale is a null direction handled by the box,
and `eps` acts purely on the *pattern*. (This is also why the objective does **not**
transplant into a reconstruction/ISTA setting — there the log competes with an
unbounded quadratic and loses. Do not attempt that here.)

### 0.3 The sparsity law — sets `eps`, don't tune it

For binary codes of support size `κ` with uniform usage, `z̄_u = κ/d`, so
`t = κ²/d + eps` and `s = κ + eps`. Minimising

```
L(κ) = log(κ²/d + eps) − log(κ + eps)
```

gives `κ² + 2κ·eps − eps·d = 0`, i.e.

```
κ = sqrt(eps² + eps·d) − eps  ≈  sqrt(eps·d)
```

Measured on free codes (d ∈ {128,256,512}, eps ∈ {1e-3 … 1}, Adam, box [0,1]):
fitted `κ ∝ eps^0.445 · d^0.451` against the predicted (0.5, 0.5), with binarity
**1.000** in every run. The prefactor is ≈1.8, attributable to non-uniform usage
(measured usage CV 0.13–0.38), which is exactly the assumption the derivation makes.

**Practical inversion — use this to set `eps`:**

```
eps ≈ κ_target² / (3.24 · d)          # 3.24 = 1.8²
```

e.g. `d=64, κ_target=8 → eps ≈ 0.31`. Verify the realised κ; the agreement is
itself Experiment 3.

> Note `eps` is **not** small for realistic targets. Do not assume `eps ≈ 1e-8`
> numerical-stability values; that regime gives κ ≈ 0.

---

## 1. Repo facts (verified)

| Fact | Consequence |
|---|---|
| Components instantiated via hydra `_target_`; all `__init__.py` under `models/`, `trainers/` are **empty** | Adding an arm = new files + new config yaml. **No registry edits.** |
| `configs/transforms/cibhash.yaml` uses `utils.transforms.NCropsTransform` with two Compose pipelines | Two-view dataloader **already exists**. Zero data-pipeline changes. |
| `configs/dataset/cifar10.yaml`: `ep: 1`, `R: 1000` → 100 query/class, 500 train/class | This is **CIFAR-10(I)**, CIBHash's protocol. Directly comparable. |
| `models/arch/cibhash.py` returns `(x, prob, z)`; in training `z = SignHashLayer(prob − 0.5)`; at eval `z` is the **raw logits** | `sign(logits) == sign(prob−0.5)`, so eval is consistent. Our arch must respect the same train/eval split. |
| `models/loss/cibhash.py` signature: `forward(prob_i, prob_j, z_i, z_j, f_i, f_j)`; contrastive on `z` (±1) via cosine, plus symmetric KL on `prob` | Arm C plugs straight in using `prob_i, prob_j`. |
| Losses expose a `self.losses` dict; trainer logs every key | Log `contrast`, `kappa`, `binarity` here for free. |
| `experiments/train_helper.py` (~L206) and `experiments/test_hashing.py` (~L74) loop over **every key containing `'codes'`** and evaluate each with a name postfix | **Return `codes` and `codes_cont` from `inference_one_batch` and both get evaluated automatically.** This is the hook for Experiment 1. No driver changes. |
| `utils/hashing.py: preprocess_on_codes(codes, threshold, sign = dist_metric=='hamming')` applies `torch.sign()` | **TRAP — see §2.1** |
| `zero_mean_eval` config key exists in `train_helper` | Must be **off** for our codes (mean-centering destroys the sparse structure). |
| Contrastive baselines present: **CIBHash** and **WCH** (`models/loss/wch.py`) | Both share `transforms: cibhash`. CIBHash is the direct comparison. |

### 1.1 Backbone caveat

`configs/model/cibhash.yaml` overrides `backbone: vgg16`; `wch.yaml` uses
`vit_base_hf` with `backbone_lr_scale: 0`. **All arms MUST inherit CIBHash's
backbone and optimizer settings unchanged**, or mAP numbers are not comparable.

---

## 2. Interventions

### 2.1 **MUST FIX FIRST** — the `[0,1]` vs `{−1,+1}` domain trap

`utils/hashing.py` assumes `{−1,+1}` codes throughout. For a `[0,1]` code,
`torch.sign()` yields `{0,1}`, and `hamming(a,b) = 0.5·(nbit − a·bᵀ)` then returns
numbers that look reasonable and mean nothing:

```
z            = [[0.9, 0.0, 0.0, 0.8],
                [0.0, 0.7, 0.0, 0.0]]
torch.sign(z)= [[1., 0., 0., 1.],
                [0., 1., 0., 0.]]        # 0, not −1
hamming      = [[1.0, 2.0],
                [2.0, 1.5]]              # meaningless; note the non-integer
true overlap = [[2., 0.],
                [0., 1.]]
```

This fails **silently**. Required changes in `utils/hashing.py`:

1. Config key `code_domain: signed | unit` (default `signed`, preserving current
   behaviour for all baselines).
2. When `unit`: binarize at **0.5**, not `sign`.
3. New distance `overlap`: `dist = −(a @ b.T)` on `{0,1}` codes (sum-AND
   similarity, negated to a distance). Register in `get_distance_func`.
4. **MUST** assert in `preprocess_on_codes` that unit-domain codes lie in `[0,1]`
   and signed-domain codes are not all-non-negative. Cheap insurance against the
   exact failure above.

Report **both** `overlap` and `hamming` for our arms. Hamming on sparse codes is
dominated by shared zeros; presenting only one invites the reviewer question
"did you pick the flattering metric?".

### 2.2 New loss — `models/loss/sbdr.py`

Subclass `models.loss.base.BaseLoss`, populate `self.losses`.

```python
class SBDRCriticLoss(BaseLoss):
    def __init__(self, eps=0.31, symmetric=True, detach_mean=False, **kwargs):
        ...

    def _one_way(self, za, zb, zall):
        zbar = zall.mean(0)
        if self.detach_mean:                    # ABLATION ONLY
            zbar = zbar.detach()
        t = (za * zbar).sum(1) + self.eps
        s = (za * zb).sum(1) + self.eps
        return (t.log() - s.log()).mean()

    def forward(self, z1, z2):
        assert z1.min() >= -1e-6 and z1.max() <= 1 + 1e-6, "codes must be in [0,1]"
        zall = torch.cat([z1, z2], 0)
        L = self._one_way(z1, z2, zall)
        if self.symmetric:
            L = 0.5 * (L + self._one_way(z2, z1, zall))
        self.losses['contrast'] = L
        self.losses['kappa'] = (z1 > 0.5).float().sum(1).mean()
        return L
```

Match the CIBHash loss signature where an arm reuses the CIBHash trainer.

### 2.3 New arch — `models/arch/sbdr.py` (Arm B only)

Copy `models/arch/cibhash.py`; replace `SignHashLayer` with a bounded activation:

```python
z = x.clamp(0, 1) if self.act == "clip" else torch.sigmoid(x)
```

Config flag `model.act: clip | sigmoid`. **Prefer `clip`**: the gradient stays
linear on the interior instead of vanishing at saturation, which matters given
that the dynamics drive codes to the box boundary.

Return `(x, z, z)` so the trainer sees both a continuous and a code slot;
mirror CIBHash's train/eval branch so eval returns the pre-binarization value.

### 2.4 Trainers

- `trainers/sbdr.py` (Arm B) — mirrors `trainers/cibhash.py`.
- Arms C and D **import `models.arch.cibhash.CIBHash` untouched** and only change
  what the loss consumes. If a trainer subclass suffices, do that rather than
  duplicating.
- `inference_one_batch` **MUST** return **both**:
  - `codes` — binarized at 0.5 (unit domain)
  - `codes_cont` — the raw `z ∈ [0,1]`

  Both are then evaluated automatically (see §1). This *is* Experiment 1.

### 2.5 Configs

`configs/model/{sbdr,sbdr_probs,sbdr_aux}.yaml`, each mirroring `cibhash.yaml`
(same `transforms`, `backbone`, `optim`, `epochs`, `batch_size`), adding
`eps`, `beta`, `act`, `code_domain: unit`, `dist_metric: overlap`.
Keep `zero_mean_eval` off.

### 2.6 Metrics — `utils/sbdr_metrics.py`

Computed over the database split at each eval:

- `kappa` — mean active bits; per-bit usage vector; **dead-bit count**.
- `bit_entropy` — normalized. *This is the head-to-head against BiHalf, whose
  entire premise is maximizing bit entropy as an explicit objective.* If we match
  it without an explicit term, that is a clean in-repo result.
- `binarity` — fraction of `z` within 1e-2 of {0,1}; plus a histogram for the paper figure.
- **Arms C/D:** mean Bernoulli entropy `H(p)`, and *code agreement* — sample `b`
  twice per input, report the fraction of bits equal. This is the quantitative
  version of "sampling becomes near-deterministic".
- Optional: ST-path gradient variance across minibatches, A vs C vs D.

---

## 3. The four arms

| Arm | Head | Sampling | Loss | Isolates |
|---|---|---|---|---|
| **A** | CIBHash (Bernoulli + STE) | yes | CIBHash | baseline |
| **B** | clip/sigmoid, no sampling | no | ours on `z` | full proposal |
| **C** | CIBHash **unchanged** | yes | ours on probabilities `p` | **loss alone** |
| **D** | CIBHash **unchanged** | yes | CIBHash + `β`·ours on `p` | **drop-in gain** |

**Arm C** is the clean ablation: identical architecture and sampling, only the
objective differs. **Arm D** is our loss as an auxiliary regularizer on an
unmodified baseline — if `β>0` beats `β=0`, that is a self-contained result
requiring no architectural argument, and it is the cheapest positive outcome.

**Mechanism claim for C/D:** sparse, sharp `p` ⟹ lower Bernoulli entropy ⟹
near-deterministic sampling ⟹ lower STE gradient variance. This directly targets
a stated open problem in the literature (noise from skipped gradients
destabilizing contrastive hashing). **Measure it, don't assert it.**

> **Scope caveat for Arm C.** If the loss touches only `p` and never the sampled
> `b`, no gradient flows through the STE path during training. That is fine, but
> it means the *reduced-STE-variance* claim properly belongs to **Arm D**, where
> CIBHash's loss still consumes `b`. Keep this straight when writing up.

---

## 4. Two matching regimes

**(a) Matched code length** — `nbit=64` for everyone, ours with `κ≈8`.
Storage is **identical** (both are d-bit bitmaps). Claim: 8× fewer active bits at
equal storage and comparable mAP. **This is the honest headline.**

**(b) Matched active bits** — ours `nbit=512, κ=64` vs baselines `nbit=64` dense.
The SBDR "blessing of dimensionality" comparison, and plausibly where we win mAP.

> **MUST state the storage cost model for (b).** A 512-bit dense bitmap is 8×
> larger; index-based storage costs `κ·log₂d = 64·9 = 576` bits, still worse than
> 64. Frame (b) as a compute/accuracy trade, **not** a storage win. A reviewer
> will find this.

Run both. Lead with (a).

---

## 5. Experiments

| # | What | Deliverable | Wins without beating mAP? |
|---|---|---|---|
| 1 | mAP on `codes` vs `codes_cont`; activation histogram | binarization gap ≈ 0 vs baselines' gap | **yes** |
| 2 | mAP @ 16/32/64 bits, CIFAR-10(I) + NUS-WIDE | main table | no |
| 3 | `eps` sweep → measured κ vs `1.8·sqrt(eps·d)` | the law on real encoders | **yes** |
| 4 | Ablate: STE/Bernoulli **off** for CIBHash, **on** for us | emergent vs imposed | **yes** |
| 5 | Bit entropy / dead bits vs BiHalf | uniform usage for free | **yes** |
| 6 | High-d sparse (regime b) | SBDR dimensionality claim | maybe |
| 7 | `detach_mean` on/off | the self-contribution term matters | **yes** |

Experiments 1, 3, 4, 5 are the defensible core — none requires a SOTA number.

Reference mAP@1000, CIFAR-10(I), 16/32/64 bits:
CIBHash 0.427/0.463/0.473 · CIMON 0.451/0.472/0.494 ·
BiHalf 0.428/0.432/0.441 · GreedyHash 0.287/0.317/0.354.

---

## 6. Work order, with gates

**Step 0 — reproduce.** `python main_v2.py model=cibhash model.nbit=64 dataset=cifar10`, then `val.py`.
**GATE: mAP must match the published CIBHash number (~0.47 @ 64 bits) within noise.**
If not, stop. Nothing downstream is interpretable. Record exact backbone / optim /
epochs / augmentation; all arms inherit these.

**Step 1 — Arm D** (smallest diff: one added loss term, no arch change).
**GATE: does `β>0` improve mAP over `β=0`?** If yes, a result already exists.

**Step 2 — Arm C.**
**GATE: is mean `H(p)` lower than Arm A's, and code agreement higher?**

**Step 3 — Arm B** + the §2.1 eval work + §2.6 metrics.
**GATE: binarization gap ≈ 0, and realised κ tracks `1.8·sqrt(eps·d)`.**

**Step 4 —** full sweeps (hydra multirun over `model × nbit × dataset`), NUS-WIDE,
high-d regime.

Front-loads the cheapest likely-positive result; defers eval engineering until the
loss is known to do something.

---

## 7. Failure modes to watch

1. **Silent domain corruption** (§2.1) — the single highest-risk item. Assert early.
2. **Detached mean** — halves regularization silently. Default `False`.
3. **`eps` misread as a numerical-stability constant** — realistic values are
   O(0.1–10), not 1e-8.
4. **`zero_mean_eval`** — must stay off; mean-centering destroys sparse structure.
5. **Backbone mismatch** — CIBHash uses vgg16, WCH uses vit_base_hf with a frozen
   backbone. Mixing makes tables incomparable.
6. **Collapse to κ→0 or κ→d** — if the box never binds or `eps` is far off target.
   Log κ every epoch from the start; it is the earliest warning signal.

---

## 8. What is already established (do not re-derive)

Verified numerically before this integration:

- `κ ≈ 1.8·sqrt(eps·d)`, exponents (0.445, 0.451) vs predicted (0.5, 0.5); binarity 1.000.
- The self-contribution term is ~49% of the effective gradient.
- The critic does **not** work against a reconstruction objective (radial force
  saturates at ≈2λ while the quadratic grows unboundedly). Five separate attempts
  failed for this one structural reason. Do not revisit it here.
- The conjunctive/product critic variant `Σ_u log(1 + ...)` drives `κ → d/2`
  (predicted 48, measured 47.9 at d=96). The log-of-sum form is necessary.
