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

> **Update (2026-09-03, partial evidence): the law above is wrong for trained
> encoders.** It was fitted on free codes (an unconstrained `[0,1]^d` box, no
> encoder, no data) — §0.3's own framing. Against actual trained Arm-D checkpoints
> (`sbdr_aux`, d ∈ {64,256,512}, various `eps`), the fitted relationship is instead
> `κ ≈ 0.85 · eps^0.302 · d^0.660`, i.e. close to `(eps·d²)^(1/3)` rather than
> `(eps·d)^(1/2)` — a materially different `d`-scaling (0.66 vs 0.5 measured, 0.451
> on free codes). Treat this as a partial-evidence finding, not a re-derived law:
> the fit comes from the same handful of `sbdr_aux` runs already on record in
> `logs/cifar10/`, not a dedicated sweep. **No further experiments on this for now**
> — the practical-inversion formula above should be read as approximate for a
> trained encoder, with the realised `κ` always verified per-checkpoint rather than
> trusted from the formula (as the section already recommended).

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

**Addendum: `topk_eval: null | int`, same function, same code path.** `eps` only
sets κ *in expectation* (§0.3); a single checkpoint's realised per-sample κ varies.
`topk_eval`, when set, keeps only the top-`k` largest continuous activations per
sample before binarizing (unit domain, `overlap`/`hamming` only) — an *exact*
per-sample κ, overriding the plain 0.5 threshold. This turns Experiment 3's
`eps`-sweep-across-checkpoints into an `eps`-sweep-across-checkpoints *plus* a free
κ-sweep within each checkpoint (mAP-vs-κ from one trained model, no retraining),
and gives Experiment 8 a clean way to ask "how does tie-block size change with κ
alone, holding the encoder fixed?". Default `null` (baselines untouched); loud
`ValueError` on domain/metric mismatch or an out-of-range `k`, matching the other
guards in this section.

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

### 2.2b Second-order critic (2026-09-03)

**Motivation.** The critic `g(z_a,z_b) = log(<z_a,z_b> + eps)` is the *first-order*
truncation of an exponential critic. Writing `s = <z_i,z_j>`:

```
eps * exp(s/eps) = eps + s + s²/(2·eps) + ...
```

Order 1 (§2.2, current) keeps only the linear term. Order ∞ is the untruncated
`eps·exp(s/eps)`, a plain dot-product critic with temperature `eps` — essentially
CIBHash's cosine-similarity InfoNCE. So the truncation order is a knob
interpolating between sparse-binary (order 1) and dense-SOTA (order ∞), with order
2 the next point on that path.

**Why it matters.** §0.1 already flags that `exp(g)` is linear in `z_b`, so the
order-1 denominator collapses to `eps + <z_i, z̄>` — every sample is repelled from
the *batch mean*, not from individual negatives. The loss has no pairwise
structure; it is a marginal-usage regularizer wearing a contrastive-loss shape.
That is the leading suspect for why retrieval underperforms despite the emergent
sparsity/binarization working exactly as predicted (Experiments 1–5's territory).
The `s²/(2·eps)` term is the first correction that depends on `z_i` and `z_j`
*jointly* rather than through their separate contributions to a mean — it restores
pairwise discrimination to the denominator.

**Formula** (`critic_order: 1 | 2`, `models/loss/sbdr.py`):

```
C = (1/K) Σ_j z_j z_jᵀ                                    # (d, d), K = batch of both views

denominator_i = eps + <z_i, z̄> + λ₂ · z_iᵀ C z_i
numerator_i   = eps + s⁺_i + λ₂ · (s⁺_i)²                 # s⁺_i = <z_i, z_j>, the positive pair

L_i = log(denominator_i) − log(numerator_i)
```

`λ₂` defaults to `1/(2·eps)`, the exact Taylor coefficient. `λ₂ = 0` (any
`critic_order`) is bit-identical to the order-1 loss — `critic_order=1` forces the
effective `λ₂` to 0 regardless of the configured value, so order-1 is a genuine
special case of order-2's code path, not a separate formula that happens to agree.

**Cost.** `z_iᵀ C z_i` is evaluated through the `d×d` matrix
`C = zallᵀ @ zall / K` — `O(K·d²)` to build, `O(N·d²)` to apply to `N` query rows —
never through the mathematically equivalent `O(K²·d)` pairwise double sum
`z_iᵀ C z_i = (1/K) Σ_j <z_i,z_j>²`, which is what a naive per-pair
implementation would do. `C` is **not detached** — sample `i` contributes to `C`
via `zall`, the same self-contribution issue as `z̄` in §0.1 (~49% of the
effective gradient there); detaching would silently remove part of it here too.

Unit tests in `tests/test_sbdr_second_order.py` pin: the `C`-matrix path against a
naive `O(K²d)` double loop (~1e-6), `λ₂=0` reproducing order-1 exactly (including
when `critic_order=1` is given a nonzero `λ₂` alongside it — it must be ignored),
the default `λ₂ = 1/(2·eps)`, and that gradient reaches both `z̄` and `C` (i.e.
neither is accidentally detached).

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

## 4. Matching regimes

**(a) Matched code length** — `nbit=64` for everyone, ours with `κ≈8`.
Storage is **identical** (both are d-bit bitmaps). Claim: 8× fewer active bits at
equal storage and comparable mAP. **This is the honest headline.**

**(b) Matched active bits** — ours `nbit=512, κ=64` vs baselines `nbit=64` dense.
The SBDR "blessing of dimensionality" comparison, and plausibly where we win mAP.

> **Storage cost model for (b), corrected.** A dense-bitmap comparison (512-bit
> code = 8× a 64-bit one) is the wrong model for *very* sparse high-`d` codes.
> **Index storage** — `κ·log₂d` bits, i.e. store the κ active positions rather
> than the whole bitmap — is: `d=512, κ=64 → 64·log₂512 = 64·9 = 576` bits (still
> ~9× a dense 64-bit code, so (b) is a compute/accuracy trade, not a storage
> win); `d=1024, κ=8 → 8·log₂1024 = 80` bits, only ~25% over a dense 64-bit
> code (see §4(c) below — that pair carries ≈64.7 bits of actual capacity, so
> the 80-bit index cost is close to the information-theoretic floor, not an
> 8× penalty). Whether index storage or a dense bitmap is the right model
> depends entirely on `κ/d`; state which one applies for the specific `(d,κ)`
> pair being reported, every time.

**(c) Matched capacity** — a κ-sparse code over `d` bits carries at most
`log₂ C(d,κ)` bits of information, not `d`. So `nbit=64, κ=8` is really a
**~32-bit** code hiding inside a 64-bit bitmap, not a 64-bit one:

| `(d, κ)` | `log₂ C(d,κ)` (exact) | ≈ dense-bit equivalent |
|---|---|---|
| (64, 8) | 32.0 | 32 |
| (128, 16) | 66.3 | 66 |
| (1024, 8) | 64.7 | 65 |

`(128,16)` and `(1024,8)` both land within a bit of dense-64 capacity despite
wildly different `d`, which is exactly the point: capacity is set by `(d,κ)`
jointly, not by `d` alone, and regime (b)'s `nbit=512, κ=64` should be read
against *this* table, not against nominal bit count. Use this regime to pick
`(d,κ)` pairs that are capacity-matched to a 64-bit dense baseline (rather than
storage- or nbit-matched) as a third point of comparison alongside (a) and (b).

Run (a) and (b). Lead with (a). Treat (c) as a lens for choosing `(d,κ)` pairs
in (b), not a fourth full experiment sweep.

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
| 8 | Overlap-value histogram + tie-block sizes on the full DB | similarity-resolution risk, not just capacity | **yes** |

Experiments 1, 3, 4, 5, 8 are the defensible core — none requires a SOTA number.

**Reference mAP@1000, CIFAR-10(I), VGG16 features, 16/32/64 bits** (SDC paper Table 1 —
this is the number Step 0 actually reproduces against; see §6):

| Method | 16 | 32 | 64 |
|---|---|---|---|
| ITQ | 46.8 | 51.3 | 54.4 |
| GreedyHash | 44.9 | 51.9 | 55.7 |
| TBH | 48.2 | 50.2 | 50.7 |
| BiHalf | 54.7 | 58.1 | 60.6 |
| CIBHash | 56.2 | 59.2 | 61.2 |
| SDC | 59.8 | 64.0 | 66.3 |

**The real Experiment-2 competitor is SDC at 66.3, not CIBHash at 61.2** — SDC is
in this same repo (`models/{arch,loss}/sdc.py`) and trivially re-runnable by a
reviewer with `model=sdc`, so a table that only beats CIBHash invites the obvious
"why not compare to the paper this repo is actually about?" SDC with ResNet50
reaches 78.4 — backbone dominates method choice by ~12 points at fixed method, so
every arm MUST stay on CIBHash's VGG16 config (§1.1) or comparisons are void
regardless of which baseline is cited.

### 5.1 Experiment 8 — similarity resolution

Probably a **bigger risk than capacity (§4c).** Overlap similarity
`<b_i,b_j> ∈ {0, ..., κ}` has only `κ+1` distinct levels — e.g. **9** for `κ=8` —
versus 65 distinct levels for dense 64-bit Hamming. Against a 59,000-image CIFAR-10
database (this repo's own protocol, §1) that means enormous tie blocks at each
overlap value, and `get_rank` (`utils/hashing.py`, `torch.topk`) breaks ties by
implementation-defined order — effectively database index — not similarity. mAP
under `overlap` can then be partly an artifact of dataset ordering rather than
retrieval quality. This is exactly the *similarity collapse* problem the SDC paper
(this repo, §1.1) exists to address, just showing up from the opposite direction
(too few *achievable* similarity values, vs. SDC's too little *spread* across the
achievable range).

**Deliverable:** for each arm/κ under evaluation, on the database split: a
histogram of pairwise overlap values, the count of distinct levels actually
realised, and the tie-block size distribution at the R cutoff used for mAP. If
ties dominate (most of a query's neighborhood at a single overlap value), the fix
is a **larger κ** (pointing at `d=512, κ=64` from regime (b), which has 65 levels —
matching dense-64 resolution), not more nominal capacity (§4c's `(1024,8)` has the
same 9 levels as `(64,8)` despite ~2× the capacity — capacity and resolution are
different axes).

**Possible synergy, not required for the core experiments:** SDC's own
contribution is calibrating the *similarity distribution* toward a target spread
(beta calibration) — directly applicable to our overlap distribution once it's
measured here. Composition, not competition, with this repo's own method.

---

## 6. Work order, with gates

**Step 0 — reproduce. PASSED (2026-09-02).**
`python main_v2.py model=cibhash model.nbit=64 dataset=cifar10`, then `val.py`.
**GATE: mAP must match CIBHash @ 64 bits from this repo's own paper (SDC, Table 1,
§5) within noise — 0.612, not the original CIBHash paper's 0.473** (that number was
the wrong reference; see the correction below the gate). Achieved: **0.6167** best
training-loop mAP (epoch 90/100), **0.6168** from an independent `val.py` run on
that checkpoint — the two eval paths agree, and both are within noise of 0.612.
GPU 2, VGG16 backbone, Adam lr=1e-4, 100 epochs, CIBHash's default augmentation
(`configs/transforms/cibhash.yaml`). All arms inherit these settings unchanged
(§1.1).

> **Correction to the original gate number.** This section originally cited 0.473,
> CIBHash @ 64 bits from the *original CIBHash paper's own* protocol. That was the
> wrong reference for this repo: the number to reproduce against is this repo's own
> paper (SDC, BMVC 2023, Table 1 — VGG16 features, CIFAR-10(I), mAP@1000), which
> reports **CIBHash @ 64 bits = 0.612**, not 0.473. The two papers' CIBHash
> re-implementations evidently differ enough in protocol detail to produce a ~30%
> relative gap; this repo's own reported number is the correct target since it's
> what every arm here is actually being compared against (see the full table in
> §5). The training run itself was correct on the first attempt against the right
> reference — the gate's reference number was the bug, not the run.

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

**Closed negatives from continued work (2026-09-03) — do not re-run:**

- **Arm D (auxiliary loss) is dead.** `β=0 → 0.6144±0.0010`, `β=0.03 → 0.6168±0.0038`
  over 3 seeds, `p≈0.32` — no significant effect. Logs:
  `logs/cifar10/sbdr_aux64_100/rep_beta0_seed{42,43,44}_*`,
  `logs/cifar10/sbdr_aux64_100/rep_beta003_seed{42,43,44}_*`.
- **The high-d regime (§4b) fails.** At matched `κ` via `topk_eval`, `d=64 > d=256 >
  d=512` uniformly — mAP gets *worse* as `d` grows at fixed active-bit budget, the
  opposite of the SBDR "blessing of dimensionality" claim §4b hoped to reproduce.
  Logs: `logs/cifar10/sbdr_aux{64,256,512}_40/`.
- **The §0.3 sparsity law is wrong for trained encoders** — see the update appended
  to §0.3 directly. Partial evidence, no further experiments on it for now.

Every run behind these three points used the `sbdr_aux` trainer (Arm D: CIBHash
loss + ours as an auxiliary term on CIBHash's unmodified architecture) — **Arm B
(our loss alone, on the bounded head of §2.3/§2.4) had not been run before the
second-order critic sweep below.**

---

## 9. Arm B, second-order critic sweep (2026-09-03)

First Arm B runs on record. `d=64`, `eps=0.31`, 40 epochs, `critic_order=2`,
`λ₂ ∈ {0, 0.4, 0.8, 1.6, 3.2}` (`λ₂=0` is the Arm B order-1 baseline; `λ₂=1.6 ≈
1/(2·0.31)` is the Taylor value). Logs:
`logs/cifar10/sbdr64_40/lambda2_{0,0.4,0.8,1.6,3.2}_42_*`. Per-run metrics in
`<logdir>/sbdr_report.json`; combined in `experiments/sbdr_sweep_report.json`.
`eval_interval=10` (config default, not overridden) — eval only ran at epochs
10/20/30/40.

### 9.1 Training-curve observation (raw numbers from `train_history.json` / `test_history.json`)

For `λ₂ ∈ {0, 0.4, 0.8}`, `train_contrast` loss and `train_kappa` go flat partway
through the 40 epochs and stay flat:

| `λ₂` | last epoch with moving loss | flat-loss value (all subsequent epochs) | flat-`kappa` value | test mAP at ep10 / 20 / 30 / 40 |
|---|---|---|---|---|
| 0   | ep9  (loss -0.010)  | ep10–40: `0.0000` (±1e-4)  | `7.00`  | 0.1002 / 0.1003 / 0.1003 / 0.1003 |
| 0.4 | ep16 (loss -0.301)  | ep19–40: `-0.01` to `-0.02` | `~11.1–11.5` | 0.3849 / 0.0859 / 0.1112 / 0.1100 |
| 0.8 | ep30 (loss -1.040)  | ep32–40: `-0.0001` to `-0.0003` | `~6.0`  | 0.4555 / 0.4489 / 0.3478 / 0.1009 |
| 1.6 | still moving at ep40 (loss trend: -1.51→ decreasing `train_kappa` from 6.4 to lower over training; no flattening observed) | n/a | n/a | 0.4205 / 0.4626 / 0.4793 / 0.5117 |
| 3.2 | still moving at ep40 | n/a | n/a | 0.4628 / 0.4991 / 0.4974 / 0.5302 |

Full per-epoch `train_loss`/`train_kappa` in each run's `train_history.json`;
per-eval-epoch `mAP` in `test_history.json`.

Because `db_best.pth`/`test_best.pth` are saved only on a new best `test mAP`
(`experiments/train_helper.py`), and eval only ran at epochs 10/20/30/40, the
saved "best" checkpoint for `λ₂=0` is the epoch-20 state (`mAP=0.1003`), already
inside the flat-loss region reported above — there is no saved checkpoint from
before epoch 10 for any run in this sweep.

### 9.2 Per-run metrics at the saved best checkpoint (`sbdr_report.json`)

| `λ₂` | best ep | mAP native | mAP topk8 | mAP topk16 | mAP topk32 | κ mean±std | binarity | dead bits | overlap mean±std (db, sampled) | overlap=0 frac | pos-pair overlap mean | rand-pair overlap mean | separation ratio | FN-rate@50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0   | 20 | 0.1003 | 0.1003 | 0.1003 | 0.1003 | 7.00±0.03  | 1.000 | 55 | 7.00±0.00  | 0.000 | 7.00 | 7.00 | 1.00  | 0.1001 |
| 0.4 | 10 | 0.3849 | 0.3933 | 0.3882 | 0.3854 | 6.84±1.59  | 0.993 | 5  | 1.18±1.64  | 0.519 | 5.23 | 1.20 | 4.37  | 0.2479 |
| 0.8 | 10 | 0.4555 | 0.4684 | 0.4652 | 0.4491 | 5.90±1.34  | 0.984 | 1  | 0.67±1.17  | 0.659 | 4.46 | 0.67 | 6.64  | 0.3081 |
| 1.6 | 40 | 0.5117 | 0.5095 | 0.4990 | 0.4721 | 4.54±1.03  | 0.974 | 1  | 0.34±0.73  | 0.774 | 3.69 | 0.34 | 10.94 | 0.3853 |
| 3.2 | 40 | 0.5302 | 0.5307 | 0.5238 | 0.4927 | 3.90±0.96  | 0.967 | 1  | 0.25±0.60  | 0.816 | 3.23 | 0.25 | 12.97 | 0.4252 |

Overlap distinct-value counts (of `κ+1` possible): `λ₂=0` → 1 distinct value (all
sampled pairs `=7.0`, `4498500` pairs sampled); `0.4` → 13 (7 carrying >1% mass);
`0.8` → 10 (6 >1%); `1.6` → 8 (4 >1%); `3.2` → 7 (4 >1%).

Tie-block size at `R=1000` (mean±std over 1000 test queries against the 59000-item
db; `median`/`max` in parens): `λ₂=0` → `58999.97±0.80` (median `59000`, i.e. the
entire database is one tie block); `0.4` → `2009.60±1302.44` (median `1519`, max
`8745`); `0.8` → `1885.93±1019.92` (median `1606`, max `9841`); `1.6` →
`1669.49±747.08` (median `1467`, max `5232`); `3.2` → `1910.79±1031.59` (median
`1751`, max `11101`).

`n_positive_pairs=1024` (from `trainer.dataloader['train']`, augmented-view pairs)
for every row; `n_pairs_sampled=4498500` (3000 random db items, all off-diagonal
pairs) for every overlap-distribution row.

---

## 10. Activation ablation + batch/optimizer match (2026-09-03 continued)

**§9's runs all used `act=clip`** (`configs/model/sbdr.yaml` sets `act: clip`,
never overridden in the §9 sweep). Collapse-detection diagnostics added to
`trainers/sbdr.py::train_one_batch`/`train_one_epoch` (logged every epoch to
`train_history.json` as `train_usage_mean`, `train_usage_std`,
`train_dead_bits_exact` [`z_bar_u == 0.0` exactly], `train_dead_bits_near`
[`z_bar_u < 1e-4`, inclusive], `train_overlap_std`, `train_kappa_std`; a
`logging.warning` fires in `log.txt` whenever epoch-avg `dead_bits_exact` rises;
per-epoch `(epoch, 64)` snapshot of `z_bar` saved to `<logdir>/usage_history.pt`).

### 10.1 Task A — `act ∈ {clip, sigmoid} × λ₂ ∈ {0, 0.8, 1.6}`, d=64, ε=0.31, 100 epochs, eval_interval=2

Logs: `logs/cifar10/sbdr64_100/act{clip,sigmoid}_lambda2_{0,0.8,1.6}_42_*`.
Per-run metrics: `<logdir>/sbdr_report.json`; combined:
`experiments/sbdr_taskA_report.json`. Full per-epoch curves in each run's
`train_history.json`/`test_history.json`/`usage_history.pt`.

| act | λ₂ | flat/collapse onset (epoch) | max `dead_bits_exact` (any epoch) | final (ep100) `dead_bits_exact` | best mAP (epoch) | final (ep100) mAP |
|---|---|---|---|---|---|---|
| sigmoid | 0   | 8  | 56.00 | 55.90 | 0.2757 (ep4)  | 0.1000 |
| clip    | 0   | 12 | 58.00 | 58.00 | 0.3884 (ep2)  | 0.1000 |
| sigmoid | 0.8 | never (100 epochs) | 0.00 | 0.00 | 0.4467 (ep96) | 0.4404 |
| clip    | 0.8 | 16 | 56.00 | 56.00 | 0.4529 (ep2)  | 0.1000 |
| sigmoid | 1.6 | never (100 epochs) | 0.00 | 0.00 | 0.4767 (ep96) | 0.4752 |
| clip    | 1.6 | never (100 epochs) | 6.96 (transient, recovered) | 2.22 | 0.5380 (ep100) | 0.5380 |

"Flat/collapse onset" = first epoch of 3+ consecutive epochs with
`|train_contrast| < 1e-3` (heuristic on the raw `train_history.json` values). At
`λ₂=0`, **sigmoid collapsed faster than clip** (ep8 vs ep12), and both reach a
stable terminal state (loss exactly `0.0000`, integer `kappa`, `overlap_std`
exactly `0.0000`) that persists unchanged through epoch 100. At `λ₂=0.8` and
`λ₂=1.6`, **clip collapses (at `λ₂=0.8`) or nearly does (at `λ₂=1.6`, transient
peak `dead_bits_exact=6.96` before dropping back to 2.22) while sigmoid's
`dead_bits_exact` stays at exactly `0.00` for every single epoch of both
100-epoch runs.**

Full per-run `sbdr_report.json` metrics (best-checkpoint, i.e. `db_best.pth`/`test_best.pth`):

| act | λ₂ | mAP native | mAP@κ8 | κ mean | dead bits (post-hoc, best ckpt) | binarity | overlap mean±std | separation ratio | FN-rate@50 |
|---|---|---|---|---|---|---|---|---|---|
| clip    | 0   | 0.3884 | 0.3922 | 6.358 | 5  | 0.9774 | 0.863±1.703 | 5.511  | 0.2348 |
| sigmoid | 0   | 0.2757 | 0.2759 | 9.178 | 14 | 0.9823 | 2.127±3.796 | 3.079  | 0.1116 |
| clip    | 0.8 | 0.4529 | 0.4574 | 5.767 | 3  | 0.9695 | 0.743±1.273 | 5.518  | 0.3063 |
| sigmoid | 0.8 | 0.4467 | 0.4627 | 5.052 | 0  | 0.9723 | 0.408±0.846 | 10.479 | 0.3391 |
| clip    | 1.6 | 0.5380 | 0.5353 | 4.329 | 2  | 0.9769 | 0.311±0.684 | 12.308 | 0.4001 |
| sigmoid | 1.6 | 0.4767 | 0.4879 | 4.367 | 0  | 0.9683 | 0.308±0.690 | 12.214 | 0.3431 |

(Note: post-hoc `dead bits` here is on the saved best checkpoint's continuous
`codes_cont`, thresholded per `utils.sbdr_metrics.usage_stats`'s `usage==0`
predicate over the full 59000-item db — a different quantity from
`train_dead_bits_exact`, which is `z_bar_u==0.0` on a single training batch. The
two agree qualitatively but are not the same measurement.)

**Best-performing Task A cell: `act=clip, λ₂=1.6`, mAP=0.5380 at ep100** (used
for Task C).

### 10.2 Task C — best cell (`clip, λ₂=1.6`) at `batch_size=128`, AdamW `lr=1e-3, weight_decay=1e-5`

Config: `configs/optim/adamw.yaml` (new). Run:
`main_v2.py model=sbdr dataset=cifar10 model.act=clip criterion.critic_order=2
criterion.lambda2=1.6 epochs=100 eval_interval=2 batch_size=128 optim=adamw`.
Log: `logs/cifar10/sbdr64_100/taskC_bs128_adamw_actclip_lambda2_1.6_42_260903_184150_517623/`.
Metrics: `<logdir>/sbdr_report.json`.

Collapsed by epoch 2 (first eval), confirmed as a stable terminal state through
epoch 18 (`train_contrast=0.0000`, `train_dead_bits_exact=18.00` exactly, for 17
consecutive logged epochs) before the run was stopped early (all per-epoch data
through ep18, and the ep2 best-checkpoint's `db_best.pth`/`test_best.pth`, are on
disk and complete):

| quantity | batch=64, Adam lr=1e-4 (§10.1 clip/λ₂=1.6) | batch=128, AdamW lr=1e-3 (this run) |
|---|---|---|
| collapse onset | never (100 ep) | ep 2 (first eval) |
| terminal/best `train_kappa` | 4.329 (never-collapsed) | 46.0 exactly |
| terminal `dead_bits_exact` | 2.22 (ep100, transient peak 6.96) | 18.00 exactly (stable, ep2–18) |
| best-checkpoint mAP native | 0.5380 | 0.1000 |
| best-checkpoint mAP@κ8/16/32 | 0.5353 / — / — | 0.1000 / 0.1000 / 0.1000 |
| best-checkpoint binarity | 0.9769 | 1.0000 |
| best-checkpoint dead/saturated bits | 2 / — | 18 / 46 |
| best-checkpoint overlap mean±std | 0.311±0.684 | 46.0±0.0 |
| best-checkpoint overlap=0 frac | — | 0.0000 |
| best-checkpoint tie-block @R=1000 (mean/median/max) | — | 59000.0 / 59000.0 / 59000 |
| best-checkpoint separation ratio | 12.308 | 1.0000 |
| best-checkpoint FN-rate@50 | 0.4001 | 0.1000 |

Full clip/λ₂=1.6 (batch=64) numbers cross-referenced from §10.1's table above.

---

## 11. Two distinct failure modes; §9/§10's conclusions superseded pending further work (2026-09-04)

**§9/§10's "the objective collapses" conclusion is superseded.** Diagnostics (this
section's predecessor investigation, 2026-09-03/04) established the loss math is
correct: gradients match finite differences to ~1e-11 (`tests/test_sbdr_loss_math.py`
2a), closed-form references match (identical codes → exactly 0.0; disjoint-limit
→ -3.395 vs analytic -3.402; random κ=9 → -1.775 vs analytic -1.776), positive
pairs are correctly paired (§ below), no stray `.detach()`/`no_grad()` on `z̄` or
`C`, no AMP anywhere. Instead, two distinct, activation-specific failure modes
were found and confirmed:

**Failure 1 — sigmoid never trains.** At init, near-zero pre-activations map to
≈0.5 under sigmoid, so `t_i ≈ s_i ≈ 16` (dense, κ≈30-32/64) and the loss sits at
≈-0.0005 with head-weight gradient norm 0.0062 — 266× smaller than clip's 1.66 at
the same init. This explains §10's odd finding that sigmoid collapsed *faster*
than clip at λ₂=0.

**Failure 2 — clip trains, then ratchets to death (confirmed, Task A below).**
At init, clip's κ=0 for every sample (pre-activation std 0.073 ≪ ε=0.31 swamps
the inner products), but clip's gradient (1.66 at init, non-zero, unlike
sigmoid) *does* escape this: a single-batch overfit reaches loss ≈ -1.96 with
62-64 distinct codes within 300 steps. Extended to 5000 steps
(`experiments/sbdr_diagnose_ratchet.py`, act=clip, λ₂=0, one fixed real batch,
Adam lr=1e-4 wd=1e-5, backbone_lr_scale=1 — the actual §9/§10 optimizer config):

| step | loss | κ mean±std | distinct codes/64 | dead_bits_exact |
|---|---|---|---|---|
| 1 | -0.0758 | 0.000±0.000 | 1 | 2 |
| 250 | -1.9382 | 4.250±1.141 | 64 | 3 |
| 1000 | -1.9390 | 4.109±1.143 | 63 | 2 |
| 1500 | -1.8670 | 4.469±1.168 | 60 | 2 |
| **1750** | **0.0000** | **4.000±0.000** | **1** | **60** |
| 3000 | 0.0000 | 4.000±0.000 | 1 | 60 |
| 5000 | 0.0000 | 4.000±0.000 | 1 | 60 |

Dead bits accumulate from 2 to 60/64 between step 1500 and step 1750, then
**freeze exactly** (loss, κ, distinct-code-count all bit-identical) through step
5000 — while the per-unit pre-activation mean, logged in parallel, diverges to
~±1e5-5e5 magnitude over the same window and keeps drifting with **zero
observable effect** on the (already fully saturated) output or loss. **Confirms
the ratchet**: per the stated criterion, fixing initialization alone will not be
sufficient, since the collapse is a training-dynamics property, not an
init-state property. Full trajectory: `experiments/sbdr_ratchet_trajectory.pt`.

### 11.1 Task B — initialization fix attempts (`models/arch/sbdr.py`, no training)

Added `feature_norm: none|standardize|batchnorm` (BatchNorm1d, `affine=False`/
`True` resp., before the encoder) and `head_init_gain: float` (post-init
multiplicative scale on the final Linear's weights), both off by default.
Diagnostic: `experiments/sbdr_diagnose_init_fix.py`, one fixed real batch,
act=clip, same backbone/head init seed across configs.

| config | pre-act std | κ@init | distinct codes/64 | head grad norm | loss@init |
|---|---|---|---|---|---|
| baseline (none, gain=1.0) | 0.075 | 0.000 | 1 | 1.702 | -0.0973 |
| standardize, gain=1.0 | 0.194 | 1.953 | 41 | 2.886 | -0.1465 |
| batchnorm, gain=1.0 | 0.194 | 1.953 | 41 | 2.886 | -0.1465 |
| none, gain=6.3 (std target ≈0.46) | **0.470** | 19.688 | 64 | 3.195 | -0.1543 |
| standardize + gain=6.3 | 1.219 | 24.625 | 64 | 3.264 | -0.1581 |
| none, gain=15.0 | 1.118 | 29.500 | 64 | 3.062 | -0.1289 |
| none, gain=30.0 | 2.237 | 33.047 | 64 | 3.363 | -0.1149 |
| none, gain=60.0 | 4.474 | 34.984 | 64 | 4.248 | -0.1075 |
| standardize + gain=15.0 | 2.902 | 29.344 | 64 | 4.074 | -0.1384 |
| standardize + gain=30.0 | 5.804 | 31.438 | 64 | 5.531 | -0.1306 |

Target: loss ≈ -1.7764 (analytic, random κ=9, d=64). `none, gain=6.3` hits the
predicted std almost exactly (0.470 vs predicted 0.46) — **but loss stays at
-0.154, not -1.78**. Across the full swept range (std 0.075 → 5.804, three
orders of magnitude), loss stays inside [-0.16, -0.11] throughout; it does not
move toward -1.78 as std increases. At `none, gain=6.3`: t_i mean=10.62,
s_i mean=12.54 (ratio 0.85); at the genuinely-random-code reference (§4(iii)):
t mean=1.58, s mean=9.31 (ratio 0.17) — t and s scale together as std grows here,
rather than s pulling away from t as in the reference. **No tested config reaches
the loss target; per the stated instruction, none were carried into training.**

### 11.2 Task C — delta offset (`models/loss/sbdr.py`, `delta` config) + `ste_clip` activation (`models/arch/sbdr.py`)

`t_i = <z_i,z̄> - δ·‖z_i‖₁ + ε`, clamped at `ε/2`; `δ=0` reproduces the
undamped critic bit-for-bit (verified: `torch.equal` on a random batch). `act=
ste_clip` added: value = `clamp(x,0,1)`, backward = identity everywhere (`z =
x + (clamp(x,0,1)-x).detach()`). 5000-step single-batch tests, `δ∈{0,0.010}` ×
`act∈{clip,sigmoid,ste_clip}` (clip/δ=0 reused from Task A above, not rerun):

| act | δ | dead_bits_exact: first→last (max) | κ mean: first→last | loss: first→last (min) | outcome |
|---|---|---|---|---|---|
| clip | 0 | 2→60 (60) | 0.000→4.000 | -0.076→0.000 (~-1.94) | full collapse, frozen at step ~1750 (Task A) |
| clip | 0.010 | 2→46 (46) | 0.000→18.000 | -0.125→-0.0099 (-2.066) | fewer dead bits than δ=0, still degrading |
| sigmoid | 0 | 0→44 (44) | 31.938→20.000 | -0.00059→0.000 (-1.861) | ratchets to collapse more slowly than clip, ends at exactly 0 |
| sigmoid | 0.010 | 0→54 (54) | 31.938→10.000 | -0.0204→-0.0097 (-1.952) | *more* dead bits than δ=0 (54 vs 44); still degrading |
| ste_clip | 0 | 2→64 (64, all) | 0.000→0.000 | -0.0758→0.000 (-0.0758) | full 64/64 collapse, immediate, loss never improves |
| ste_clip | 0.010 | 2→56 (56, at crash) | 0.000→1.922 | -0.125→-0.192 (-0.192) | **crashed at step 80**: `z` out of `[0,1]` (float32 cancellation in the STE add/detach at pre-activation abs-max 6.065e8) |

Predicted pattern ("clip+δ dies; sigmoid+δ and ste_clip+δ survive") **not
observed**: all six cells degrade within 5000 steps (dead-bit accumulation
and/or numerical divergence); δ=0.010 has a small mitigating effect for clip
(46 vs 60 max dead bits) but the opposite effect for sigmoid (54 vs 44); `ste_clip`
is the worst performer in both conditions tested (full collapse at δ=0,
fastest-observed numerical blow-up at δ=0.010). Full trajectories:
`experiments/sbdr_delta_trajectory_*.pt`.

### 11.3 Status

Neither Task B (init fix) nor Task C (delta offset / ste_clip) produced a
configuration meeting its own stated bar (Task B: loss ≈ -1.78 at init; Task C:
the predicted survive/die pattern) before Task D's full 100-epoch key-cell runs
were to be launched. Task D has not been run.

### 11.4 Task A/B follow-up — coupling vs. implementation hypothesis for the ratchet (2026-09-04, new)

Two hypotheses for *why* the order-1/clip ratchet (§11 above) lands on the
all-samples-one-code degenerate manifold: (1) shared-backbone gradient coupling
correlates per-sample updates and pulls all samples' codes together; (2)
something specific to this repo's head/init/optimizer, independent of
cross-sample feature correlation. Diagnostic: `experiments/sbdr_diagnose_ratchet_hdiv.py`
(extends `sbdr_diagnose_ratchet.py` — same fixed real batch, seed, act=clip,
critic_order=1, Adam lr=1e-4 wd=1e-5, 5000 steps — adding per-step cross-sample
diversity of the backbone features `h_i` feeding the encoder, pairwise cosine
similarity and Euclidean distance, mean+var over the batch, and of the
binarized codes, pairwise Hamming distance mean+var).

**Rerun of the original (unfrozen) config** reproduces the collapse
qualitatively (loss→0.0000 exact, distinct codes→1, dead_bits_exact jumps to
56/64) but at a later step (~3000-3050 vs. the original run's ~1550-1600) and
onto a different point on the degenerate manifold (κ=8.000±0.000 this time vs.
κ=4.000±0.000 originally) — expected run-to-run variance (GPU/cuDNN
nondeterminism), consistent with "one absorbing state out of a manifold of
them," not a fixed step or fixed code. Re-inspecting the *original* trajectory
(`sbdr_ratchet_trajectory.pt`) at full 50-step resolution (finer than the
7-row table above) shows the same pattern: distinct codes 58→1 and
dead_bits_exact 4→60 within a single logged 50-step window (1550→1600), not
gradually across 1500-1750 as the sparse table suggested — the transition is a
sharp, ~50-step event in both runs, not a slow drift.

**Backbone-feature diversity collapses in the same window as code diversity,
not before or after it.** In the rerun, at step 3000 (pre-collapse):
distinct codes=60/64, feature pairwise cosine mean=0.227 (var 0.0157) — clearly
non-degenerate. At step 3050 (one logging interval later): distinct codes=1,
feature cosine mean=0.999 (var ≈0.0000). Both the output-code collapse and the
backbone-feature collapse complete within the same ≤50-step window; pre-collapse
feature cosine mean was stable/slowly drifting (~0.21-0.26) for the preceding
~2750 steps, so this is a sudden joint event, not a slow feature-diversity
erosion that gradually drags codes down with it.

**Freezing the backbone prevents the collapse entirely over the same 5000-step
horizon.** `--frozen_backbone` (mirrors `trainers/base.py`'s
`backbone_lr_scale=0` path exactly: backbone params `requires_grad_(False)`,
excluded from the optimizer): feature cosine mean is pinned at its init value
(0.4590, by construction) for all 5000 steps. Loss reaches -1.948 (min of any
config seen so far, including the init-fix sweep in §11.1) by step 250 and
**stays there, bit-stable, through step 5000** — distinct codes settle at
62/64, dead_bits_exact at 5/64, both flat. No collapse onset, no ratchet, no
ambiguous slow drift — a clean non-collapsing terminal state.

| run | backbone | terminal step 5000: loss / κ / distinct / dead_bits | collapse? | step of collapse onset |
|---|---|---|---|---|---|
| original (§11 above) | trainable | 0.0000 / 4.000±0.000 / 1 / 60 | yes | ~1550-1600 |
| rerun (this section) | trainable | 0.0000 / 8.000±0.000 / 1 / 56 | yes | ~3000-3050 |
| frozen-backbone (this section) | frozen | -1.948 / 4.531±0.992 / 62 / 5 | no | n/a (stable 250→5000) |

**Assessment.** This favors hypothesis 1 (architecture/gradient-coupling)
over hypothesis 2 (something idiosyncratic to this repo's head/init/optimizer
independent of input correlation): removing the shared-backbone gradient path
removes the collapse outright, on the same batch/seed/optimizer/loss/activation
that reliably collapses twice when the backbone trains. The coincident-timing
result (feature and code diversity collapse in the same ~50-step window) is
consistent with, but does not by itself prove, this direction — it is also
consistent with the reverse causal story (head collapses first for a
head-internal reason, e.g. an Adam/clip-saturation effect, and the resulting
collapsed gradient signal then homogenizes the backbone features as a
*consequence*, not a cause). The frozen-backbone result is the one that
actually discriminates: with features unable to move, there is nothing for
that reverse story to act through, and collapse still doesn't happen — so the
homogenization is not merely a downstream symptom of a head-only failure mode.

**What this does not establish:** only one frozen-backbone seed/batch was run
(5000 steps, one fixed batch, one init) — not swept across seeds, batches, or
backbone architectures, so "collapse rate 0/1" is suggestive, not a
confidence interval. It also does not distinguish "inherent to the order-1
critic with *any* hard-saturating activation, given correlated multi-sample
gradients" from "specific to this repo's particular `Linear-ReLU-Linear` head
and VGG16 backbone pairing" — that would need testing other backbones
(§11's coupling story, per the original task framing, is about *any*
architecture where many samples' units saturate together under a shared
gradient step, which frozen-backbone alone can't isolate from
"VGG16-specific"). It also does not test whether a trainable backbone with
decorrelated per-sample features (e.g. forced via a diversity regularizer, or
a backbone architecture less prone to early-training feature homogeneity)
would avoid collapse without freezing anything — that would more directly
test the causal mechanism (correlated features → correlated gradient step →
joint saturation) rather than just removing the backbone gradient path
wholesale. One correction to the coupling hypothesis as originally stated: the
head is not a single `Linear` layer but `Linear(features_size,1024)-ReLU-
Linear(1024,nbit)` (`models/arch/sbdr.py`); the coupling argument (shared
input feature correlated across samples → correlated push on shared weights)
still applies, just through a two-layer head rather than one.

Full trajectories: `experiments/sbdr_ratchet_trajectory_unfrozen_hdiv.pt`,
`experiments/sbdr_ratchet_trajectory_frozen.pt`. Script:
`experiments/sbdr_diagnose_ratchet_hdiv.py`. GPU 2 used for both runs
(sequentially, released and confirmed idle via `nvidia-smi` between runs).

## 12. Frozen backbone adopted as protocol; full-length comparison (2026-09-04, new)

### 12.0 Task 0 — is the backbone pretrained, and was it being fine-tuned?

`configs/backbone/vgg16.yaml` sets `pretrained: True` -> `models/backbone/vgg16.py`
calls `torchvision.models.vgg16(pretrained=True)` (ImageNet weights). `configs/
train.yaml` sets `backbone_lr_scale: 1` by default and `configs/model/sbdr.yaml`
does not override it, so every collapsing run in §9-§11.4 was fine-tuning an
ImageNet-pretrained VGG16 end-to-end, not training one from random init. This
is the "pretrained-and-fine-tuned" branch: the practical fix is to freeze the
backbone rather than chase the from-scratch mechanism further (which does not
apply here).

### 12.1 10-epoch frozen-backbone check, Arm A vs. Arm B (`d=64`)

10 epochs, `backbone_lr_scale=0`, batch size 64, `eps=0.31`, seed 42, GPU 3
(GPU 2 occupied by another user's job throughout this section; never used).

| Arm | mAP @ ep5 | mAP @ ep10 (best) | κ mean±std | dead bits | binarity | separation ratio |
|---|---|---|---|---|---|---|
| A — CIBHash | 0.5819 | 0.5828 | n/a (signed) | n/a | n/a | n/a |
| B — SBDR (order-1, clip) | 0.5734 | 0.5815 | 5.18±1.05 | 4/64 | 0.937 | 7.96x |

No collapse in either arm; mAP close to parity. This already exceeds every
trainable-backbone Arm B config from §9's 40-epoch sweep (best there: 0.5302).

### 12.2 Full-length (100-epoch) frozen-backbone runs

Same protocol, extended to 100 epochs (matches §6's CIBHash reproduce
schedule; `configs/train.yaml`'s own default `epochs: 100`), eval every 10
epochs, seed 42, GPU 3 only, sequential, confirmed idle via `nvidia-smi`
before/after each run.

| epoch | A — CIBHash (`d=64`) | B — SBDR `d=64`, `eps=0.31` | B — SBDR `d=512`, `eps=1.0`, batch 128 |
|---|---|---|---|
| 10 | 0.5858 | 0.5817 | 0.6325 |
| 20 | 0.5941 | 0.5949 | 0.6431 |
| 30 | 0.6035 | 0.6011 | 0.6492 |
| 40 | 0.6084 | 0.6083 | 0.6497 |
| 50 | 0.6113 | 0.6151 | 0.6499 |
| 60 | 0.6139 | 0.6134 | 0.6491 |
| 70 | 0.6169 | 0.6032 | **0.6580 (best)** |
| 80 | 0.6189 | **0.6200 (best)** | 0.6562 |
| 90 | **0.6225 (best)** | 0.6143 | 0.6559 |
| 100 (final) | 0.6221 | 0.6158 | 0.6560 |

Runtimes: 0.69h / 0.69h / 0.73h. Arm B `d=64` tracks Arm A closely but does not
overtake it (best-mAP gap 0.0025, final-epoch gap 0.0063, and its curve is
noisier/non-monotonic where Arm A's is smooth). Arm B `d=512` clearly exceeds
Arm A throughout training (best +0.0355, ~5.7% relative) -- but see §12.3,
this comparison is confounded by active-bit count, not dimensionality alone.

Arm B diagnostics at best checkpoint:

| | `d=64`, `eps=0.31` (ep80) | `d=512`, `eps=1.0` (ep70) |
|---|---|---|
| κ mean±std | 4.70±1.18 | 24.36±5.57 |
| dead bits | 4/64 | 6/512 |
| binarity | 0.966 | 0.969 |
| separation ratio | 10.16x | 14.37x |
| distinct overlap values (native) | 9 | 36 |
| tie-block mean/median/max @ R=1000 | 2496/2160/14143 | 484/384/2706 |

No collapse in either Arm B variant at full training length -- dead bits stay
a tiny fraction of `d`, binarity near 1, no saturated bits.

`d=512` realized κ (24.36±5.57) vs. the two §0.3 predictive laws: free-code
`1.8·√(eps·d) ≈ 40.7`; trained-encoder `0.85·eps^0.302·d^0.660 ≈ 52.1`.
Realized κ is ~60% of the free-code prediction and ~47% of the trained-encoder
prediction -- right order of magnitude (tens, not single digits, not
hundreds) but noticeably below both point predictions. Caveat: neither law
was fit on a frozen-backbone encoder, so this is a stress-test of laws
derived elsewhere, not a validation of them.

### 12.3 Test 1 — matched-κ (`topk_eval=8`) comparison, `d=64` vs. `d=512`

The §12.2 `d=512` win is confounded: realized κ differs ~5x between the two
(4.70 vs 24.36), so dimensionality and active-bit count were never isolated.
Using `topk_eval` (`utils/hashing.py`'s `preprocess_on_codes`, exact per-sample
κ override at eval time, no retraining) on the existing best checkpoints from
§12.2:

| | native | `topk_eval=8` |
|---|---|---|
| `d=64` (`eps=0.31`) | 0.6200 | 0.6165 |
| `d=512` (`eps=1.0`) | 0.6580 | 0.6281 |

`d=512` still wins at matched κ=8 (Δ=0.0116), but the gap shrinks from
Δ=0.0380 (native) to Δ=0.0116 -- roughly 70% of the native advantage was the
active-bit-count confound; a smaller genuine dimensionality effect (~1.2 mAP
points) survives matching. At κ=8, both are capped at κ+1=9 possible overlap
values and both realize all 9 (`d=64`: 9 distinct, 7 over 1% mass; `d=512`: 9
distinct, 4 over 1% mass, 89.5% of pairs at overlap=0 vs. 22.3% for `d=64`) --
so the native-run "36 vs. 9 distinct overlap values" gap (§12.2) was mostly an
artifact of native κ, not intrinsic dimensionality. Tie-block size at κ=8
still favors `d=512` (mean 1112 vs. 1741, ~36% smaller), consistent with the
small residual dimensionality effect seen in mAP.

**Implication:** the `d=512` advantage over `d=64` (§12.2) is mostly, but not
entirely, an active-bit-count effect. A real but much smaller dimensionality
effect remains after matching κ.

### 12.4 Test 2 — SDC, frozen backbone, same protocol as Arm A

`models/arch/sdc.py` / `models/loss/sdc.py`, `nbit=64`, `backbone_lr_scale=0`,
100 epochs, seed 42, eval every 10 epochs, GPU 3, same protocol as §12.2's
Arm A run. Note: `configs/model/sdc.yaml` already sets `backbone_lr_scale: 0`
as its own default (unlike `cibhash.yaml`/`sbdr.yaml`, which inherit
`train.yaml`'s default of 1) -- so this run's frozen-backbone setting is
actually this repo's existing default for SDC, not a new override. This
contradicts the premise that this repo's 66.3 reference number came from a
trainable-backbone run under this repo's own config; flagged here as an
open discrepancy, not resolved by this test.

| epoch | mAP |
|---|---|
| 10 | 0.5907 |
| 20 | 0.5916 |
| 30 | 0.5867 |
| 40 | **0.5976 (best)** |
| 50 | 0.5959 |
| 60 | 0.5847 |
| 70 | 0.5877 |
| 80 | 0.5876 |
| 90 | 0.5930 |
| 100 (final) | 0.5917 |

Runtime: 0.48h.

- vs. this repo's own published trainable-backbone SDC reference (66.3 / 0.663):
  best frozen mAP 0.5976 is **0.0654 lower** (~9.9% relative) -- a substantial gap.
- vs. frozen-backbone Arm A / CIBHash (0.6225, §12.2): best frozen SDC mAP
  0.5976 is **0.0249 lower** -- SDC also loses to CIBHash specifically under
  the frozen protocol, whereas CIBHash frozen (0.6225) met/slightly exceeded
  its own trainable-backbone reference (0.6167/0.612, §6).

**Implication:** "frozen backbone doesn't hurt" does not generalize from
CIBHash to SDC. SDC loses ground when frozen, both against its own
trainable-backbone reference and against frozen CIBHash -- frozen backbone is
not a neutral protocol choice across methods for this paper's comparisons.

GPU discipline: only GPU 3 used for every run and diagnostic pass in §12
(GPU 2 occupied by another user's unrelated job the entire time, never
touched); confirmed idle via `nvidia-smi` before each launch and after the
last run. `experiments/sbdr_report.py`'s hardcoded combined-output write to
`experiments/sbdr_sweep_report.json` was triggered twice more in this section
(once per Arm B variant) and reverted via `git checkout` immediately after
each call, as in §11.4.

### 12.5 Correction to §12.4's framing, and a config audit against the paper (2026-09-04)

§12.4 framed the SDC gap as a frozen-vs-trainable-backbone question. That
premise was wrong: the paper's own supplementary material (`docs/suppmat.pdf`,
Sec. A, read directly for this section rather than assumed) states plainly
that **all** compared methods, including SDC, use a **frozen** pretrained
VGG16, Adam lr=0.0001, batch 64, 100 epochs, with lr dropped to 0.00001 after
epoch 80 -- this repo's frozen protocol was already correct, and 66.3 is
itself a frozen-backbone number. The real question is a reproduction gap
under matching protocols, not a frozen/trainable choice. §12.4's framing is
superseded by this section.

**Step 1 -- config audit against the paper (`configs/model/sdc.yaml`, `models/loss/sdc.py`, `docs/suppmat.pdf` Sec. A/B/C):**

| item | paper | this repo (`sdc.yaml`, as run in §12.4) | match? |
|---|---|---|---|
| λq (`quan`) | 1 | 1 | yes |
| λcl (`cont` weight) | 1 (only when Lcl is used) | 1 (default), but `contrastive` (the module implementing Lcl) is `None` -- `models/loss/sdc.py`'s `forward` only adds the Lcl term `if self.contrastive is not None`, so Lcl is **structurally absent**, not just default-weighted | **no** -- see below |
| calibration Beta(α,β) (`beta_ab`) | α=β=5 (best row, Table 1) | 5 | yes |
| backbone / frozen | VGG16, frozen | VGG16, frozen (`backbone_lr_scale=0` is `sdc.yaml`'s own default) | yes |
| Adam lr | 0.0001 | 0.0001 | yes |
| batch size | 64 | 64 | yes |
| epochs / lr schedule | 100 epochs, lr/10 after epoch 80 | 100 epochs; `configs/scheduler/step.yaml`: `step_size=int(0.8*100)=80`, `gamma=0.1` -> lr/10 at epoch 80 | yes |
| **weight decay** | **0.0005** | `configs/optim/adam.yaml`: **0.00001** | **no -- 50x mismatch, shared by every arm's runs so far, not SDC-specific** |
| `rec_type` / quantization form | L1 on `(s-C)`; quan = `1-cossim(f,f.sign())` (Algorithm 1) | `rec_type="l1"`, `quan_type="cs"` | yes |
| orthogonality constraint on `C` | none in Algorithm 1 (no relu/clamp) | `ortho_constraint=False` (matches; `sdc_simclr.yaml`'s `True` is the outlier, not paper-matching) | yes |

Two real findings, of different character:

1. **weight_decay mismatch (simple, correctable):** 0.00001 vs. the paper's
  0.0005, a plain optimizer-setting error affecting every arm run so far
  (Arm A, Arm B x2, and SDC), not something specific to SDC. Fixed for the
  Step 3 rerun below (SDC only, per the task's scope).
2. **The 66.3 target itself needs correcting, not just the config:** the
  paper's own ablation (suppmat Table 2) reports 66.3 only **with** Lcl
  included; the same Beta(5,5) calibration **without** Lcl scores **63.0**
  (matches Table 1's Beta(5,5) row exactly, which is explicitly run at
  `Lcl=0`). `configs/model/sdc.yaml` never wires up `contrastive`, so every
  SDC run in this repo so far (§12.4 and this section) structurally
  corresponds to the paper's "wo/ Lcl" ablation cell -- **63.0 is the correct
  comparison point for this config, not 66.3.** Properly enabling Lcl per the
  paper's own Algorithm 1 is not a config fix: Algorithm 1 needs a *second,
  independently-augmented* view pair (`x2`) distinct from the pair used for
  the main SDC loss, and applies SimCLR to **raw pre-hash features**, not
  hash codes. This repo's `models/arch/sdc.py.forward` never returns a third
  ("`cont_feats`") output, and `trainers/sdc.py`'s trainer only ever
  constructs one augmented view pair per step -- both would need real code
  changes, not a config change, so this is flagged as an open gap rather than
  fixed here (out of scope for a one-shot corrective rerun).

**Step 2 -- dataset/eval protocol audit:** `docs/suppmat.pdf` Sec. A.1
states, for CIFAR-10: "100 images from each class as queries (1K total)...
remaining 59K as database... 5K images sampled from the database as training
images" -- this matches this repo's `configs/dataset/cifar10.yaml` split
exactly (test=1000, db=59000, train=5000, confirmed by direct tensor load in
§0's initial audit, not just config values). `R` and `mAP@1000` also already
match (§6/§12 always used `dataset.R=1000`). No mismatch found in Step 2.

**Step 3 -- corrective rerun (weight_decay fix only):** same as §12.4
(`model=sdc`, `nbit=64`, `backbone_lr_scale=0`, 100 epochs, seed 42, GPU 3),
`optim.weight_decay=0.0005` instead of the shared default `0.00001`.

| epoch | mAP |
|---|---|
| 10 | 0.5929 |
| 20 | 0.5941 |
| 30 | 0.5819 |
| 40 | 0.5957 |
| 50 | 0.5926 |
| 60 | 0.5948 |
| 70 | 0.5878 |
| 80 | 0.6040 |
| 90 | **0.6053 (best)** |
| 100 (final) | 0.6042 |

Runtime: 0.47h.

- vs. §12.4's original run (wd=0.00001): best 0.6053 vs. 0.5976 (**+0.0077**),
  final 0.6042 vs. 0.5917 (**+0.0125**) -- a small, consistent improvement in
  the expected direction.
- vs. **63.0** (the correct reference for this Lcl-disabled config): best
  0.6053 is **0.0247 lower** (~3.9% relative) -- narrower than the naive
  comparison against 66.3 (§12.4: -0.0654) but still a real, unexplained gap.
- vs. 66.3 (not the correct target for this config, reported per the original
  ask anyway): **0.0577 lower**.

**Bottom line:** the weight_decay fix helps modestly but does not close the
gap. After correcting both the protocol-framing error (frozen was always
right) and the target-recalibration error (63.0, not 66.3, is what this
Lcl-disabled config should be compared against), a ~2.5-point unexplained gap
remains. Per the task's stopping condition, this is now flagged as an open
problem rather than guessed at further: candidates not ruled out include a
subtler implementation difference in `models/loss/sdc.py` vs. Algorithm 1, a
data-pipeline difference not visible in the config, or a metric-computation
difference -- none investigated further here.

GPU discipline: GPU 3 only, one run at a time, confirmed idle via `nvidia-smi`
before the rerun and after it completed; GPU 2 remained occupied by another
user's unrelated job throughout.

### 12.6 Provenance of §12.5's two discrepancies, and the `sdc_simclr` run that closes the gap (2026-09-04)

**Provenance (pure git archaeology, no training).** This repo's history is
not shallow and contains `f429ca9 "Merge branch 'master' of
https://github.com/kamwoh/sdc"` -- the exact URL the paper's README links to
-- so upstream history is directly available, not just inferable. This
fork's own commits (`d0d2e03` onward) begin strictly after the last upstream
commit and never touch either file below.

- `configs/optim/adam.yaml`'s `weight_decay=0.00001`: `git log --follow`
  shows exactly one commit ever touches this file -- `1e3410e "init commit"`
  (kam woh, 2023-03-10), which introduced it with this value already set. No
  commit since, upstream or fork, has changed it. **Traced to the original
  authors' repo unchanged since its first commit -- an upstream default, not
  fork-introduced.** The paper's stated 0.0005 has apparently never matched
  the authors' own shipped code.
- `configs/model/sdc.yaml` / `models/loss/sdc.py`'s `contrastive` handling:
  `git log -S"contrastive"` shows the string enters `models/loss/sdc.py` in
  exactly one commit, `0174b37 "updated for official release"` (kam woh,
  2023-08-31), and the diff shows this is the mechanism's *introduction* (the
  `cont`/`contrastive` args, `contrastive_loss()`, and the `if self.contrastive
  is not None` gate did not exist before), not a removal of prior wiring.
  The same commit simultaneously created `configs/model/sdc_simclr.yaml`
  from scratch, which does set `contrastive`. **Traced to the original
  authors' repo, unchanged since the commit that added the feature -- an
  upstream design choice (ship Lcl opt-in via a separate config, default
  `sdc.yaml` off), not fork-introduced and not a wired-up-then-removed
  feature.**

**`sdc_simclr` run.** Before running, confirmed `configs/model/sdc_simclr.yaml`
against the `sdc.yaml` fields §12.5 already checked: `rec=1`, `rec_type="l1"`,
`quan=1` (λq), `quan_type="cs"`, `beta_ab=5` all match. Two differences beyond
`contrastive`/`cont=1`, flagged before running rather than assumed away:
`ortho_constraint=True` (vs. `sdc.yaml`'s paper-matching `False`, §12.5) and
`/transforms: cibhash` (real augmentation, vs. `sdc.yaml`'s `no_augmentation`
-- expected, since Lcl needs two distinct augmented views, but a real
pipeline difference nonetheless, not an isolated change).

Ran `model=sdc_simclr`, `nbit=64`, `backbone_lr_scale=0`, `optim.weight_decay
=0.0005` (paper value), 100 epochs, seed 42, eval every 10 epochs, GPU 3.

| epoch | mAP |
|---|---|
| 10 | 0.6150 |
| 20 | 0.6246 |
| 30 | 0.6447 |
| 40 | 0.6404 |
| 50 | 0.6523 |
| 60 | 0.6570 |
| 70 | 0.6488 |
| 80 | 0.6555 |
| 90 | 0.6616 |
| 100 (final, best) | **0.6646** |

Runtime: 0.71h. Curve is still rising at epoch 100, unlike the plain
`sdc.yaml` runs (§12.4/12.5), which peaked mid-schedule and drifted down.

- vs. **66.3** (paper's Lcl-enabled number): 0.6646 is **+0.0016 higher** --
  matches, marginally exceeds it (well within single-seed noise).
- vs. **63.0** (paper's w/o-Lcl number): +0.0346 higher, as expected with Lcl active.
- vs. **§12.5's 0.6053** (plain `sdc.yaml` + weight_decay fix, Lcl disabled): +0.0593 higher.

**This closes the gap.** The reproduction gap opened in §12.4 is now fully
explained across §12.5-12.6: not a frozen-vs-trainable-backbone issue
(§12.5, closed -- frozen was always correct), not primarily the weight_decay
mismatch (§12.5, small effect: +0.008-0.013 mAP), and not a deeper
implementation bug -- the default `sdc.yaml` config correctly reproduces the
paper's own lower ("wo/ Lcl") ablation cell, and the paper's headline number
requires the separately-shipped `sdc_simclr.yaml` config, which §12.4/§12.5
simply hadn't tested yet.

GPU discipline: GPU 3 only, confirmed idle via `nvidia-smi` before launch and
after completion; GPU 2 occupied by another user's unrelated job throughout.

## 13. Weight-decay correctness pass, activation comparison, higher-d baselines, and a quantization-loss arm (2026-09-05)

All runs: frozen VGG16 backbone, 100 epochs, seed 42, eval every 10 epochs,
`optim.weight_decay=0.0005` (§12.5's paper-matching value) unless noted.
GPU discipline for this whole section: only GPU 1 and GPU 3 used (by user
permission, up to 2 GPUs at once); GPU 0 and GPU 2 remained occupied by other
users' unrelated jobs throughout and were never touched. 7 training jobs run,
mostly as 3 parallel pairs (one per GPU) plus 2 solo runs; 3
`sbdr_report.py` diagnostic passes, each followed by `git checkout --
experiments/sbdr_sweep_report.json` to revert that script's shared-file
side effect, as in §11.4/§12.

### 13.1 Task 1 -- weight-decay correctness pass, Arm A and Arm B (`d=64`)

| epoch | Arm A (wd=5e-4) | Arm B `d=64,eps=0.31` (wd=5e-4) |
|---|---|---|
| 10 | 0.5832 | 0.5850 |
| 20 | 0.5887 | 0.5967 |
| 30 | 0.5933 | 0.6024 |
| 40 | 0.5993 | 0.6140 |
| 50 | 0.5969 | 0.6078 |
| 60 | 0.6025 | 0.6092 |
| 70 | 0.6068 | 0.6184 |
| 80 | **0.6105 (best)** | 0.6108 |
| 90 | 0.6089 | 0.6140 |
| 100 (final) | 0.6079 | **0.6202 (best)** |

| | Arm A | Arm B `d=64` |
|---|---|---|
| §12.2 (wd=1e-5) best/final | 0.6225/0.6221 | 0.6200/0.6158 |
| this section (wd=5e-4) best/final | 0.6105/0.6079 | 0.6202/0.6202 |
| Δ best | **-0.0120** | +0.0002 |
| Δ final | **-0.0142** | +0.0044 |

Arm B diagnostics (best ckpt, ep100): κ=4.74±1.24, dead bits=1/64,
binarity=0.959, separation ratio=10.59x -- all consistent with §12.2's
values, no qualitative change.

**Three different directions across three methods.** SDC's wd fix helped
(+0.008/+0.013, §12.5), Arm B is essentially unaffected (+0.0002 best, within
noise), and Arm A/CIBHash gets clearly *worse* (-0.0120 best, -0.0142 final)
-- the paper-matching weight decay is not a universal improvement; its effect
is method-specific and in Arm A's case actively harmful, at least at this
frozen-backbone, 100-epoch, single-seed setting.

### 13.2 Task 2 -- activation comparison, `clip` vs `sigmoid` (Arm B, wd-fixed)

| epoch | `act=clip` (13.1) | `act=sigmoid` |
|---|---|---|
| 10 | 0.5850 | 0.4814 |
| 20 | 0.5967 | 0.5346 |
| 30 | 0.6024 | 0.5680 |
| 40 | 0.6140 | 0.5663 |
| 50 | 0.6078 | 0.5715 |
| 60 | 0.6092 | 0.5710 |
| 70 | 0.6184 | 0.5630 |
| 80 | 0.6108 | 0.5805 |
| 90 | 0.6140 | **0.5821 (best)** |
| 100 (final) | **0.6202 (best)** | 0.5740 |

| | clip | sigmoid |
|---|---|---|
| best / final mAP | 0.6202 / 0.6202 | 0.5821 / 0.5740 |
| κ mean±std | 4.74±1.24 | 4.62±1.21 |
| dead bits | 1/64 | 12/64 |
| binarity | 0.959 | 0.911 |
| separation ratio | 10.59x | 8.36x |

`clip` clearly outperforms `sigmoid` on every metric (mAP, dead bits,
binarity, separation) once collapse is not a confound (both frozen-backbone)
-- consistent with §11's established mechanism (sigmoid's near-zero-init
gradient is ~266x smaller than clip's), now confirmed to also produce a
weaker *converged* model, not just slower/failed training.

### 13.3 Task 3 -- CIBHash and SDC at `d=512`

| epoch | CIBHash `d=512` | SDC_simclr `d=512` | Arm B `d=512,eps=1.0` (wd-fixed) |
|---|---|---|---|
| 10 | 0.5923 | 0.6487 | 0.6308 |
| 20 | 0.5978 | 0.6591 | 0.6410 |
| 30 | 0.5981 | 0.6658 | 0.6457 |
| 40 | **0.6066 (best)** | 0.6685 | 0.6477 |
| 50 | 0.6024 | 0.6676 | 0.6453 |
| 60 | 0.5972 | 0.6810 | 0.6433 |
| 70 | 0.6024 | 0.6763 | 0.6526 |
| 80 | 0.6021 | 0.6767 | **0.6543 (best)** |
| 90 | 0.6052 | 0.6875 | 0.6505 |
| 100 (final) | 0.6041 | **0.6880 (best)** | 0.6519 |

`d=64` -> `d=512` deltas (best mAP, same wd=5e-4 protocol throughout):

| | `d=64` | `d=512` | Δ |
|---|---|---|---|
| CIBHash | 0.6105 (13.1) | 0.6066 | **-0.0039** |
| SDC_simclr | 0.6646 (§12.6) | 0.6880 | **+0.0234** |
| Arm B | 0.6202 (13.1) | 0.6543 | **+0.0341** |

Arm B `d=512` diagnostics (wd-fixed, best ckpt ep80): κ=25.19±5.76, dead
bits=6/512, binarity=0.960, separation ratio=13.94x, 38 distinct overlap
values, tie-block mean=523.8 -- all close to §12.2/§12.3's pre-wd-fix values
(κ=24.36±5.57, dead=6/512, sep=14.37x, 36 distinct, tie=483.7; mAP
0.6580->0.6543, Δ=-0.0037, same small-negative direction as Arm A's wd
effect but far smaller magnitude). The outstanding item flagged in §12.3
(whether Arm B `d=512` needs a wd-corrected rerun) is now resolved: it does
not materially change any conclusion.

**Does the `d=512` advantage survive giving the baselines the same
dimensionality? Partially, and the picture is more interesting than a yes/no.**
CIBHash does **not** benefit from `d=512` at all (-0.0039, marginally worse)
-- so the original §12.2/§12.3 comparison was not simply "unfair
dimensionality." But SDC_simclr **does** benefit substantially (+0.0234), and
by `d=512` it actually **overtakes Arm B**: SDC_simclr 0.6880 > Arm B 0.6543
> CIBHash 0.6066. The `d=512`-helps-more-than-`d=64` effect is real but not
specific to Arm B/our method -- it generalizes to at least one baseline
(SDC_simclr) strongly enough to flip the ranking at that dimensionality.

### 13.4 Task 4 -- Arm B + SDC-style quantization loss

Implementation (`models/arch/sbdr.py`, `models/loss/sbdr.py`,
`trainers/sbdr.py`, `configs/model/sbdr.yaml`): `SBDR.forward`'s previously-
unused third return slot (always discarded as `_` by every caller, checked
via grep before changing) now returns the pre-activation logits instead of a
duplicate of `z`. `SBDRCriticLoss` gained an optional `lambda_q` (default
0.0, bit-identical to before when off -- confirmed via `tests/
test_sbdr_loss_math.py`, all 6 checks still pass unchanged) that adds
`lambda_q * Lq` where `Lq = models.loss.sdc.SDCLoss.quantization_loss`
(`quan_type='cs'`, reused directly rather than reimplemented) applied to the
concatenated pre-activation logits of both views.

**Choosing `lambda_q` (one real batch, at init, frozen backbone, gradient
norm w.r.t. `encoder[-1]` weight+bias):**

| quantity | value |
|---|---|
| `L_eps` (plain critic) | -0.0623 |
| grad norm of `L_eps` | 0.7322 |
| `Lq` (SDC quantization) | 0.1839 |
| grad norm of `Lq` | 1.1196 |
| ratio (`Lq` grad / `L_eps` grad) | 1.529 |

Target: `Lq`'s contribution at 10-30% of `L_eps`'s initial gradient norm.
Chose the middle of that range (20%): `lambda_q = 0.20 * 0.7322 / 1.1196 =
0.131`, rounded to **`lambda_q = 0.13`** (realizes ≈19.9%, inside the target
band).

**Training result** (`d=64`, `eps=0.31`, wd=5e-4, otherwise identical to §13.1's Arm B):

| epoch | plain Arm B (13.1) | Arm B + Lq (`lambda_q=0.13`) |
|---|---|---|
| 10 | 0.5850 | 0.5761 |
| 20 | 0.5967 | 0.5921 |
| 30 | 0.6024 | 0.6037 |
| 40 | 0.6140 | 0.6099 |
| 50 | 0.6078 | 0.6145 |
| 60 | 0.6092 | 0.6104 |
| 70 | 0.6184 | 0.6082 |
| 80 | 0.6108 | 0.6135 |
| 90 | 0.6140 | **0.6227 (best)** |
| 100 (final) | **0.6202 (best)** | 0.6223 |

| | plain Arm B | Arm B + Lq |
|---|---|---|
| best / final mAP | 0.6202 / 0.6202 | **0.6227** / 0.6223 |
| κ mean±std | 4.74±1.24 | 4.46±1.12 |
| dead bits | 1/64 | 8/64 |
| binarity | 0.959 | 0.962 |
| separation ratio | 10.59x | 9.62x |

**Mixed result, not a clean win.** mAP improves slightly (+0.0025 best,
+0.0021 final), but dead bits rise (1->8/64) and separation ratio drops
(10.59x->9.62x) -- the quantization term nudges a few units toward permanent
saturation (consistent with `clip`'s zero-gradient-at-saturation mechanism,
§11) in exchange for a marginal mAP gain. At `lambda_q=0.13` this is a wash
at best, not evidence the quantization term is worth adopting outright.
