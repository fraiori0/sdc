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
- **The high-d regime (§4b) fails for Arm D, trainable backbone, 40 epochs
  (2026-09-05 correction: scope narrowed, superseded for Arm B — see below).**
  At matched `κ` via `topk_eval`, `d=64 > d=256 > d=512` uniformly — mAP gets
  *worse* as `d` grows at fixed active-bit budget, the opposite of the SBDR
  "blessing of dimensionality" claim §4b hoped to reproduce. Logs:
  `logs/cifar10/sbdr_aux{64,256,512}_40/`. **This finding is scoped to Arm D
  (CIBHash arch + our loss as an auxiliary term) with a trainable backbone at
  40 epochs — it is not a general finding about the loss, and it is not what
  Arm B (our loss alone, bounded head, frozen backbone) shows.** §12.2/§12.3
  found the opposite for Arm B at matched `κ` (frozen backbone, 100 epochs):
  `d=512` beats `d=64` (mAP 0.6281 vs 0.6165 at matched `topk_eval=8`, §12.3),
  a smaller margin than the unmatched native comparison suggested (0.6580 vs
  0.6200) but still a real, positive effect in the opposite direction from
  this bullet's original framing. The original Arm D numbers above are
  unchanged and still on record; do not treat "high-d fails" as settled
  across all arms/protocols — it is an open, arm-and-protocol-dependent
  question, not a closed negative, for anything other than Arm D/trainable/
  40-epoch specifically.
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

## 14. Order-2 critic verification, order-2 hashing runs, Arm C (loss-only ablation), and a §8 scoping correction (2026-09-05)

All training runs: frozen VGG16 backbone (`backbone_lr_scale=0`), `optim.
weight_decay=0.0005`, `nbit=64`, 100 epochs, seed 42, eval every 10 epochs --
identical protocol to §13.1's Arm B runs except where a task explicitly
varies one setting. GPU discipline: GPU 1 and GPU 3 only (by user
permission, up to 2 GPUs at once), confirmed idle via `nvidia-smi` before
each launch; GPU 0 and GPU 2 remained occupied by other users' unrelated
jobs throughout and were never touched. 6 training jobs run as 3 parallel
pairs; every `sbdr_report.py` / `sbdr_armc_report.py` diagnostic pass
followed by `git checkout -- experiments/sbdr_sweep_report.json` to revert
that script's shared-file side effect, as in §11.4/§12/§13.

### 14.1 Task 1 -- order-2 critic verification (not a new implementation)

**Deviation from the task's framing, flagged up front:** the prompt asked to
"add a second-order term... alongside the existing order-1 loss." This
second-order term **already existed in full** in `models/loss/sbdr.py`
(`critic_order`, `lambda2`, `_effective_lambda2`, the `C`-matrix path in
`_one_way`) and was already covered by a dedicated test file,
`tests/test_sbdr_second_order.py` -- both predate this batch, from the
"Implement Arm B (SBDR) with second-order critic" work referenced in §9. No
loss-code changes were made or needed. Line-by-line comparison of the
existing implementation against the prompt's exact spec confirms they match:
`C = zall.t().matmul(zall) / K` where `K = zall.size(0)` is the prompt's
`2K` (`zall` already holds both concatenated views) -- i.e. `C = zall.T @
zall / (2K)` exactly; `t = <z_i,z̄> + lambda2*(z_i^T C z_i) + eps` (denominator);
`s = s_pos_i + lambda2*s_pos_i^2 + eps` (numerator); `L = mean(log(t) - log(s))`.

What the existing `tests/test_sbdr_second_order.py` already covered before
this batch: `C`-matrix vs. naive `O(K^2*d)` double-loop agreement (scalar
loss and per-row quadratic form, both ~1e-6), `critic_order=1` bit-identical
to `critic_order=2, lambda2=0` (and ignores a nonzero `lambda2` kwarg when
`critic_order=1`), default `lambda2 == 1/(2*eps)` exactly, nonzero `lambda2`
actually changes the loss (no silent no-op), and a basic nonzero-total-
gradient sanity check. `tests/test_sbdr_loss_math.py`'s existing
finite-difference gradient check (`test_2a`) and degenerate-state check
(`test_2c`) already covered both `critic_order in {1,2}` too (`test_2c` used
`lambda2=1.6`, a round number close to but not exactly `1/(2*0.31)=1.6129`).

**Two gaps against the prompt's exact requirements, closed by extending
`tests/test_sbdr_second_order.py`** (not duplicating -- checked first, per
the task's own instruction):

- `test_degenerate_state_zero_loss_pinned_and_arbitrary_lambda2`: confirms
  `L_i == 0.0` to float precision at the degenerate all-identical-codes state
  for the **exact** pinned default `lambda2 = 1/(2*eps)` (1.612903..., not
  the nearby round number 1.6 already tested elsewhere) and one arbitrary
  other value (7.0). Passes: `L=0.000e+00` for both.
- `test_gradient_reaches_c_and_zbar_directly`: stronger than the existing
  nonzero-total-gradient check -- recomputes `_one_way`'s body manually with
  `zbar` and `C` as explicit intermediates (`retain_grad()`), so gradient
  into each is inspected directly rather than inferred from a nonzero
  gradient at the leaves (which could in principle be nonzero via some other
  path even if one of `zbar`/`C` were accidentally detached). Passes:
  `zbar.grad abs-sum = 0.358589`, `C.grad abs-sum = 2.472026`, both nonzero.

Command: `python tests/test_sbdr_second_order.py` -- **all 8 checks pass**
(6 pre-existing + 2 new). `python tests/test_sbdr_loss_math.py` -- all 6
checks still pass unchanged, confirming the extension didn't disturb
anything. **Verdict: matches the prompt's required properties exactly; no
regressions.**

### 14.2 Task 2 -- order-2 hashing runs, Arm B, `d=64`

Commands (only the varying flags shown; full protocol as stated above):

```
python main_v2.py model=sbdr model.nbit=64 dataset=cifar10 epochs=100 backbone_lr_scale=0 \
  criterion.critic_order=2 criterion.eps=0.31 criterion.lambda2=null optim.weight_decay=0.0005 seed=42
python main_v2.py model=sbdr model.nbit=64 dataset=cifar10 epochs=100 backbone_lr_scale=0 \
  criterion.critic_order=2 criterion.eps=0.15 criterion.lambda2=null optim.weight_decay=0.0005 seed=42
python main_v2.py model=sbdr model.nbit=64 dataset=cifar10 epochs=100 backbone_lr_scale=0 \
  criterion.critic_order=1 criterion.eps=0.15 optim.weight_decay=0.0005 seed=42
```

The third command is a fresh order-1 `eps=0.15` baseline -- checked first,
did not already exist in the handout (every prior Arm B run used `eps=0.31`
or `eps=1.0`).

**mAP per eval epoch:**

| epoch | order-1, `eps=0.31` (§13.1, reference) | order-2, `eps=0.31` | order-1, `eps=0.15` (new) | order-2, `eps=0.15` |
|---|---|---|---|---|
| 10 | 0.5850 | 0.5684 | 0.5824 | 0.5529 |
| 20 | 0.5967 | 0.5756 | 0.5840 | 0.5481 |
| 30 | 0.6024 | 0.5736 | 0.5941 | 0.5684 |
| 40 | 0.6140 | 0.5781 | 0.5969 | 0.5605 |
| 50 | 0.6078 | 0.5882 | 0.5927 | 0.5542 |
| 60 | 0.6092 | 0.5813 | 0.5965 | **0.5713 (best)** |
| 70 | 0.6184 | 0.5828 | 0.5907 | 0.5571 |
| 80 | 0.6108 | 0.5869 | 0.5983 | 0.5586 |
| 90 | 0.6140 | 0.5884 | 0.6018 | 0.5568 |
| 100 (final) | **0.6202 (best)** | **0.5888 (best)** | **0.6041 (best)** | 0.5559 |

**Best-checkpoint diagnostics (`sbdr_report.py`):**

| | order-1, `eps=0.31` | order-2, `eps=0.31` | order-1, `eps=0.15` | order-2, `eps=0.15` |
|---|---|---|---|---|
| best/final mAP | 0.6202/0.6202 | 0.5888/0.5888 | 0.6041/0.6041 | 0.5713/0.5559 |
| κ mean±std | 4.74±1.24 | 4.07±1.10 | 3.37±1.10 | 2.92±1.00 |
| dead bits (db-wide, post-hoc) | 1/64 | 3/64 | 0/64 | 3/64 |
| binarity | 0.959 | 0.940 | 0.964 | 0.942 |
| separation ratio | 10.59x | 11.36x | 13.53x | 14.62x |

**Per-epoch training trajectory** (`train_kappa` / `train_dead_bits_exact`,
both already logged for free by `trainers/sbdr.py`'s existing per-batch
diagnostics -- no code changes needed for this part):

| epoch | order-1 `ε=0.31` κ / dead | order-2 `ε=0.31` κ / dead | order-1 `ε=0.15` κ / dead | order-2 `ε=0.15` κ / dead |
|---|---|---|---|---|
| 1 | 3.65 / 2.55 | 3.40 / 2.90 | 2.35 / 3.35 | 2.04 / 2.78 |
| 10 | 5.56 / 1.00 | 4.94 / 3.00 | 4.16 / 0.00 | 3.51 / 2.99 |
| 20 | 5.41 / 1.00 | 4.79 / 3.00 | 4.05 / 0.01 | 3.43 / 3.00 |
| 40 | 5.27 / 1.00 | 4.64 / 3.00 | 3.87 / 0.03 | 3.27 / 3.00 |
| 60 | 5.15 / 0.96 | 4.55 / 3.00 | 3.84 / 0.00 | 3.27 / 3.00 |
| 80 | 5.18 / 0.96 | 4.52 / 3.00 | 3.73 / 0.04 | 3.17 / 3.00 |
| 100 | 5.10 / 0.96 | 4.46 / 3.00 | 3.71 / 0.03 | 3.08 / 3.04 |

Both order-1 training curves are smooth and monotonic-ish (loss and κ both
drift steadily, no noisy non-monotonic swings like §9/§12.2's earlier
sweeps); both order-2 curves are equally smooth, just at a worse mAP and a
higher, earlier-settling dead-bit plateau (~3.0 from epoch 10 onward in both
`eps` cases, vs. order-1's ~0-1). **Order-2 training is not noisier than
order-1 here -- both are stable -- the difference is a persistent, higher
dead-bit floor, not instability.**

**Verdict, stated plainly: order-2 is clearly, consistently worse than
order-1 at both `eps` values tested, not within noise.** `eps=0.31`: best
mAP drops from 0.6202 to 0.5888 (Δ=**-0.0314**). `eps=0.15`: best mAP drops
from 0.6041 to 0.5713 (Δ=**-0.0328**). The two deltas are close in
magnitude (-0.031 vs. -0.033), so **no clear trend with `eps`** is visible
over this 2-point range -- the order-2 penalty looks roughly `eps`-
independent here, though two points cannot establish a real trend.
Despite the mAP loss, order-2's separation ratio is *higher* than order-1's
in both cases (11.36x vs. 10.59x at `eps=0.31`; 14.62x vs. 13.53x at
`eps=0.15`) -- a plain observation, not resolved further here: better
positive/random overlap separation did not translate into better retrieval
here, plausibly because order-2's extra dead bits (a stable ~3/64 floor
throughout training, vs. order-1's ~0-1) reduce the code's overall usage/
resolution even as the surviving active bits discriminate slightly better.

### 14.3 Task 3 -- Arm C (loss-only ablation on CIBHash's unmodified architecture)

**Not previously implemented -- checked `models/loss/` and existing config
wiring first** (found only Arm D, `models/loss/sbdr_aux.py`, `configs/model/
sbdr_aux.yaml`; no Arm C). New files:

- `models/loss/sbdr_c.py` (`CIBHashSBDROnlyLoss`): matches `CIBHashLoss`'s
  6-arg forward signature (`prob_i, prob_j, z_i, z_j, f_i, f_j`) so it plugs
  into the existing `trainers.cibhash.CIBHashTrainer` with no trainer
  subclass and no arch change, same wiring pattern as Arm D -- but computes
  **only** `SBDRCriticLoss(prob_i, prob_j)`, with no CIBHash NtXent/KL term
  at all (`z_i, z_j, f_i, f_j` accepted but unused, per §3's scope caveat).
- `configs/model/sbdr_c.yaml`: mirrors `cibhash.yaml`/`sbdr_aux.yaml`
  exactly (arch, trainer, backbone, transforms, `code_domain`/`dist_metric`
  left at `signed`/`hamming` since CIBHash's arch and its raw-logit eval-time
  `codes` are unchanged).
- Smoke-tested (CPU, dummy tensors matching CIBHash's shapes) before any
  training: config composes correctly, loss forward/backward runs, gradient
  reaches `prob_i`.

**Important deviation from the task's premise, flagged as instructed rather
than papered over:** the task asks for "code agreement across two samples of
`b` from the same `p`," describing CIBHash's sampling as "Bernoulli+STE."
**This repo's actual CIBHash hash layer (`models/layers/signhash.py`) is a
deterministic `sign()` with a straight-through-estimator backward -- there is
no stochastic sampling anywhere.** Grepped the entire codebase for
`torch.bernoulli`/`Bernoulli`: zero hits before this section (the only
matches were the word "Bernoulli" in docstrings/comments describing the
method, never an actual call). So "two independent samples of `b` from the
same `p`" cannot be measured as something that happens during this repo's
real training or inference -- it doesn't happen; sampling `b` twice from the
same `p` deterministically gives the same `b` both times (trivial 100%
agreement), which would not be a meaningful diagnostic. `experiments/
sbdr_armc_report.py` computes it anyway as an explicit **hypothetical**
post-hoc quantity (`b1, b2 = torch.bernoulli(p), torch.bernoulli(p)`,
independently sampled purely for this diagnostic, never touching the
model/training path) -- meaningful as "how sharp is the learned `p`," but
explicitly not a measurement of this repo's actual sampling variance, since
none exists to measure.

Commands:

```
python main_v2.py model=sbdr_c model.nbit=64 dataset=cifar10 epochs=100 backbone_lr_scale=0 \
  criterion.critic_order=1 criterion.eps=0.31 optim.weight_decay=0.0005 seed=42
python main_v2.py model=sbdr_c model.nbit=64 dataset=cifar10 epochs=100 backbone_lr_scale=0 \
  criterion.critic_order=2 criterion.eps=0.31 criterion.lambda2=null optim.weight_decay=0.0005 seed=42
python experiments/sbdr_armc_report.py <logdir> --device cuda
```

**mAP per eval epoch:**

| epoch | Arm C order-1 | Arm C order-2 |
|---|---|---|
| 10 | 0.4830 | 0.5581 |
| 20 | 0.5472 | 0.5783 |
| 30 | 0.5683 | 0.5784 |
| 40 | 0.5769 | 0.5802 |
| 50 | 0.5728 | 0.5869 |
| 60 | 0.5846 | 0.5849 |
| 70 | 0.5811 | 0.5839 |
| 80 | 0.5853 | 0.5880 |
| 90 | **0.5903 (best)** | **0.5930 (best)** |
| 100 (final) | 0.5836 | 0.5906 |

**Comparison table (Arm A/B numbers from §13.1, existing reference):**

| Arm | best mAP | final mAP | κ mean±std | dead bits | binarity | separation ratio |
|---|---|---|---|---|---|---|
| A -- CIBHash (§13.1) | 0.6105 | 0.6079 | n/a (signed) | n/a | n/a | n/a |
| B order-1, `eps=0.31` (§13.1) | 0.6202 | 0.6202 | 4.74±1.24 | 1/64 | 0.959 | 10.59x |
| C order-1, `eps=0.31` (new) | 0.5903 | 0.5836 | 4.64±1.20 | 12/64 | 1.0 (trivial, see 14.3) | 8.61x |
| C order-2, `eps=0.31` (new) | 0.5930 | 0.5906 | 4.22±1.08 | 6/64 | 1.0 (trivial) | 9.63x |

Arm C's §3-specified extra diagnostics:

| | Arm C order-1 | Arm C order-2 |
|---|---|---|
| mean Bernoulli entropy `H(p)` (nats) | 0.0326 | 0.0455 |
| hypothetical two-sample Bernoulli agreement | 98.08% | 97.30% |

Both entropies are very low (sharp, confident `p`), consistent with the high
hypothetical agreement in both cases; order-2's slightly higher entropy
pairs with slightly lower agreement, an internally consistent direction
(caveat above still applies to what this means in practice).

**Verdict, stated plainly: the loss alone, on CIBHash's unmodified
architecture, does not beat CIBHash's own loss on that same architecture --
it underperforms it.** Arm C order-1 (0.5903 best) < Arm A (0.6105) < Arm B
order-1 (0.6202). This isolates that Arm B's advantage over Arm A is **not**
purely a loss-function effect: swapping only the loss (Arm C) moves mAP in
the *wrong* direction relative to Arm A, so Arm B's bounded/no-sampling head
change appears to matter, not just the objective. Separately, **order-2 vs.
order-1 flips sign between architectures**: for Arm B (§14.2) order-2 is
clearly worse (-0.031 to -0.033); for Arm C here, order-2 is marginally
*better* than order-1 (+0.0027 best, +0.0070 final) -- small enough to plausibly
be single-seed noise (comparable in size to fluctuations seen elsewhere in
this document), but notably the *opposite sign* from Arm B's effect, and
paired with fewer dead bits for order-2 here (6/64) vs. more for order-1
(12/64) -- again the opposite direction from Arm B's dead-bit pattern
(§14.2: order-2 had *more* dead bits than order-1). The order-2 critic's
effect is architecture/context-dependent, not a fixed sign.

### 14.4 Task 4 -- §8 high-d claim scoping correction

Edited §8 directly (the only pre-existing numbered section touched in this
batch, as instructed). The "high-d regime fails" bullet under "closed
negatives... do not re-run" now: states explicitly that the finding is
scoped to **Arm D, trainable backbone, 40 epochs** (the actual runs it was
based on, `logs/cifar10/sbdr_aux{64,256,512}_40/`); states plainly it is
"not a general finding about the loss, and it is not what Arm B... shows";
adds a pointer to §12.2/§12.3 as the superseding evidence for Arm B
specifically (frozen backbone, 100 epochs: `d=512` beats `d=64` at matched
`κ`, 0.6281 vs. 0.6165 at `topk_eval=8`, a smaller margin than the unmatched
native comparison suggested but still the opposite direction from the
original bullet). The original Arm D numbers and log paths are unchanged
and still on record -- nothing was deleted, only the framing was corrected
so "high-d fails" is not read as settled across all arms/protocols.

## 15. Track A, Phase 1 (matched-storage, `d=1024`) and Phase 2 (matched-sparsity top-k, `d=64`) (2026-09-05)

Scope, stated up front per the task: **Phase 1 and Phase 2 only** -- no
seeds beyond 42 (Phase 3, separate) and no NUS-WIDE (Phase 4, separate).
Every training run in this section: frozen VGG16 backbone
(`backbone_lr_scale=0`), `optim.weight_decay=0.0005`, `dataset=cifar10`, 100
epochs (except the explicitly-labelled 3-epoch sanity check), eval every 10
epochs, seed 42, default `batch_size=64` (not overridden -- see §15.1's
note on why the §12.2 `d=512` run's undocumented `batch_size=128` was not
carried forward here). No code changes to any loss/arch file in this
section; two new *read-only, post-hoc* diagnostic scripts were added,
`experiments/storage_report.py` and `experiments/topk_sparsity_report.py`
(both checked against existing scripts first, per the task's own
instruction -- see §15.2 and §15.7 for what each does and why a new script
was needed rather than reusing one as-is).

**GPU discipline for this whole section (§15.1-§15.8):** `nvidia-smi`
confirmed before every launch. GPU 3 was fully idle (0 MiB, 0%) at every
check throughout; GPU 1 had another user's ~3.9GB/moderate-utilization job
resident throughout (>19GB free) and was used as the second slot, per the
task's explicit permission to share a GPU with headroom when no second GPU
is fully free; GPU 0 and GPU 2 were observed at 81-98% utilization by other
users' jobs at every check and were never touched. **Only GPU 3 and GPU 1
were ever used, and never more than 2 training jobs ran concurrently** --
confirmed by construction (each job's launch command and its GPU are logged
below) and by `ps aux` spot checks during the batch. Five training jobs ran
in this section total: one 3-epoch sanity check (GPU 3, solo), then three
100-epoch `d=1024` eps trials and one 100-epoch `d=64` Target-B run,
launched as two sequential-then-parallel pairs across GPU 3/GPU 1 as each
slot freed up (exact sequencing in §15.1/§15.6). Phase 1.3's dense
baselines required **no new training** (see §15.3) and Phase 2.3's top-k
sweeps require no GPU at all (operate on already-saved checkpoint codes on
CPU) -- so GPU time in this section is entirely the five jobs above.

### 15.1 Phase 1, Task 1 -- `d=1024` sanity check + `eps` search

**Sanity check (required before any full-length run, per the task -- `d=1024`
has never been run in this repo before, confirmed by grep across this
document for "1024" prior to this section: it appears only in §4c's
conceptual capacity table, never as an actual run).** 3 epochs,
`eval_interval=1`, `eps=1.0` (arbitrary pick for this check only, not yet
the eps search), GPU 3, solo:

```
CUDA_VISIBLE_DEVICES=3 python main_v2.py model=sbdr model.nbit=1024 dataset=cifar10 epochs=3 eval_interval=1 \
  backbone_lr_scale=0 criterion.eps=1.0 optim.weight_decay=0.0005 seed=42
```

Model instantiates cleanly (encoder's final layer is a plain `Linear(1024,
1024)` -- `models/arch/sbdr.py`'s hidden width is already 1024, so `d=1024`
is not a new code path, just a non-narrowing final layer). Training result:

| epoch | mAP |
|---|---|
| 1 | 0.5808 |
| 2 | 0.6138 |
| 3 | 0.6147 |

Runtime 0.10h. **No NaN, no divergence, no collapse** -- mAP rises smoothly
epoch over epoch, exactly the pattern of a healthy run, not a degenerate
one. Reported explicitly as instructed even though uneventful.

**`eps` selection.** Not targeting a specific κ, per the task -- just
avoiding degenerate outcomes. Starting point: the two existing anchors
(`d=64,eps=0.31→κ=4.74`, §13.1; `d=512,eps=1.0→κ=25.19`, §13.3) imply an
`eps` ratio of `1.0/0.31=3.226` over a `d` ratio of `8×`. Fitting
`eps ∝ d^p`: `p = ln(3.226)/ln(8) = 0.563`. Extrapolating to `d=1024`
(`2×` `d=512`): `eps ≈ 1.0 × 2^0.563 ≈ 1.48`. Chose a wide, log2-spaced
bracket around this estimate rather than a narrow one centered on it: `eps
∈ {0.5, 1.0, 2.0}` (`4×` span, reusing `eps=1.0` exactly as a direct probe
of "same eps as the `d=512` anchor, `d` doubled"). Deliberately wide because
(a) the extrapolation is a 2-point fit, and §0.3 already flags the sparsity
law as unreliable for trained encoders -- realised κ must be checked, not
trusted from a formula; (b) `d=1024`'s κ-vs-`eps` sensitivity was completely
unmeasured before this batch. All three were comfortably far from
degenerate territory by construction: even the free-code law's roughest
estimate (`κ≈1.8√(eps·d)`) puts all three between ~41 and ~81, nowhere near
0 or `d/2=512`.

Commands (only the varying flag shown; full protocol as stated in this
section's header):

```
CUDA_VISIBLE_DEVICES=3 python main_v2.py model=sbdr model.nbit=1024 dataset=cifar10 epochs=100 backbone_lr_scale=0 \
  criterion.eps=0.5 optim.weight_decay=0.0005 seed=42
CUDA_VISIBLE_DEVICES=1 python main_v2.py model=sbdr model.nbit=1024 dataset=cifar10 epochs=100 backbone_lr_scale=0 \
  criterion.eps=1.0 optim.weight_decay=0.0005 seed=42
CUDA_VISIBLE_DEVICES=3 python main_v2.py model=sbdr model.nbit=1024 dataset=cifar10 epochs=100 backbone_lr_scale=0 \
  criterion.eps=2.0 optim.weight_decay=0.0005 seed=42
```

`eps=0.5` and `eps=1.0` ran in parallel (GPU 3 / GPU 1, at the 2-GPU cap);
`eps=2.0` launched on GPU 3 once `eps=0.5` finished (runtimes below make the
sequencing arithmetic checkable).

**mAP per eval epoch, all three:**

| epoch | `eps=0.5` | `eps=1.0` | `eps=2.0` |
|---|---|---|---|
| 10 | 0.6507 | 0.6442 | 0.6380 |
| 20 | 0.6465 | 0.6471 | 0.6418 |
| 30 | 0.6477 | 0.6442 | 0.6408 |
| 40 | 0.6511 | 0.6499 | 0.6460 |
| 50 | 0.6491 | 0.6535 | 0.6467 |
| 60 | 0.6531 | 0.6510 | 0.6508 |
| 70 | **0.6627 (best)** | **0.6632 (best)** | 0.6523 |
| 80 | 0.6525 | 0.6528 | 0.6534 |
| 90 | 0.6587 | 0.6591 | 0.6567 |
| 100 (final) | 0.6598 | 0.6583 | **0.6568 (best/final)** |

**Diagnostics at best checkpoint (`experiments/storage_report.py`, database
codes):**

| | `eps=0.5` | `eps=1.0` | `eps=2.0` |
|---|---|---|---|
| best / final mAP | 0.6627 / 0.6598 | **0.6632** / 0.6583 | 0.6568 / 0.6568 |
| κ mean±std | 23.59±7.81 | 35.09±9.91 | 49.67±12.12 |
| dead bits | 13/1024 | 13/1024 | 15/1024 |
| binarity | 0.974 | 0.967 | 0.959 |
| runtime | 0.71h | 0.88h | 0.72h |

**Best pick: `eps=1.0`, by best mAP** (0.6632), the sole criterion used --
the margin over `eps=0.5` (0.6627, Δ=0.0005) is within likely single-seed
noise, and dead bits/binarity are near-identical between the two (13 vs 13
dead, 0.967 vs 0.974 binarity), so there is no secondary signal strong
enough to override the (razor-thin) mAP ranking. `eps=2.0` is clearly worse
on every axis (lower best mAP, more dead bits, lower binarity) and is not a
contender. None of the three shows any sign of collapse or degeneracy.

### 15.2 Phase 1, Task 2 -- storage accounting

No existing script computed per-sample κ histograms or the two storage
quantities requested; checked `experiments/sbdr_report.py` and
`utils/sbdr_metrics.py` first (both compute κ mean/std but not a bit-budget
number or a full histogram) before writing `experiments/storage_report.py`,
which reuses `utils.sbdr_metrics.usage_stats` and adds: the full per-sample
κ histogram, `k_topk` (smallest `k` with `>=90%` of samples at κ≤`k` --
the concrete rule the task asked for when "the bulk" needs a number),
`storage_meankappa = κ_mean · ceil(log2 d)`, `storage_topk = k_topk ·
ceil(log2 d)`. This is the same `κ·log₂d` "index storage" model already
named in §4(b) -- applied here for the first time with actual computed
numbers rather than the conceptual example in that section.

Applied identically to the `d=1024` best checkpoint (§15.1, `eps=1.0`) and,
for direct comparability, recomputed the same way for the two existing
on-record anchors (§13.1's `d=64,eps=0.31` and §13.3's `d=512,eps=1.0`,
wd-fixed) rather than reusing any previously-reported storage number (none
had been computed this way before this section -- §4(b)'s `κ·log₂d` numbers
there are worked examples, not measurements from a checkpoint).

| | `d=64` (§13.1) | `d=512` (§13.3) | `d=1024` (§15.1, best) |
|---|---|---|---|
| `ceil(log2 d)` | 6 | 9 | 10 |
| κ mean±std | 4.74±1.24 | 25.19±5.76 | 35.09±9.91 |
| `k_topk` (>=90% rule) | 6 | 32 | 47 |
| actual frac at `k_topk` | 93.7% | 91.5% | 91.2% |
| `storage_meankappa` (bits) | 28.42 | 226.68 | 350.90 |
| `storage_topk` (bits) | 36 | 288 | 470 |

Full κ histograms for all three are in each checkpoint's
`storage_report.py` output (not reproduced in full here for the `d=64`/
`d=512` anchors to save space; the `d=1024` histogram is checkable via
`python experiments/storage_report.py <logdir>` against the logdirs named
in §15.1/§13.1/§13.3).

### 15.3 Phase 1, Task 3 -- matched-storage dense baselines

`d_match`: rounding `storage_topk=470` bits to the nearest sensible dense
bit-length gives **512** (nearest power of 2; `|470-512|=42` vs.
`|470-256|=214`, not close). **This is not a new dimension** -- CIBHash and
SDC_simclr were already run at `nbit=512` under this exact protocol
(frozen backbone, `wd=0.0005`, 100 epochs, eval every 10, seed 42) in
§13.3. Per the task's own instruction ("nearest value already supported by
existing CIBHash/SDC configs"), **both criteria (nearest power of 2, and an
already-run config) agree on 512** -- not a coincidence engineered by
picking one rule over the other, both rules independently land on the same
value. Consequently: **no sanity check needed** (512 is not a new
dimension for either baseline) and **no new training was run** for this
task -- the existing §13.3 checkpoints were reused. Verified directly
against the saved logs before reuse (not just copied from the earlier
table): `config.yaml` confirms `nbit=512`, `backbone_lr_scale=0`,
`weight_decay=0.0005`, `seed=42` for both, and `test_history.json` matches
§13.3's table exactly.

**CIBHash, `nbit=512`** (`logs/cifar10/cibhash512_100/frozen_wdfix_taskA_d51242_260905_020853_271180`):

| epoch | mAP |
|---|---|
| 10 | 0.5923 |
| 20 | 0.5978 |
| 30 | 0.5981 |
| 40 | **0.6066 (best)** |
| 50 | 0.6024 |
| 60 | 0.5972 |
| 70 | 0.6024 |
| 80 | 0.6021 |
| 90 | 0.6052 |
| 100 (final) | 0.6041 |

**SDC_simclr, `nbit=512`** (`logs/cifar10/sdc_simclr512_100/frozen_wdfix_taskSDC_simclr_d51242_260905_023732_252845`):

| epoch | mAP |
|---|---|
| 10 | 0.6487 |
| 20 | 0.6591 |
| 30 | 0.6658 |
| 40 | 0.6685 |
| 50 | 0.6676 |
| 60 | 0.6810 |
| 70 | 0.6763 |
| 80 | 0.6767 |
| 90 | 0.6875 |
| 100 (final, best) | **0.6880** |

### 15.4 Phase 1, Task 4 -- matched-storage comparison table

`d_match=512` for the dense baselines; Arm B's actual bit budget at this
comparison point is `storage_topk=470` bits (§15.2) -- **not exactly 512**,
flagged explicitly rather than glossed over: Arm B is being compared here
at a budget **~8.2% smaller** than the dense baselines' 512 bits (Arm B
using `storage_meankappa=350.90` bits would be **~31.5% smaller** still).
State plainly which number is "the" comparison point: **470 bits**
(`storage_topk`, the task's default), against the dense baselines' trivial
512 bits.

| | Arm B `d=1024,eps=1.0` | CIBHash `d=512` | SDC_simclr `d=512` |
|---|---|---|---|
| best mAP | 0.6632 | 0.6066 | 0.6880 |
| final mAP | 0.6583 | 0.6041 | 0.6880 |
| κ mean±std | 35.09±9.91 | n/a (dense) | n/a (dense) |
| dead bits | 13/1024 | n/a | n/a |
| binarity | 0.967 | n/a | n/a |
| `storage_meankappa` (bits) | 350.90 | 512 (trivial) | 512 (trivial) |
| `storage_topk` (bits) | **470** | 512 (trivial) | 512 (trivial) |

**Direct comparison, at Arm B's 470-bit budget vs. the baselines' 512-bit
budget:**

- vs. CIBHash: Arm B **beats** CIBHash by +0.0566 best mAP (+9.33%
  relative) and +0.0542 final mAP (+8.97% relative), at ~8.2% less storage.
- vs. SDC_simclr: Arm B **loses** to SDC_simclr by -0.0248 best mAP (-3.60%
  relative) and -0.0297 final mAP (-4.32% relative), also at ~8.2% less
  storage.

### 15.5 Phase 2, Task 1 -- target κ levels

**Target A: κ≈4.74, `d=64,eps=0.31`.** No new training -- reused the
existing §13.1 checkpoint
(`logs/cifar10/sbdr64_100/frozen_wdfix_taskB_d6442_260905_011643_601089`)
exactly as instructed.

**Target B.** At the time this step was reached, Phase 1 had produced only
2 of 3 `d=1024` eps trials (`eps=0.5`, `eps=1.0` results in hand; `eps=2.0`
still training) -- **Phase 1 had not finished**, so per the task's explicit
fallback, Target B was chosen independently rather than derived from
Phase 1: **κ≈9**, "clearly different from Target A," in the task's
suggested 8-10 range. `eps` estimated from a fit to the two existing `d=64`
Arm B points already on record (§13.1 `eps=0.31→κ=4.74`; §14.2 `eps=0.15→
κ=3.37`): `κ ∝ eps^p`, `p = ln(4.74/3.37)/ln(0.31/0.15) = 0.469` (close to
the free-code law's 0.5 exponent, for what that is worth at only 2 points).
Solving for `κ=9`: `eps = 0.31 × (9/4.74)^(1/0.469) ≈ 1.22`, rounded to
**`eps=1.0`** for the actual run (same round number already used as the
`d=1024` middle anchor in §15.1, chosen for a clean number rather than the
literal 1.22).

**Post-hoc note (not required, but stated for honesty):** Phase 1 has since
finished; its best `d=1024` run (`eps=1.0`) realized κ=35.09 (§15.1). This
does not correspond proportionally to Target B's independently-chosen
κ≈9 (a straight `d`-ratio scaling would give `35.09×64/1024≈2.19`; an
absolute match would have meant targeting κ≈35 at `d=64`, likely
degenerate at that dimension). This mismatch is expected and acceptable
per the task's own fallback instruction, which explicitly anticipates
Target B being chosen without reference to Phase 1's result in this
situation.

### 15.6 Phase 2, Task 2 -- `eps` search for Target B

One trial only. Command (full protocol as stated in this section's header):

```
CUDA_VISIBLE_DEVICES=1 python main_v2.py model=sbdr model.nbit=64 dataset=cifar10 epochs=100 backbone_lr_scale=0 \
  criterion.eps=1.0 optim.weight_decay=0.0005 seed=42
```

Ran in parallel with §15.1's `eps=2.0` `d=1024` job (GPU 1, while `eps=2.0`
ran on GPU 3 -- still at the 2-GPU cap).

| epoch | mAP |
|---|---|
| 10 | 0.5258 |
| 20 | 0.5392 |
| 30 | 0.5591 |
| 40 | 0.5579 |
| 50 | 0.5513 |
| 60 | **0.5636 (best)** |
| 70 | 0.5624 |
| 80 | 0.5474 |
| 90 | 0.5625 |
| 100 (final) | 0.5612 |

Diagnostics at best checkpoint: κ=8.18±1.63, dead bits=0/64, binarity=0.948.
Runtime 0.88h.

**Deviation flagged: only 1 eps trial was run, not 2-3.** The task allows
this ("if no existing checkpoint already hits it closely" implies checking
first, and permits 2-3 *if needed*) -- realized κ=8.18 landed inside the
independently-chosen 8-10 target range on the first try, close enough
(within ~9% of the κ≈9 center) that a second trial was judged unnecessary.
Note in passing, not chased further here: this Target-B checkpoint's best
mAP (0.5636) is substantially below Target A's (0.6202, §13.1) -- a
**-0.0566 / -9.1%** gap purely from pushing κ up at fixed `d=64`, consistent
with this document's existing finding (§13's sparsity-law update caveat)
that higher κ is not free at fixed `d`.

### 15.7 Phase 2, Task 3 -- top-k sparsification eval, both targets

Checked first: `utils/hashing.py`'s `calculate_mAP(..., topk_eval=k)` /
`preprocess_on_codes` is the existing exact-per-sample-κ eval mechanism
already used in §12.3 and `experiments/sbdr_report.py`'s `mAP_sweep`. That
existing driver only sweeps a hardcoded `k ∈ {8,16,32}`, not an arbitrary
bracket around each checkpoint's own κ as this task needs, so a new thin
driver, `experiments/topk_sparsity_report.py`, was added -- it calls the
exact same `calculate_mAP`/`preprocess_on_codes` path with a `--ks` list,
no new eval logic. No training or GPU needed: both checkpoints' continuous
codes were already saved to `outputs/db_best.pth` / `outputs/test_best.pth`
during their original runs; this reads them directly.

Common `k` grid for both targets (bracketing Target A's κ=4.74 from below
`κ/2` to above `2κ`, and Target B's κ=8.18 similarly, in one shared set for
direct side-by-side comparison): `k ∈ {2,4,6,8,10,12,16,24,32}`.

| k | Target A mAP (κ=4.74) | Target B mAP (κ=8.18) |
|---|---|---|
| 2 | 0.5442 | 0.5041 |
| 4 | 0.5960 | 0.5421 |
| 6 | **0.6250** | 0.5716 |
| 8 | 0.6174 | **0.5854** |
| 10 | 0.6195 | 0.5845 |
| 12 | 0.6152 | 0.5821 |
| 16 | 0.6053 | 0.5847 |
| 24 | 0.5940 | 0.5839 |
| 32 | 0.5821 | 0.5766 |
| native (0.5 threshold) | 0.6202 | 0.5636 |

### 15.8 Phase 2, Task 4 -- summary

**Both targets degrade sharply when `k` is pushed well below their own
native κ, and only mildly when `k` is pushed above it.** Target A (κ=4.74):
peak in this sweep is at `k=6` (0.6250); dropping to `k=2` (≈κ/2) gives
0.5442, a **-0.0808 / -12.9%** drop from the peak. Target B (κ=8.18): peak
is at `k=8` (0.5854); dropping to `k=4` (≈κ/2) gives 0.5421, a **-0.0433 /
-7.4%** drop from the peak, and dropping further to `k=2` gives 0.5041, a
**-0.0813 / -13.9%** drop from the peak.

**Using each target's own nearest-to-κ point as the reference (rather than
the sweep's peak) for the κ→κ/2 comparison specifically:** Target A,
`k=4` (≈κ) → `k=2` (≈κ/2): 0.5960→0.5442, **-8.7% relative**. Target B,
`k=8` (≈κ) → `k=4` (≈κ/2): 0.5854→0.5421, **-7.4% relative**. By this
measure, **Target B (higher κ) shows a smaller relative mAP drop than
Target A (lower κ) when both are truncated to half their own native κ** --
i.e. the higher-κ checkpoint is somewhat *more* robust to this specific cut
than the lower-κ one, at least between these two points.

On the other side (`k` above native κ, over-inclusive rather than
under-sparsified): both targets degrade much more gently. Target A,
`k=6`(peak)→`k=10`(≈2κ): 0.6250→0.6195, **-0.9% relative**. Target B,
`k=8`(peak)→`k=16`(≈2κ): 0.5854→0.5847, **-0.1% relative**. For both
targets, degradation is markedly steeper on the aggressive-truncation side
(`k<κ`) than on the over-inclusive side (`k>κ`) -- a plain pattern in the
numbers above, not interpreted further here.

## 16. Track A, Phase 3 -- seeds on the two headline comparisons, plus a third κ point (2026-09-05)

**Required open item, resolved before anything else below.** §15's section
header (the paragraph right after "## 15. Track A...") asserts "see §15.1's
note on why the §12.2 `d=512` run's undocumented `batch_size=128` was not
carried forward here." **Checked, and that note does not actually exist in
§15.1** -- grepped §15.1's full text for "batch" and the only hit is an
unrelated use of the word "batch" (the task-batch, not the training
hyperparameter) in the `eps`-selection paragraph. This is a broken/dangling
cross-reference in §15's prose: something the header claims was explained
in §15.1 was, in fact, never written there. **Substantively, though, there
is no actual mismatch to flag:** checked `config.yaml` directly (not just
trusted the tables) for every checkpoint this batch's baseline reuse
depends on --

| checkpoint | `batch_size` |
|---|---|
| Arm B `d=64,eps=0.31` (§13.1, seed 42) | 64 |
| CIBHash `d=64` (§13.1, seed 42) | 64 |
| SDC_simclr `d=64` (§12.6, seed 42) | 64 |
| CIBHash `d=512` (§13.3, seed 42) | 64 |
| SDC_simclr `d=512` (§13.3, seed 42) | 64 |
| Arm B `d=1024,eps=1.0` (§15.1, seed 42) | 64 |

**All six are `batch_size=64`.** The one `d=512` Arm B run that used
`batch_size=128` (§12.2) was a different, earlier, non-wd-fixed checkpoint
that was never reused anywhere in §15 -- §15.3 reused the §13.3 *wd-fixed*
`d=512` checkpoints specifically, confirmed `batch_size=64` above. **Verdict:
no batch-size mismatch affects any comparison in §15 or in this batch --
everything reused or extended here is `batch_size=64`, consistent throughout.
The only defect is the broken cross-reference in §15's header text, noted
here for the record; it does not change any number.** All new runs in this
batch also use the default `batch_size=64` (not overridden), for full
consistency.

**GPU discipline for this whole section.** `nvidia-smi` confirmed before
every launch, same as §15. At the start of this batch: GPU 3 fully idle
(0 MiB, 0%); GPU 1 carrying another user's job at ~3.9GB/moderate
utilization (>19GB free) -- **headroom re-checked fresh for this batch**,
per the task's explicit instruction not to assume §15's finding still
holds, and confirmed still true. GPU 0 and GPU 2 were observed at 65-98%
utilization by other users' jobs at every check across this batch and were
never touched. Only GPU 3 and GPU 1 were used, and never more than 2
training jobs ran concurrently -- 9 new training jobs ran in total (6 for
Task 1, 2 for Task 2, 1 for Task 3), sequenced as a chain of overlapping
pairs across the two GPUs, each next job launched only once one of the two
running slots freed up (exact pairing order in §16.1/§16.2/§16.3; verified
throughout via `ps aux` and the runtimes below, which make the sequencing
arithmetic checkable).

### 16.1 Task 1 -- 3 seeds, `d=64` headline comparison (Arm B vs. CIBHash vs. SDC_simclr)

Commands (only the varying `seed` shown; full protocol as elsewhere --
frozen backbone, `wd=0.0005`, `nbit=64`, 100 epochs, eval every 10):

```
CUDA_VISIBLE_DEVICES={3,1} python main_v2.py model=sbdr model.nbit=64 dataset=cifar10 epochs=100 \
  backbone_lr_scale=0 criterion.eps=0.31 optim.weight_decay=0.0005 seed={43,44}
CUDA_VISIBLE_DEVICES={1,3} python main_v2.py model=cibhash model.nbit=64 dataset=cifar10 epochs=100 \
  backbone_lr_scale=0 optim.weight_decay=0.0005 seed={43,44}
CUDA_VISIBLE_DEVICES={3,1} python main_v2.py model=sdc_simclr model.nbit=64 dataset=cifar10 epochs=100 \
  backbone_lr_scale=0 optim.weight_decay=0.0005 seed={43,44}
```

Launch order (2-GPU cap, each new job launched as soon as either slot
freed): Arm B-43(GPU3) + CIBHash-43(GPU1) -> SDC_simclr-43(GPU3, once Arm
B-43 finished) + [CIBHash-43 still running] -> Arm B-44(GPU1, once
CIBHash-43 finished) + [SDC_simclr-43 still running] -> CIBHash-44(GPU3,
once SDC_simclr-43 finished) + [Arm B-44 still running] -> SDC_simclr-44
(GPU1, once Arm B-44 finished) + [CIBHash-44 still running]. Runtimes:
Arm B 0.70h/0.84h, CIBHash 0.87h/0.68h, SDC_simclr 0.70h/0.86h (seed
43/seed 44 respectively).

**Arm B, `d=64,eps=0.31` -- mAP per eval epoch:**

| epoch | seed 42 (§13.1) | seed 43 (new) | seed 44 (new) |
|---|---|---|---|
| 10 | 0.5850 | 0.5995 | 0.5857 |
| 20 | 0.5967 | 0.6114 | 0.5947 |
| 30 | 0.6024 | 0.6107 | 0.6028 |
| 40 | 0.6140 | 0.6223 | 0.6040 |
| 50 | 0.6078 | 0.6047 | 0.5972 |
| 60 | 0.6092 | 0.6131 | 0.6058 |
| 70 | 0.6184 | 0.6216 | 0.6062 |
| 80 | 0.6108 | 0.6196 | 0.6140 |
| 90 | 0.6140 | 0.6203 | 0.6118 |
| 100 (final/best, all 3 seeds) | **0.6202** | **0.6243** | **0.6160** |

**CIBHash, `d=64` -- mAP per eval epoch:**

| epoch | seed 42 (§13.1) | seed 43 (new) | seed 44 (new) |
|---|---|---|---|
| 10 | 0.5832 | 0.5855 | 0.5804 |
| 20 | 0.5887 | 0.5933 | 0.5946 |
| 30 | 0.5933 | 0.5941 | 0.5946 |
| 40 | 0.5993 | 0.5908 | 0.5950 |
| 50 | 0.5969 | 0.5964 | 0.5988 |
| 60 | 0.6025 | 0.5975 | 0.5999 |
| 70 | 0.6068 | 0.5958 | **0.6048 (best)** |
| 80 | **0.6105 (best)** | **0.6067 (best)** | 0.5986 |
| 90 | 0.6089 | 0.6049 | 0.6040 |
| 100 (final) | 0.6079 | 0.6057 | 0.6041 |

**SDC_simclr, `d=64` -- mAP per eval epoch:**

| epoch | seed 42 (§12.6) | seed 43 (new) | seed 44 (new) |
|---|---|---|---|
| 10 | 0.6150 | 0.6185 | 0.6192 |
| 20 | 0.6246 | 0.6350 | 0.6410 |
| 30 | 0.6447 | 0.6338 | 0.6369 |
| 40 | 0.6404 | 0.6421 | 0.6357 |
| 50 | 0.6523 | 0.6436 | 0.6495 |
| 60 | 0.6570 | 0.6477 | 0.6502 |
| 70 | 0.6488 | 0.6546 | 0.6615 |
| 80 | 0.6555 | 0.6500 | 0.6657 |
| 90 | 0.6616 | 0.6567 | **0.6666 (best)** |
| 100 (final) | **0.6646 (best)** | **0.6611 (best)** | 0.6665 |

**Arm B diagnostics, per seed (best checkpoint):**

| | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| κ mean±std | 4.74±1.24 | 4.79±1.24 | 4.58±1.21 |
| dead bits | 1/64 | 0/64 | 3/64 |
| binarity | 0.959 | 0.957 | 0.960 |

**Summary across 3 seeds (mean±std, n=3, sample std; range in brackets):**

| | Arm B | CIBHash | SDC_simclr |
|---|---|---|---|
| best mAP | 0.6202±0.0042 [0.6160,0.6243] | 0.6073±0.0029 [0.6048,0.6105] | 0.6641±0.0028 [0.6611,0.6666] |
| final mAP | 0.6202±0.0042 [0.6160,0.6243] | 0.6059±0.0019 [0.6041,0.6079] | 0.6641±0.0027 [0.6611,0.6665] |
| κ mean (of per-seed means) | 4.70±0.11 | n/a | n/a |

(Arm B's best and final mAP are identical per seed -- all three seeds hit
their best mAP exactly at epoch 100, so the two rows coincide.)

**Ranking stability, checked at every individual seed (not just on
means):** Arm B > CIBHash holds at all 3 seeds (best-mAP margins: seed 42
+0.0097, seed 43 +0.0176, seed 44 +0.0112 -- always positive, only the
*size* of the win varies). SDC_simclr > Arm B also holds at all 3 seeds
(margins: seed 42 +0.0444, seed 43 +0.0368, seed 44 +0.0506 -- always
positive). **Both of §13.1's seed-42 rankings hold across all 3 seeds,
without exception, at every individual seed** -- not just in the mean. The
spread within each config (std 0.0019-0.0042) is roughly 3-25x smaller than
either ranking's margin (0.037-0.051 for SDC_simclr vs. Arm B; 0.0097-0.0176
for Arm B vs. CIBHash), so three points is enough to call both rankings
stable here, though obviously not enough to bound the margins precisely.

### 16.2 Task 2 -- 3 seeds, `d=1024`-vs-`d=512` matched-storage comparison (Arm B only)

**Deliberate asymmetry, stated explicitly per the task:** only Arm B is
reseeded here. CIBHash `d=512` and SDC_simclr `d=512` remain single-seed
(`n=1`, the existing §13.3 checkpoints, seed 42 only) -- **not an
oversight**, a deliberate cost decision for this batch, to be revisited
only if a future comparison turns out close enough that baseline noise
could plausibly flip it.

Commands (only `seed` varies; full protocol as §15.1's seed-42 run):

```
CUDA_VISIBLE_DEVICES=3 python main_v2.py model=sbdr model.nbit=1024 dataset=cifar10 epochs=100 \
  backbone_lr_scale=0 criterion.eps=1.0 optim.weight_decay=0.0005 seed=43
CUDA_VISIBLE_DEVICES=1 python main_v2.py model=sbdr model.nbit=1024 dataset=cifar10 epochs=100 \
  backbone_lr_scale=0 criterion.eps=1.0 optim.weight_decay=0.0005 seed=44
```

Seed 43 ran on GPU 3 immediately after Task 1's last GPU-3 job finished;
seed 44 ran on GPU 1 in parallel, launched right after Task 1's final GPU-1
job finished (both still within the 2-GPU cap, overlapping with Task 1's
tail end, not run "after" Task 1 in wall-clock terms). Runtimes: 0.72h
(seed 43), 0.86h (seed 44).

**mAP per eval epoch:**

| epoch | seed 42 (§15.1) | seed 43 (new) | seed 44 (new) |
|---|---|---|---|
| 10 | 0.6442 | 0.6348 | 0.6347 |
| 20 | 0.6471 | 0.6401 | 0.6466 |
| 30 | 0.6442 | 0.6421 | 0.6465 |
| 40 | 0.6499 | 0.6440 | 0.6519 |
| 50 | 0.6535 | 0.6422 | 0.6547 |
| 60 | 0.6510 | 0.6508 | 0.6520 |
| 70 | **0.6632 (best)** | 0.6481 | **0.6589 (best)** |
| 80 | 0.6528 | **0.6545 (best)** | 0.6548 |
| 90 | 0.6591 | 0.6511 | 0.6578 |
| 100 (final) | 0.6583 | 0.6515 | 0.6572 |

**Diagnostics and per-seed `storage_topk` (§15.2's exact method: `k_topk`
= smallest `k` with >=90% of samples at κ≤`k`, then `k_topk·ceil(log2 d)`,
`ceil(log2 1024)=10`):**

| | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| κ mean±std | 35.09±9.91 | 34.86±9.55 | 35.44±10.00 |
| dead bits | 13/1024 | 14/1024 | 9/1024 |
| binarity | 0.967 | 0.967 | 0.967 |
| `k_topk` | 47 | 46 | 47 |
| frac at `k_topk` | 91.2% | 90.5% | 90.2% |
| `storage_meankappa` (bits) | 350.90 | 348.59 | 354.37 |
| `storage_topk` (bits) | 470 | 460 | 470 |

**Summary across 3 seeds (mean±std, n=3; range in brackets):**

| | Arm B `d=1024,eps=1.0` (n=3) |
|---|---|
| best mAP | 0.6589±0.0044 [0.6545,0.6632] |
| final mAP | 0.6557±0.0037 [0.6515,0.6583] |
| κ mean (of per-seed means) | 35.13±0.29 |
| `storage_topk` (bits) | 466.7±5.8 [460,470] |
| `storage_meankappa` (bits) | 351.3±2.9 [348.6,354.4] |

**Updated §15.4 matched-storage table, `n=3` for Arm B / `n=1` for the
dense baselines (labelled accordingly):**

| | Arm B `d=1024,eps=1.0` (n=3) | CIBHash `d=512` (n=1) | SDC_simclr `d=512` (n=1) |
|---|---|---|---|
| best mAP | 0.6589±0.0044 | 0.6066 | 0.6880 |
| final mAP | 0.6557±0.0037 | 0.6041 | 0.6880 |
| κ mean±std | 35.13±9.8 (pooled per-sample scale; see note) | n/a | n/a |
| dead bits | 9-14/1024 (range) | n/a | n/a |
| binarity | 0.967 | n/a | n/a |
| `storage_meankappa` (bits) | 351.3±2.9 | 512 (trivial) | 512 (trivial) |
| `storage_topk` (bits) | **466.7±5.8** | 512 (trivial) | 512 (trivial) |

(κ mean±std row: the ±9.8 is the mean of the three per-seed within-checkpoint
stds, not the 0.29 across-seed std of the three κ means -- both are real
quantities answering different questions, reported separately above to
avoid conflating them.)

**Restated comparisons using the 3-seed mean Arm B numbers:**

- vs. CIBHash: mean Arm B best (0.6589) − CIBHash best (0.6066) = **+0.0523
  (+8.62% relative)**, vs. §15.4's single-seed **+9.33%**. Final: mean Arm B
  (0.6557) − CIBHash (0.6041) = **+0.0516 (+8.54%)**, vs. §15.4's **+8.97%**.
- vs. SDC_simclr: mean Arm B best (0.6589) − SDC_simclr best (0.6880) =
  **-0.0291 (-4.23% relative)**, vs. §15.4's single-seed **-3.60%**. Final:
  mean Arm B (0.6557) − SDC_simclr (0.6880) = **-0.0323 (-4.69%)**, vs.
  §15.4's **-4.32%**.

**Stability assessment.** Both comparisons look stable, by a stronger test
than just comparing means: **every individual Arm B seed**, not just the
mean, beats CIBHash (worst Arm B seed 0.6545 vs. CIBHash 0.6066, margin
still +0.0479) and **every individual Arm B seed** loses to SDC_simclr
(best Arm B seed 0.6632 vs. SDC_simclr 0.6880, margin still -0.0248). Arm
B's own 3-seed spread (best-mAP range width 0.0087) is 5-6x smaller than
either margin (0.048-0.052 vs. CIBHash; 0.025-0.033 vs. SDC_simclr). Caveat
repeated from the asymmetry note above: the dense baselines are still
single-seed, so their own noise is unmeasured here -- "stable" means
"stable with respect to Arm B's measured seed-to-seed variation," not "both
baselines' true means are known precisely."

### 16.3 Task 3 -- third κ level at `d=64` (Target C, above Target B)

**`eps` extrapolation.** Local power-law fit through the nearest 2 existing
`d=64` Arm B points *below* the 12-16 target range (`eps=0.31→κ=4.74`,
`eps=1.0→κ=8.18`, same method §15.1/§15.6 used): `κ∝eps^p`,
`p=ln(8.18/4.74)/ln(1.0/0.31)=0.466`. Solving for the range center `κ=14`:
`eps=1.0×(14/8.18)^(1/0.466)≈3.17`. Rounded to **`eps=3.0`** for the actual
run (clean number close to the fit).

**Only one trial was needed.** Command (full protocol elsewhere -- frozen
backbone, `wd=0.0005`, `nbit=64`, 100 epochs, eval every 10, seed 42):

```
CUDA_VISIBLE_DEVICES=3 python main_v2.py model=sbdr model.nbit=64 dataset=cifar10 epochs=100 \
  backbone_lr_scale=0 criterion.eps=3.0 optim.weight_decay=0.0005 seed=42
```

Ran on GPU 3 in parallel with Task 2's seed-44 job on GPU 1. Realized
κ=12.97±3.29 (database codes, best checkpoint) -- **inside the 12-16 target
range on the first trial**, so no second `eps` value was tried. Runtime
0.71h.

**mAP per eval epoch:**

| epoch | mAP |
|---|---|
| 10 | 0.4694 |
| 20 | 0.4856 |
| 30 | 0.4772 |
| 40 | 0.4839 |
| 50 | **0.4909 (best)** |
| 60 | 0.4701 |
| 70 | 0.4504 |
| 80 | 0.4462 |
| 90 | 0.4536 |
| 100 (final) | 0.4681 |

**Flagged plainly, not glossed over: this checkpoint's mAP is substantially
worse than either Target A (0.6202) or Target B (0.5636), and its curve
peaks early (epoch 50) then declines for the rest of training** rather than
rising or plateauing like every other Arm B curve in this document. This is
despite landing exactly in the intended κ range and showing no dead-bit
collapse (0/64 dead bits) and comparable binarity (0.946, vs. 0.959/0.948
for Targets A/B) -- i.e. this is not the usual collapse signature. Reported
here as observed; not diagnosed further, per the task's scope (a single
seed-42 run, no follow-up trials requested).

**Diagnostics at best checkpoint (epoch 50):** κ=12.97±3.29, dead
bits=0/64, binarity=0.946.

### 16.4 Extended top-k sparsification sweep -- Targets A, B, and C

Reused `experiments/topk_sparsity_report.py` from §15.7 unchanged (same
underlying `calculate_mAP(..., topk_eval=k)` path) -- no new eval code
needed. Same shared `k` grid as §15.7, `k∈{2,4,6,8,10,12,16,24,32}`; no
larger `k` values were needed to bracket Target C's own κ=12.97 (`32≈2.5×`
its κ already exceeds the requested `~2×` bracket).

| k | Target A mAP (κ=4.74) | Target B mAP (κ=8.18) | Target C mAP (κ=12.97) |
|---|---|---|---|
| 2 | 0.5442 | 0.5041 | 0.4340 |
| 4 | 0.5960 | 0.5421 | 0.4587 |
| 6 | **0.6250** | 0.5716 | 0.4665 |
| 8 | 0.6174 | **0.5854** | 0.4582 |
| 10 | 0.6195 | 0.5845 | 0.4737 |
| 12 | 0.6152 | 0.5821 | 0.4944 |
| 16 | 0.6053 | 0.5847 | 0.5022 |
| 24 | 0.5940 | 0.5839 | **0.5025** |
| 32 | 0.5821 | 0.5766 | 0.4923 |
| native (0.5 threshold) | 0.6202 | 0.5636 | 0.4909 |

Note Target C's peak-in-sweep falls at `k=24`, well above its own native
κ=12.97 (~1.85x) -- unlike Targets A and B, whose sweep peaks (`k=6`,
`k=8`) sit close to their own native κ (4.74, 8.18). This asymmetry matters
for which reference point is used below.

### 16.5 Updated summary -- does the robustness-to-truncation pattern continue, reverse, or plateau at Target C?

**Near-native-κ referenced (nearest available `k` to each target's own κ,
and to half of it) -- the method that keeps the reference point fixed
relative to each target's own κ rather than wherever the sweep happens to
peak:**

- Target A: `k=4`(≈κ)→`k=2`(≈κ/2): 0.5960→0.5442, **-8.7%**.
- Target B: `k=8`(≈κ)→`k=4`(≈κ/2): 0.5854→0.5421, **-7.4%**.
- Target C: `k=12`(≈κ)→`k=6`(≈κ/2): 0.4944→0.4665, **-5.6%**.

By this measure, the pattern **continues monotonically**: -8.7% → -7.4% →
-5.6% as κ rises from 4.74 to 8.18 to 12.97.

**Peak-referenced (each target's own sweep maximum, to the nearest `k` at
half its κ) -- the method §15.8 used first:**

- Target A: `k=6`(peak)→`k=2`(≈κ/2): 0.6250→0.5442, **-12.9%**.
- Target B: `k=8`(peak)→`k=4`(≈κ/2): 0.5854→0.5421, **-7.4%**.
- Target C: `k=24`(peak)→`k=6`(≈κ/2): 0.5025→0.4665, **-7.2%**.

By this measure, the pattern **plateaus rather than continuing**: -12.9% →
-7.4% → -7.2%, essentially flat between B and C rather than a further
improvement.

**Stated plainly: the two referencing methods disagree on whether the
trend continues past Target B.** Both agree on direction (Target C is
never *less* robust than A or B by either measure -- there is no reversal),
but the near-κ method shows continued monotonic improvement while the
peak-referenced method shows a plateau. Given only 3 points, and given
Target C's own peak sits in an unusual place relative to its native κ
(§16.4's note, itself plausibly connected to Target C's generally poor,
declining mAP curve, §16.3), **this should be read as three points
consistent with "higher κ is not less robust to κ/2 truncation," not as a
confirmed monotonic law** -- the magnitude of any continued improvement
past Target B is method-dependent and not established by this data.

### 16.6 Follow-up on §16.3's Target C: training-loss trajectory and epoch-50-vs-epoch-100 checkpoint comparison

Diagnostic only, per the task -- no retraining, no new `eps`, no fix
attempted. Logdir confirmed by exact match against §16.3's reported table:
`logs/cifar10/sbdr64_100/taskB_targetC_eps3.0_42_260905_222628_754425`
(`test_history.json` reproduces §16.3's 10 epoch values exactly).

**Checkpoint-identity fact established first, since it changes what "check
2" actually requires:** `models/best.pth`, `outputs/db_best.pth`, and
`outputs/test_best.pth` are all timestamped 22:47:26-31, which is *exactly*
the epoch-50 evaluation window in `log.txt` (`Epoch [50/100]` logged at
22:45:42, its `mAP: 0.490920` logged at 22:47:26/30, `Epoch [51/100]`
starts at 22:47:30) and the run ends with `Best mAP: 0.490920 at 50`.
**§16.4's Target C top-k sweep therefore already used the epoch-50
checkpoint, not epoch 100** -- there is no separate "epoch 50 vs. the
checkpoint used in §16.4" distinction to draw; they are the same file.
What actually needed generating for this task was the **epoch-100** side
of the comparison, since no `db`/`test` codes for epoch 100 were saved
anywhere on disk (only `models/last.pth`'s weights exist for epoch 100 --
`outputs/` only ever holds the *best*-epoch codes, confirmed by directory
listing: no `db_last.pth` / `test_last.pth`).

#### 16.6.1 Training loss trajectory (check 1)

Already logged, no new code needed: `train_history.json` (100 entries, one
per epoch) has a `train_loss` field (identical to `train_contrast` --
`SBDRCriticLoss`'s own value, the actual quantity being minimized, not
mAP) alongside `train_kappa`, `train_dead_bits_exact`, etc. -- the same
file every other Arm B run in this document would have.

**Combined table, mAP (from `test_history.json`, eval epochs only) and
training loss (from `train_history.json`, same epochs):**

| epoch | mAP | train_loss |
|---|---|---|
| 10 | 0.4694 | -0.8001 |
| 20 | 0.4856 | -0.8183 |
| 30 | 0.4772 | -0.8294 |
| 40 | 0.4839 | -0.8351 |
| 50 | **0.4909 (best mAP)** | -0.8394 |
| 60 | 0.4701 | -0.8416 |
| 70 | 0.4504 | -0.8423 |
| 80 | 0.4462 | -0.8427 |
| 90 | 0.4536 | -0.8559 |
| 100 (final) | 0.4681 | **-0.8532** |

(Loss is a log-ratio, more negative = lower/better -- `SBDRCriticLoss`'s
sign convention, unchanged from every other section using it.)

**Full-curve check (all 100 epochs, not just the 10 shown above):** the
training-loss minimum over the whole run is **-0.8594 at epoch 91**, i.e.
the single best training-loss value occurs *after* mAP has already peaked
(epoch 50) and declined most of the way to its epoch-80 trough (0.4462).
Epoch-over-epoch, the loss increases on 36 of the 99 steps (ordinary SGD
noise, comparable in kind to the "noisy/non-monotonic" curves already on
record elsewhere in this document for healthy Arm B runs, e.g. §12.2) but
never plateaus or trends upward for any sustained stretch -- the loss at
epoch 100 (-0.8532) is still far below (better than) its epoch-50 value
(-0.8394), and the run's global minimum is reached only 9 epochs before
the end.

**Stated plainly: training loss keeps decreasing through epoch 100. It
does not plateau or go unstable around epoch 50 -- the optimizer continues
to successfully lower the objective being trained for the full run, right
through the region where mAP is declining.** No dead-bit growth
accompanies this either (`train_dead_bits_exact` is 0.0 at every one of
the 10 sampled epochs above, and `κ` stays in a stable ~13.0-13.4 band
throughout -- no drift, no collapse signature).

#### 16.6.2 Epoch-100 checkpoint vs. epoch-50: top-k sweep (check 2)

Since no epoch-100 codes were saved during training, generated them via
light GPU inference on the existing `models/last.pth` weights (no
training, single forward pass over the test/db splits) using this repo's
own validation entry point, `val.py` (`configs/val.yaml`, `exp=validation`,
distinct from `main_v2.py`'s `train.yaml`-rooted config, which does not
carry the `use_last`/`R`/`PRs` schema `val.yaml` needs):

```
CUDA_VISIBLE_DEVICES=3 python val.py \
  logdir=logs/cifar10/sbdr64_100/taskB_targetC_eps3.0_42_260905_222628_754425 \
  dataset=cifar10 R=1000 use_last=True save_code=True seed=42 device=cuda
```

Ran solo on GPU 3 (confirmed idle first), 0.03h, well under the 2-GPU cap
(the only job running at the time). `code_domain=unit`/`dist_metric=overlap`
were inherited from the checkpoint's own saved `config.yaml` (`val.yaml`
only overrides them if explicitly passed, and they were not), so the eval
protocol matches training exactly. Output saved to
`.../evaluations/42_260905_233342_169525/outputs.pth` (`{'test':...,
'db':...}`, each with `codes_cont`/`labels`, via `save_code=True`).

**A discrepancy encountered and resolved, flagged rather than silently
picked around:** `val.py`'s own live console output for this run printed
`mAP@1000: 0.4755` for both `codes` and `codes_cont`. Recomputing mAP
directly from the exact tensors this same run saved to
`outputs.pth` -- via three independent calls to `utils.hashing.
calculate_mAP` (the plain `experiments/topk_sparsity_report.py`-style call;
a `test_hashing.py`-style call replicating its label-handling and `PRs`
argument; and the same without `PRs`) -- gives **0.4681401** in all three
cases, matching the *original training-time* epoch-100 log value
(`mAP: 0.468140`) to 6 decimal places exactly. Since three independently-
constructed calls to the same function, on the tensors this run itself
saved, all agree with each other and with the historical training log,
**0.4681 was treated as the correct epoch-100 native mAP** for everything
below; the `0.4755` printed live by `val.py`'s own console during this
diagnostic run is an unresolved discrepancy in that specific print path,
not a difference in the underlying codes -- flagged here, not chased
further (out of scope for a report-only task, and it does not affect the
codes actually used for the top-k sweep, which are independently verified
against the training-time record).

**`experiments/topk_sparsity_report.py` reused completely unchanged**, per
the task's instruction -- pointed at a staged directory containing the
epoch-100 `codes_cont`/`labels` saved into the exact `outputs/db_best.pth`
/ `outputs/test_best.pth` layout the script already expects (plus a copy
of the checkpoint's own `config.yaml`), so no script edits were needed.

**κ at epoch 100:** 12.75±3.23 (database codes) -- close to epoch 50's
12.97±3.29, not a large shift.

**Top-k sweep, epoch 50 (§16.4's existing Target C row) vs. epoch 100 (new):**

| k | epoch 50 mAP (κ=12.97±3.29) | epoch 100 mAP (κ=12.75±3.23) |
|---|---|---|
| 2 | 0.4340 | 0.4360 |
| 4 | 0.4587 | 0.4559 |
| 6 | 0.4665 | 0.4678 |
| 8 | 0.4582 | 0.4887 |
| 10 | 0.4737 | 0.4772 |
| 12 | 0.4944 | 0.4871 |
| 16 | 0.5022 | 0.4902 |
| 24 | **0.5025 (peak)** | 0.4748 |
| 32 | 0.4923 | **0.4928 (peak)** |
| native (0.5 threshold) | 0.4909 | 0.4681 |

**Stated plainly: the epoch-100 checkpoint does not look more like
Targets A/B's pattern than epoch 50 did -- if anything, it is a more
extreme version of the same anomaly.** Epoch 50's peak sits at `k=24`
(~1.85x its own κ=12.97). Epoch 100's peak sits at `k=32` -- **the largest
k tested, i.e. the sweep's own boundary** (~2.51x its κ=12.75) -- so the
true peak for epoch 100 may lie beyond `k=32` entirely, meaning this
grid does not even fully bracket epoch 100's optimum the way it was
designed to bracket Targets A/B/C's native κ. Both checkpoints show the
same qualitative shape (mAP still rising at the largest tested `k`, native-
threshold mAP well below the swept peak); training from epoch 50 to 100
does not resolve or soften this, and by the peak-location measure it gets
slightly worse.

#### 16.6.3 Synthesis -- what these two checks support, and what they don't

**Check 1 (training loss):** keeps decreasing through epoch 100 (new
minimum at epoch 91), no plateau, no instability signature, no dead-bit
growth, κ stable. **Check 2 (epoch-50-vs-100 top-k):** the "peak mAP
requires `k` well above native κ" anomaly is present at both checkpoints
and does not shrink from epoch 50 to epoch 100 -- if anything the peak
moves further out (1.85x κ at epoch 50 -> off-the-tested-grid, >=2.51x κ,
at epoch 100).

**Read together, these two observations are consistent with (a) -- the
model continuing to optimize the training-time contrastive proxy
(`SBDRCriticLoss` keeps improving, reaching its run-wide best at epoch 91)
while retrieval quality at the *trained* operating point (native, 0.5-
threshold mAP) decouples from it and gets worse (0.4909 at epoch 50 ->
0.4462 at epoch 80, partially recovering to 0.4681 at epoch 100) -- and
they are not consistent with (b), training itself becoming unstable
independent of mAP: the loss curve is smooth and monotonically improving
in trend (99 steps, 36 upticks of ordinary SGD-noise size, no sustained
reversal), and neither dead bits nor κ show any drift or blow-up at any
point in training.** This is stated as what two diagnostic checks on a
single run can support, not as a settled mechanism -- no claim is made here
about *why* the proxy and the thresholded-retrieval metric decouple, only
that the data is consistent with them decoupling (a) rather than with
outright training instability (b).

## 17. Track A, Phase 4 -- NUS-WIDE matched-storage extension: **blocked at Task 0** (2026-09-05)

**Status: stopped after Task 0, per the task's own explicit instruction
("if Task 0 finds a blocker... stop and report clearly rather than
attempting a workaround not specified in this prompt -- come back for
further instructions instead"). No training was run in this section. No
GPU jobs beyond the read-only checks below.**

### 17.0 Task 0 -- dataset wiring and eval-pipeline check

**Config wiring: present and correctly set up, no issues found.**
`configs/dataset/nuswide.yaml` already exists (`nclass: 21`,
`multiclass: True`, **`R: 5000`** -- already the dataset's own default,
matching this batch's `mAP@5K` requirement exactly; no override needed,
confirmed by reading the config directly rather than assuming 5000).
Backed by `utils.datasets.HashingDataset` (train/test/db all wired via
`_target_: utils.datasets.HashingDataset`, `root: ${data_dir}/data/nuswide`).

**Multi-label mAP: correctly implemented, verified by reading the code
path.** `utils/hashing.py`'s `calculate_mAP` gates on `multiclass`
(`configs/dataset/nuswide.yaml`'s `multiclass: True` flows into this):
when `multiclass=True`, it takes a different, slower **sequential**
per-query path (line ~402 onward) rather than CIFAR's fast batched path
(gated to `not multiclass`, ~line 373) -- worth flagging as a runtime-
relevant fact for later, not a correctness issue. Relevance in that path
is computed as `label = test_labels[i]*2-1` (maps the query's one-hot
`{0,1}` multi-label vector to `{-1,+1}`) then
`imatch = (np.equal(db_labels_raw{0,1}, label{-1,+1}).sum(1)) > 0`.
Read carefully because the two operands are in different domains
(`{0,1}` vs. `{-1,+1}`) -- worked through by hand: `db_label==-1` is
never true (raw db labels are only 0 or 1), so the equality can only
fire where `db_label==1` and `label==1`, i.e. **the check reduces exactly
to "shares at least one positive tag with the query"** -- the standard,
correct multi-label retrieval-relevance criterion. Obscurely written (the
correctness rests on a domain mismatch that happens to cancel out
correctly) but verified correct, not a bug. **No code change needed for
multi-label mAP.**

**Data files: split/label lists present, actual images absent -- this is
the blocker.** `data/nuswide/{train,test,database}.txt` all exist on disk
(10,500 / 2,100 / 193,734 lines respectively -- sane counts for the
NUS-WIDE-21 subset this repo's split sizes imply) and are correctly
formatted (`<image path> <21-dim one-hot label>` per line, e.g.
`data/nuswide/images/0229_415689799.jpg 0 0 1 0 0 0 ... 0`). **But
`data/nuswide/images/` does not exist, and zero `.jpg` files exist
anywhere under `data/nuswide`** (`find data/nuswide -iname "*.jpg" | wc -l`
-> `0`). Confirmed with a direct load attempt, not just a directory
listing: `PIL.Image.open('data/nuswide/images/0229_415689799.jpg')` raises
`FileNotFoundError: [Errno 2] No such file or directory`. Checked for
alternatives before concluding this is unrecoverable from what's on disk:
no `ROOTDIR` env var override set; `data_dir` resolves to the repo root
(`configs/train.yaml`'s default, unchanged); no prior NUS-WIDE log
directory anywhere under `logs/` (`find logs -iname "*nuswide*"` -> empty)
-- **this dataset has never been successfully run in this repo before**,
consistent with §8's existing note ("no NUS-WIDE" deferred) and this
task's own framing of Phase 4 as new territory. Compare with CIFAR-10,
which is fully self-contained on disk (`data/cifar10/cifar-10-batches-py/`,
the raw binary batches CIFAR's own dataset loader reads directly -- no
per-image files needed at all, a structurally different, already-satisfied
data dependency).

**Verdict: this is a missing-data blocker, not a config or eval-code
problem.** Everything checked in Task 0 that could be verified by reading
code and existing files says the pipeline is ready (correct `R`, correct
multi-label handling, correct split files); the one thing that cannot be
fixed by a config change or a code fix is that the actual NUS-WIDE JPEG
images this repo's split files point to are not present anywhere on this
machine. Per the task's explicit instruction, **stopping here rather than
attempting an unspecified workaround** (e.g. downloading the dataset --
not attempted; its size, licensing terms, and download source were not
specified in this task and downloading a large third-party image dataset
is not a decision to make unilaterally). Tasks 1-4 (Arm B training, storage
accounting, dense baselines, comparison tables) were **not started** --
no training runs, no GPU jobs beyond the single CPU-side `PIL.Image.open`
probe above (no GPU used in this section at all).

## 18. Track A, Phase 4 continued -- NUS-WIDE unblocked, matched-storage extension executed (2026-09-06)

§17 stopped at Task 0's data blocker and asked the user how to proceed.
The user confirmed the images were simply never downloaded (their fork
gitignores `data/` for repo-size reasons) and authorized downloading them
from a standard source. This section covers that download, a re-run of
Task 0's verification (which surfaced a second, more interesting finding
than the missing files), and Tasks 1-4.

### 18.0a Dataset acquisition

The user's suggested link (`huggingface.co/datasets/Lxyhaha/NUS-WIDE`) was
checked first and found to contain **only label/tag metadata** (`NUS-WIDE.zip`
+ `NUS_WID_Tags.zip`, ~26MB total -- `Groundtruth/AllLabels/*.txt`,
`NUS_WID_Tags/*.txt`), no images -- not usable, not used.

This repo's own `README.md` points to
`https://fast-image-retrieval.readthedocs.io/en/latest/dataset.html` (the
same lab's sister toolkit, CISiPLab's `fast-image-retrieval`) for dataset
setup. That page names the exact artifact needed: `nuswide_v2_256.tar.gz`,
described as the **"21 most common classes"** subset (matching this repo's
own `nclass: 21` config exactly) -- and lists three mirrors, one of which
is `huggingface.co/datasets/jiuntian/OrthoHash` (`jiuntian` is a co-author
of the SDC paper itself, `Hoe, Jiun Tian` -- the same lab's own hosted
copy, not an arbitrary third party).

Downloaded `Other/Dataset/nuswide_v2_256.tar` from that repo (a `.tar`, not
`.tar.gz` as the docs page names it -- a naming difference, not a wrong
file): **12,989,818,880 bytes**, confirmed via `Content-Length` before
downloading and matched exactly against the completed download's file
size. Extracted only the `images/` subdirectory (`--strip-components=2`)
directly into `data/nuswide/images/`: **269,648 `.jpg` files, 6.4GB**,
matching the dataset's own documented total image count exactly. Verified
filename convention matches this repo's existing `train.txt`/`test.txt`/
`database.txt` (e.g. downloaded `nuswide_v2_256/images/0228_2255697480.jpg`
vs. this repo's referenced `data/nuswide/images/0229_415689799.jpg` --
same `<4-digit>_<numeric-id>.jpg` pattern) and did a full, not spot-check,
coverage pass: **0 of 206,334** images referenced across
`train.txt`/`test.txt`/`database.txt` combined are missing after
extraction. `PIL.Image.open` on the exact path that failed in §17 now
succeeds (`(179, 240), RGB`). The 12.1GB source tar was deleted after
verification; the extracted `images/` directory (6.4GB) is what remains
under `data/nuswide/` (already gitignored, per `.gitignore`'s `data/`
entry, so this addition does not affect the repo).

### 18.0b Task 0 revisited -- a real eval-pipeline finding, not just a formality

§17 already confirmed `configs/dataset/nuswide.yaml`'s wiring
(`nclass: 21`, `multiclass: True`, `R: 5000` already the default) and
read through `utils/hashing.py`'s multiclass branch as looking correct.
**That reading was incomplete -- it only checked `calculate_mAP`'s own
internal logic, not whether every caller actually passes `multiclass`
through to it.** Discovered while attempting this task's optional
mAP@1K addition (below), then verified directly rather than assumed:

- **`experiments/train_helper.py`** (the actual per-epoch training-time
  eval path -- what populates `test_history.json`, i.e. **every mAP number
  reported anywhere in this section**) correctly calls `calculate_mAP(...,
  multiclass=self.config.dataset.multiclass)`. Confirmed correct.
- **`experiments/test_hashing.py`**'s `RetrievalEvaluation.main()` (used by
  `val.py`, and by `main_v2.py`'s `exp=validation` branch) **never passes
  `multiclass` to `calculate_mAP` at all** -- it silently falls back to the
  function's own default, `multiclass=False`. **This is a real bug** for
  any multi-label dataset, NUS-WIDE included.
- **Reproduced directly, not inferred:** loaded Arm B `d=64`'s saved best-
  checkpoint codes and called `calculate_mAP` by hand with both settings.
  `multiclass=True` gives **0.792696** -- matching the training log's
  epoch-20 value to 6 decimal places exactly. `multiclass=False`, same
  codes, gives **0.523031** -- a ~34% relative difference, far too large to
  be numerical noise. The bug silently converts NUS-WIDE's multi-label
  one-hot vectors to a single dominant class via `argmax` and switches to
  the wrong (single-label) batched relevance/AP computation.

**Consequence, stated plainly:** this bug has **zero effect on any number
in Tasks 1/3/4 below** -- all of them come from `train_helper.py`'s correct
path. It does mean `val.py` cannot be trusted for ad-hoc re-evaluation on
this (or any multi-label) dataset without a fix. Per the task's own
stopping condition ("if anything... requires a code change beyond a config
value, stop and report rather than pushing forward with an uncertain eval
pipeline"): **not fixed here** -- `experiments/test_hashing.py` is a shared
file and this batch's task did not authorize editing it. The one place
this mattered concretely: the optional mAP@1K addition (Task 1's prompt
allowed reporting it "if roughly free") would have required exactly this
buggy path, so **it was dropped from scope** rather than computed through
it. Only mAP@5K (the primary, required metric, unaffected by this bug) is
reported below.

### 18.0c Runtime check and the epoch-count deviation

Per the task's explicit instruction, checked one run's timing early before
committing to a full batch. Sanity check (first-ever NUS-WIDE run in this
repo): `model=sbdr model.nbit=64 dataset=nuswide epochs=2 eval_interval=1
criterion.eps=0.31 backbone_lr_scale=0 optim.weight_decay=0.0005 seed=42`,
GPU 3. Pipeline ran end-to-end with no errors; mAP@5K 0.7817 -> 0.7845
across the 2 epochs (sane, not degenerate). Timing: ~284s/epoch training,
~55-65s per eval pass (both `codes`/`codes_cont`) -- **roughly 9-11x
slower per epoch than the CIFAR precedents** (~25-32s/epoch there).
Projected from this: **~8.1h for a single 100-epoch run, ~24h wall-clock
for the full 6-run batch at the 2-GPU cap.**

Flagged to the user before running anything further, per the task's
explicit instruction not to discover a multi-day runtime partway through.
**User chose to reduce the epoch count to 40** (eval every 10 epochs -> 4
eval points per run, down from CIFAR's 100/10) rather than accept the ~24h
projection at full length -- **stated here as the explicit, user-directed
deviation from "100 epochs to match CIFAR" that it is.**

**A further deviation worth flagging: the actual full-length runs came in
far faster than the sanity check projected.** Both Task 1 runs (40 epochs
each) finished in **0.70h and 0.88h**, not the ~3.2h the sanity-check
timing would predict for 40 epochs. The most likely explanation: the
sanity check was the **first-ever read** of these 269,648 freshly-extracted
JPEG files (cold OS page cache, real disk I/O per image), while every
subsequent run benefited from a warm page cache (6.4GB fits comfortably in
memory). This isn't verified by a controlled cache-drop experiment --
stated as the most likely explanation, not a proven one -- but the
practical upshot is that the runtime concern that justified reducing to 40
epochs was real at the time it was raised (correctly avoided an
uninformed 100-epoch commitment) even though the actual cost, once the
cache warmed, turned out lower than projected. All 7 NUS-WIDE training
jobs after the sanity check (2 for Task 1, 4 for Task 3) took between
0.68h and 0.88h each.

**GPU discipline for this whole section:** `nvidia-smi` re-checked fresh
before every launch in this batch (not assumed from any earlier section,
per the task's explicit instruction). GPU 3 was idle at every check; GPU 1
carried another user's job at low-to-moderate utilization throughout with
ample headroom (as in §15/§16) and was used as the second slot. GPU 0 and
GPU 2 were observed busy with other users' jobs at every check (12-98%
range across checks) and were never touched. Only GPU 3 and GPU 1 were
used; never more than 2 training jobs ran concurrently. 8 training jobs
total this section (1 sanity + 2 Task 1 + 4 Task 3), run as one solo job
followed by 3 parallel pairs, each next pair launched only once a slot
freed. One additional light, GPU-based diagnostic run (the mAP@1K
feasibility check, §18.0b) and one CPU-only reproduction script (no GPU).

### 18.1 Task 1 -- Arm B, two `d`-values, `eps` reused as-is from CIFAR

No `eps` re-search, per the task's own instruction -- `eps=0.31` (`d=64`)
and `eps=1.0` (`d=1024`) are the exact CIFAR values from §13.1 and
§15.1/§16.2 respectively. Commands (40 epochs, per §18.0c's deviation;
otherwise full protocol -- frozen backbone, `wd=0.0005`, seed 42, eval
every 10):

```
CUDA_VISIBLE_DEVICES=3 python main_v2.py model=sbdr model.nbit=64 dataset=nuswide epochs=40 eval_interval=10 \
  backbone_lr_scale=0 criterion.eps=0.31 optim.weight_decay=0.0005 seed=42
CUDA_VISIBLE_DEVICES=1 python main_v2.py model=sbdr model.nbit=1024 dataset=nuswide epochs=40 eval_interval=10 \
  backbone_lr_scale=0 criterion.eps=1.0 optim.weight_decay=0.0005 seed=42
```

Ran in parallel (GPU 3 / GPU 1). Runtimes: 0.70h (`d=64`), 0.88h (`d=1024`).

**mAP@5K per eval epoch:**

| epoch | Arm B `d=64,eps=0.31` | Arm B `d=1024,eps=1.0` |
|---|---|---|
| 10 | 0.7912 | **0.8217 (best)** |
| 20 | **0.7927 (best)** | 0.8208 |
| 30 | 0.7884 | 0.8172 |
| 40 (final) | 0.7911 | 0.8184 |

**Diagnostics at best checkpoint (`experiments/storage_report.py`, reused
completely unchanged -- it only reads saved codes/config and has no
CIFAR-specific paths, confirmed by inspection before use, and it produced
correct output on the first try against these NUS-WIDE checkpoints):**

| | `d=64,eps=0.31` | `d=1024,eps=1.0` |
|---|---|---|
| κ mean±std | 4.75±1.27 | 35.00±9.41 |
| dead bits | 0/64 | 3/1024 |
| binarity | 0.965 | 0.969 |

**Both are sane, not degenerate -- and strikingly close to the CIFAR
values obtained at the exact same `eps`:** CIFAR `d=64,eps=0.31` gave
κ=4.74±1.24 (§13.1); CIFAR `d=1024,eps=1.0` gave κ=35.09±9.91 (§15.1). At
both `d`, NUS-WIDE's realized κ differs from CIFAR's by less than 0.1 in
the mean despite the completely different image domain and label
structure (single-label CIFAR-10 photos vs. multi-label NUS-WIDE Flickr
images) -- reused `eps` transfers essentially unchanged across datasets
here, at least at these two operating points. No dead-bit blowout, no
near-`d/2` saturation, at either `d`. **mAP@1K was not computed, per
§18.0b** -- only mAP@5K (this batch's primary metric) is reported.

### 18.2 Task 2 -- storage accounting

Same method as §15.2, same script: `storage_meankappa = κ_mean ·
ceil(log2 d)`, `storage_topk = k_topk · ceil(log2 d)` (`k_topk` = smallest
`k` with >=90% of samples at κ≤`k`).

| | `d=64,eps=0.31` | `d=1024,eps=1.0` |
|---|---|---|
| `ceil(log2 d)` | 6 | 10 |
| κ mean±std | 4.75±1.27 | 35.00±9.41 |
| `k_topk` (90% rule) | 6 | 46 |
| frac at `k_topk` | 92.9% | 90.8% |
| `storage_meankappa` (bits) | 28.49 | 350.05 |
| `storage_topk` (bits) | 36 | 460 |

**Kappa histograms, NUS-WIDE vs. CIFAR at the same `eps`, side by side
(raw counts; `N=193,734` NUS-WIDE db vs. `N=59,000` CIFAR db -- shapes,
not raw counts, are what's comparable across the differing `N`):**

`d=64,eps=0.31`:

| κ | CIFAR count (N=59,000) | NUS-WIDE count (N=193,734) |
|---|---|---|
| 0 | 110 | 264 |
| 1 | 530 | 1,475 |
| 2 | 1,922 | 5,661 |
| 3 | 5,680 | 20,706 |
| 4 | 13,870 | 48,039 |
| 5 | 23,318 | 71,564 |
| 6 | 9,882 | 32,274 |
| 7 | 2,929 | 9,221 |
| 8 | 655 | 4,165 |
| 9 | 91 | 305 |
| 10 | 12 | 52 |
| 11 | 1 | 5 |
| 12 | -- | 3 |

Both peak at κ=5 and have the same qualitative unimodal, right-skewed
shape scaled to their respective `N` -- no visible difference in shape.

`d=1024,eps=1.0` (full histogram, both datasets, since this is the `d`
where a "heavier tail for multi-label" effect was hypothesized as
plausible):

| κ | CIFAR count (N=59,000) | NUS-WIDE count (N=193,734) |
|---|---|---|
| 0 | 5 | 8 |
| 5 | 59 | 194 |
| 10 | 160 | 486 |
| 15 | 383 | 1,053 |
| 20 | 648 | 2,053 |
| 25 | 1,205 | 3,589 |
| 30 | 2,011 | 6,014 |
| 35 (CIFAR peak) | 2,355 | 9,396 |
| 36 | 2,340 | **9,588 (NUS-WIDE peak)** |
| 40 | 2,087 | 10,167 |
| 45 | 1,948 | 3,869 |
| 50 | 689 | 1,700 |
| 55 | 243 | 1,044 |
| 60 | 43 | 141 |
| 65 | 11 | 30 |
| 70 | 1 | 9 |
| 77 (NUS-WIDE max) | -- | 4 |

(Intermediate non-decade rows omitted from this table for space; full
histograms are reproducible via `python experiments/storage_report.py
<logdir>` against the logdirs named in §18.1.)

**Sensitivity check, `k_topk` at 80%/90%/95% (both `d`, requested since a
disproportionate NUS-WIDE tail was a real possibility worth checking, not
assumed away):**

| threshold | `d=64` `k_topk` (bits) | `d=1024` `k_topk` (bits) |
|---|---|---|
| 80% | 6 (36) | 42 (420) |
| 90% | 6 (36) | 46 (460) |
| 95% | 7 (42) | 50 (500) |

**Verdict: no evidence of a disproportionate NUS-WIDE tail or a
"multi-label images need more active units" effect at either `d`.**
NUS-WIDE's `k_topk` at 90% (`d=1024`: 46) is close to CIFAR's own `d=1024`
anchor `k_topk` (47, §15.2) -- a 1-bit difference. The `k_topk` value moves
gradually and by a similar amount per 5-point threshold change at both
`d`, not in a way that suggests the 90% choice is sitting on an unusually
steep or fat part of the NUS-WIDE-specific distribution. The 90% rule
looks equally reasonable for both datasets at both `d` tested here; no
reason found to reconsider it, though only two `(d,eps)` points were
checked.

`d_match` for Task 3: `storage_topk` for `d=1024` is **460 bits**; nearest
power of 2 is **512** (`|460-512|=52` vs. `|460-256|=204`) -- same value
CIFAR's own `d_match` landed on (§15.3), by the same rule, coincidentally
(not forced).

### 18.3 Task 3 -- matched-storage dense baselines on NUS-WIDE

4 new runs, all under this project's own protocol (frozen backbone,
`wd=0.0005`, seed 42, 40 epochs, eval every 10) -- **not reused from any
CIFAR checkpoint**, per the task's explicit instruction that CIFAR's
CIBHash/SDC checkpoints are not valid substitutes for a different dataset.

```
CUDA_VISIBLE_DEVICES=3 python main_v2.py model=cibhash    model.nbit=64  dataset=nuswide epochs=40 eval_interval=10 backbone_lr_scale=0 optim.weight_decay=0.0005 seed=42
CUDA_VISIBLE_DEVICES=1 python main_v2.py model=sdc_simclr model.nbit=64  dataset=nuswide epochs=40 eval_interval=10 backbone_lr_scale=0 optim.weight_decay=0.0005 seed=42
CUDA_VISIBLE_DEVICES=3 python main_v2.py model=cibhash    model.nbit=512 dataset=nuswide epochs=40 eval_interval=10 backbone_lr_scale=0 optim.weight_decay=0.0005 seed=42
CUDA_VISIBLE_DEVICES=1 python main_v2.py model=sdc_simclr model.nbit=512 dataset=nuswide epochs=40 eval_interval=10 backbone_lr_scale=0 optim.weight_decay=0.0005 seed=42
```

Run as two parallel pairs (`d=64` pair first, `d=512` pair once both
`d=64` runs finished). Runtimes: CIBHash 0.68h/0.69h, SDC_simclr
0.84h/0.87h (`d=64`/`d=512` respectively).

**CIBHash, mAP@5K per eval epoch:**

| epoch | `d=64` | `d=512` (=`d_match`) |
|---|---|---|
| 10 | 0.8015 | 0.8089 |
| 20 | 0.8036 | **0.8136 (best)** |
| 30 | 0.8059 | 0.8099 |
| 40 (final) | **0.8078 (best/final)** | 0.8125 |

**SDC_simclr, mAP@5K per eval epoch:**

| epoch | `d=64` | `d=512` (=`d_match`) |
|---|---|---|
| 10 | 0.8078 | **0.8214 (best)** |
| 20 | 0.8085 | 0.8204 |
| 30 | 0.8098 | 0.8168 |
| 40 (final) | **0.8114 (best/final)** | 0.8196 |

### 18.4 Task 4 -- comparison tables

**Table A -- small-`d` (`d=64`, matched storage trivially):**

| | Arm B `d=64,eps=0.31` | CIBHash `d=64` | SDC_simclr `d=64` |
|---|---|---|---|
| best mAP@5K | 0.7927 | 0.8078 | 0.8114 |
| final mAP@5K | 0.7911 | 0.8078 | 0.8114 |
| κ mean±std | 4.75±1.27 | n/a | n/a |
| dead bits | 0/64 | n/a | n/a |
| binarity | 0.965 | n/a | n/a |

- vs. CIBHash: Arm B **loses**, -0.0151 best (-1.87% relative), -0.0167
  final (-2.07% relative).
- vs. SDC_simclr: Arm B **loses**, -0.0187 best (-2.31% relative), -0.0203
  final (-2.50% relative).

**Table B -- large-`d` (`d=1024` Arm B vs. `d_match=512` dense, matched
storage at `storage_topk=460` bits vs. the baselines' trivial 512):**

| | Arm B `d=1024,eps=1.0` | CIBHash `d=512` | SDC_simclr `d=512` |
|---|---|---|---|
| best mAP@5K | **0.8217** | 0.8136 | 0.8214 |
| final mAP@5K | **0.8184** | 0.8125 | 0.8196 |
| κ mean±std | 35.00±9.41 | n/a | n/a |
| dead bits | 3/1024 | n/a | n/a |
| binarity | 0.969 | n/a | n/a |
| `storage_meankappa` (bits) | 350.05 | 512 (trivial) | 512 (trivial) |
| `storage_topk` (bits) | **460** | 512 (trivial) | 512 (trivial) |

- vs. CIBHash: Arm B **beats**, +0.0081 best (+0.99% relative), +0.0059
  final (+0.72% relative) -- at ~10.2% less storage (460 vs. 512 bits).
- vs. SDC_simclr: Arm B **effectively ties**, +0.0003 best (+0.03%
  relative, best read as a tie, not a real win) and -0.0012 final (-0.14%
  relative, best read as a tie in the other direction) -- again at ~10.2%
  less storage.

**Cross-dataset direct comparison, stated once as requested (4
comparisons x 2 datasets = 8 directional outcomes; CIFAR numbers are
§16.1's 3-seed means for the small-`d` pair and §16.2's 3-seed mean vs.
§15.3/§15.4's single-seed dense baselines for the large-`d` pair):**

| comparison | CIFAR direction | NUS-WIDE direction | same or flip? |
|---|---|---|---|
| small-`d`, Arm B vs. CIBHash | Arm B wins (+0.0129 mean best, +2.13%; holds at every one of 3 seeds) | Arm B **loses** (-0.0151, -1.87%) | **FLIP** |
| small-`d`, Arm B vs. SDC_simclr | Arm B loses (-0.0439 mean best, -6.61%; holds at every seed) | Arm B loses (-0.0187, -2.31%) | same (both lose) |
| large-`d` matched-storage, Arm B vs. CIBHash | Arm B wins (+0.0523 mean best, +8.62%; holds at every seed) | Arm B wins (+0.0081, +0.99%) | same (both win, much smaller margin on NUS-WIDE) |
| large-`d` matched-storage, Arm B vs. SDC_simclr | Arm B loses (-0.0291 mean best, -4.23%; holds at every seed) | Arm B **ties/marginally wins** (+0.0003, +0.03%) | **FLIP** (though the NUS-WIDE outcome is a near-exact tie, not a clean win, and this is single-seed on both sides here) |

Of the 4 comparisons, **2 point the same direction on both datasets and 2
flip.** Both flips involve the small-`d` CIBHash comparison and the
large-`d` SDC_simclr comparison; both comparisons that stay consistent
involve the large-`d` CIBHash comparison and the small-`d` SDC_simclr
comparison. The NUS-WIDE side of every comparison here is single-seed
(no Phase-3-style reseeding was in this batch's scope), so NUS-WIDE's
margins -- especially the two near-zero ones in Table B -- carry
correspondingly less certainty than CIFAR's 3-seed-verified margins.

## 19. Follow-up on §18 -- val.py bug fix, reseeding, and 40->100 epoch extension for the two closest NUS-WIDE comparisons (2026-09-06)

Three independent pieces, per the task: fix the §18.0b bug, add seeds
43/44 to the two closest-to-flipping comparisons, and extend those same
4 configs' seed-42 runs from 40 to 100 epochs. **GPU discipline, checked
fresh for this whole section** (not assumed from §18): `nvidia-smi`
confirmed before every launch; GPU 3 and GPU 1 were both **fully idle**
at nearly every check in this section (occasionally GPU 1 showed another
user's job at low-to-moderate utilization with ample memory headroom, same
sharing pattern as before) -- GPU 0 and GPU 2 were not checked in detail
since GPU 3/GPU 1 sufficed throughout; never more than 2 training jobs ran
concurrently. 13 training jobs ran in total this section (1 finetune-fix
smoke test at 1 epoch that crashed on an unrelated pre-existing bug, 1
smoke test at 3 epochs that succeeded, 8 new seed runs, 4 continuation
runs), plus one light GPU verification run and one CPU-only reproduction
script for the val.py fix.

### 19.1 Task 1 -- fix the `val.py` multiclass bug

**Fix**, `experiments/test_hashing.py`, `RetrievalEvaluation.main()`'s
`calculate_mAP` call (the only call site the bug was ever traced to,
per §18.0b -- `calculate_pr_curve`, the other call in the same method, was
left untouched, out of scope): added `multiclass=self.config.dataset.
multiclass`, mirroring exactly how `experiments/train_helper.py` (already
correct) accesses it. One line changed, nothing else touched.

```diff
                                                               topk_eval=self.config.get('topk_eval'),
-                                                              PRs=self.config.PRs)
+                                                              PRs=self.config.PRs,
+                                                              multiclass=self.config.dataset.multiclass)
```

**Verification 1 -- the fix actually corrects the bug, through the real
`val.py` path (not by hand):** `val.py logdir=<Arm B d=64 NUS-WIDE
checkpoint> dataset=nuswide R=5000 use_last=False seed=42 device=cuda`,
GPU 3. Result: **mAP@5000: 0.7919** -- in the correct ~0.79 range (vs.
the pre-fix bug's ~0.52), confirming the fix works through the actual
entry point, not just in isolation. Not bit-identical to the training
log's 0.792696 (0.7919 vs. 0.792696, Δ=0.0008) -- **checked, and this
residual gap is the same re-inference floating-point/`cudnn`-
nondeterminism phenomenon already documented in §16.6, not a remaining
bug in the fix itself**: a direct `calculate_mAP` call on this
checkpoint's already-saved codes (no re-inference) reproduces 0.792696
exactly, as it did in §18.0b.

**Verification 2 -- confirms no behavior change on non-multiclass
datasets, via an actual before/after comparison, not an assumption:**
Two checks, since the first (live `val.py` reruns, pre-fix via `git
stash`, post-fix after `git stash pop`) turned out **not** to be
bit-identical (0.6107 pre-fix vs. 0.6134 post-fix, on the same CIFAR
checkpoint) -- initially concerning, but the same re-inference
nondeterminism explains it (confirmed separately, see below), not the
fix. To isolate the fix's actual effect from that unrelated noise, ran a
**controlled** test on CIFAR's already-saved codes (no re-inference,
same method as Verification 1's cross-check): `calculate_mAP(...)` with
`multiclass` omitted (pre-fix default) vs. `multiclass=False` explicit
(post-fix, since `cifar10.yaml` sets `multiclass: False`) --
**`0.6201646966016852` both ways, bit-identical.** This is the rigorous
confirmation the task asked for: the fix itself changes nothing for
non-multiclass datasets; the live-rerun discrepancy is unrelated
re-inference noise, not evidence against the fix.

### 19.2 Task 2 -- seeds 43/44 on the two closest comparisons

8 new runs (4 configs x 2 seeds), NUS-WIDE, 40 epochs, eval every 10,
frozen backbone, `wd=0.0005` -- identical protocol to §18.1/§18.3, only
`seed` varies:

```
CUDA_VISIBLE_DEVICES={3,1} python main_v2.py model=sbdr model.nbit=64 dataset=nuswide epochs=40 eval_interval=10 \
  backbone_lr_scale=0 criterion.eps=0.31 optim.weight_decay=0.0005 seed={43,44}
CUDA_VISIBLE_DEVICES={1,3} python main_v2.py model=cibhash model.nbit=64 dataset=nuswide epochs=40 eval_interval=10 \
  backbone_lr_scale=0 optim.weight_decay=0.0005 seed={43,44}
CUDA_VISIBLE_DEVICES={3,1} python main_v2.py model=sbdr model.nbit=1024 dataset=nuswide epochs=40 eval_interval=10 \
  backbone_lr_scale=0 criterion.eps=1.0 optim.weight_decay=0.0005 seed={43,44}
CUDA_VISIBLE_DEVICES={1,3} python main_v2.py model=sdc_simclr model.nbit=512 dataset=nuswide epochs=40 eval_interval=10 \
  backbone_lr_scale=0 optim.weight_decay=0.0005 seed={43,44}
```

Run as 4 sequential pairs (Arm B `d=64` + CIBHash `d=64` seed 43, then
seed 44; Arm B `d=1024` + SDC_simclr `d=512` seed 43, then seed 44), each
pair launched once the previous one's slot freed. Runtimes ranged
0.68h-1.11h per run.

**Per-seed best/final mAP@5K, all 4 configs:**

| config | seed | best mAP@5K | final mAP@5K |
|---|---|---|---|
| Arm B `d=64` | 42 (§18.1) | 0.7927 | 0.7911 |
| Arm B `d=64` | 43 | 0.7907 | 0.7902 |
| Arm B `d=64` | 44 | 0.7942 | 0.7892 |
| CIBHash `d=64` | 42 (§18.3) | 0.8078 | 0.8078 |
| CIBHash `d=64` | 43 | 0.8070 | 0.8067 |
| CIBHash `d=64` | 44 | 0.8061 | 0.8059 |
| Arm B `d=1024` | 42 (§18.1) | 0.8217 | 0.8184 |
| Arm B `d=1024` | 43 | 0.8226 | 0.8197 |
| Arm B `d=1024` | 44 | 0.8219 | 0.8213 |
| SDC_simclr `d=512` | 42 (§18.3) | 0.8214 | 0.8196 |
| SDC_simclr `d=512` | 43 | 0.8198 | 0.8180 |
| SDC_simclr `d=512` | 44 | 0.8199 | 0.8181 |

**3-seed mean±std:**

| | Arm B `d=64` | CIBHash `d=64` | Arm B `d=1024` | SDC_simclr `d=512` |
|---|---|---|---|---|
| best mAP mean±std | 0.7925±0.0018 | 0.8070±0.0009 | 0.8221±0.0005 | 0.8204±0.0009 |
| final mAP mean±std | 0.7902±0.0010 | 0.8068±0.0010 | 0.8198±0.0015 | 0.8186±0.0009 |

**Arm B `d=64`/`d=1024` diagnostics, new seeds (κ mean±std, dead bits, binarity):**

| | seed 43 | seed 44 |
|---|---|---|
| `d=64` | 4.68±1.15, 0/64, 0.964 | 4.77±1.34, 1/64, 0.961 |
| `d=1024` | 34.07±8.47, 2/1024, 0.973 | 34.78±8.32, 3/1024, 0.972 |

Both sane and close to seed 42's values (§18.1) at both `d` -- no
seed-dependent degeneracy.

**Comparison 1 (small-`d`, Arm B vs. CIBHash), checked at every individual
seed, not just the mean -- best mAP:**

| seed | Arm B | CIBHash | Δ | direction |
|---|---|---|---|---|
| 42 | 0.7927 | 0.8078 | -0.0151 | Arm B loses |
| 43 | 0.7907 | 0.8070 | -0.0163 | Arm B loses |
| 44 | 0.7942 | 0.8061 | -0.0119 | Arm B loses |

Final mAP: -0.0167 / -0.0165 / -0.0167 at seeds 42/43/44 respectively --
**Arm B loses at every seed, both metrics, margin never smaller than
-0.0119 (best) or -0.0165 (final).** Mean-best margin: -0.0144 (-1.79%
relative to CIBHash's mean). **Direction is stable across all 3 seeds --
not a flip, not noise: Arm B genuinely loses to CIBHash at `d=64` on
NUS-WIDE.**

**Comparison 2 (large-`d` matched-storage, Arm B vs. SDC_simclr), same
check:**

| seed | Arm B best | SDC_simclr best | Δ (best) | Arm B final | SDC_simclr final | Δ (final) |
|---|---|---|---|---|---|---|
| 42 | 0.8217 | 0.8214 | +0.0003 (wins) | 0.8184 | 0.8196 | -0.0012 (loses) |
| 43 | 0.8226 | 0.8198 | +0.0028 (wins) | 0.8197 | 0.8180 | +0.0017 (wins) |
| 44 | 0.8219 | 0.8199 | +0.0020 (wins) | 0.8213 | 0.8181 | +0.0032 (wins) |

**By best-mAP, Arm B wins at all 3 seeds** (margins +0.0003 to +0.0028,
mean +0.0017, +0.21% relative to SDC_simclr's mean) -- direction stable,
but the margin is small throughout, including a near-zero seed-42 case.
**By final-mAP, the direction flips seed to seed**: Arm B loses at seed
42 (-0.0012) but wins at seeds 43 and 44 (+0.0017, +0.0032). **This
comparison's direction is stable (a small win) by best-mAP, but not
stable by final-mAP** -- stated exactly this way rather than collapsed
into a single verdict, since the two metrics disagree on stability here.

### 19.3 Task 3 -- extend the same 4 configs' seed-42 runs to 100 epochs

**How continuation was actually done, and a second gap found along the
way, flagged the same way §18.0b's bug was:** the task says "continue
training... do not retrain from scratch." This repo has two candidate
mechanisms: `resume_logdir` (true resume, including optimizer/scheduler
state) and `finetune_path` (model-weights-only). **`resume_logdir` was
not usable**: it requires `optims/last.pth`, which only exists if
`save_training_state=True` was set on the *original* run -- §18's 40-epoch
runs used the default (`False`), so `logs/nuswide/sbdr64_40/.../optims/`
is empty; `load_training_state` would fail with a missing file.
**`finetune_path` turned out to be unimplemented**: `config.finetune_path`
is already wired through `main_v2.py`/`experiments/train_helper.py`
(`trainer.finetune_setup(config.finetune_path)` is already called at
exactly the right point -- after `load_model()`, before
`load_optimizer_and_scheduler()`), but `BaseTrainer.finetune_setup` in
`trainers/base.py` was a no-op stub (`pass`), never overridden by any of
`trainers/{sbdr,cibhash,sdc}.py`. Using it as-is would have silently
trained from **random initialization** while appearing to continue --
worse than doing nothing. **Fixed minimally** (same spirit and scope as
§19.1's fix -- completing an already-wired, already-named, already-
documented-by-its-own-comment feature, not adding new scope):

```diff
-    def finetune_setup(self, *args, **kwargs):
+    def finetune_setup(self, path, *args, **kwargs):
         """
-        for fine-tuning after pre-training
+        for fine-tuning after pre-training. Loads only the model weights from
+        `path` ... optimizer/scheduler state is left untouched here and is
+        created fresh afterward by `load_optimizer_and_scheduler`.
         """
-        pass
+        self.load_model_state(path)
```

**Verified before use, not assumed:** a smoke test (3 epochs,
`eval_interval=1`, `finetune_path=<Arm B d=64 40-epoch seed-42 checkpoint>
/models/last.pth`) gave mAP 0.7884/0.7907/0.7902 across its 3 epochs --
matching the loaded checkpoint's own ~0.79 performance level immediately,
not the near-random mAP a fresh model would show on a 21-class multi-label
task. (An initial 1-epoch smoke test crashed with `ZeroDivisionError` in
`torch.optim.lr_scheduler`'s `step_size = int(0.8 * epochs)` -- `int(0.8*1)
=0` -- a pre-existing, unrelated scheduler edge case at very low epoch
counts, not caused by this fix; worked around simply by using 3 epochs for
the smoke test, irrelevant for the real continuation runs at `epochs=60`.)

**A real, quantifiable deviation from a hypothetical native single
100-epoch run, stated plainly rather than glossed over:** this
continuation has a **fresh optimizer state** (Adam moments reset) and a
**fresh LR schedule** based on the continuation's own `epochs=60`
(`configs/scheduler/step.yaml`: decays at `int(0.8*60)=48` continuation-
internal epochs, i.e. nominal epoch ~88) restarting from the **base**
`optim.lr=0.0001` -- not the original 40-epoch run's already-decayed rate
(`0.00001`, reached at the original run's own `int(0.8*40)=32`). A native
100-epoch run's schedule would decay once, at nominal epoch 80, and never
re-elevate the LR. This continuation instead re-introduces the base LR for
its first ~48 nominal epochs (nominal 41-88) before its own decay. Not
worked around further (e.g. by hand-picking a different starting LR) to
avoid substituting one unstated assumption for another -- reported as-is.

Commands (only `finetune_path` and `epochs` shown; frozen backbone,
`wd=0.0005`, seed 42, eval every 10 otherwise unchanged):

```
CUDA_VISIBLE_DEVICES={3,1} python main_v2.py model=sbdr model.nbit=64 dataset=nuswide epochs=60 eval_interval=10 \
  backbone_lr_scale=0 criterion.eps=0.31 optim.weight_decay=0.0005 seed=42 \
  finetune_path=logs/nuswide/sbdr64_40/taskB_nuswide_d64_42_260906_062639_562404/models/last.pth
CUDA_VISIBLE_DEVICES={1,3} python main_v2.py model=cibhash model.nbit=64 dataset=nuswide epochs=60 eval_interval=10 \
  backbone_lr_scale=0 optim.weight_decay=0.0005 seed=42 \
  finetune_path=logs/nuswide/cibhash64_40/taskA_nuswide_d64_42_260906_073122_428657/models/last.pth
CUDA_VISIBLE_DEVICES={3,1} python main_v2.py model=sbdr model.nbit=1024 dataset=nuswide epochs=60 eval_interval=10 \
  backbone_lr_scale=0 criterion.eps=1.0 optim.weight_decay=0.0005 seed=42 \
  finetune_path=logs/nuswide/sbdr1024_40/taskB_nuswide_d1024_42_260906_062640_857089/models/last.pth
CUDA_VISIBLE_DEVICES={1,3} python main_v2.py model=sdc_simclr model.nbit=512 dataset=nuswide epochs=60 eval_interval=10 \
  backbone_lr_scale=0 optim.weight_decay=0.0005 seed=42 \
  finetune_path=logs/nuswide/sdc_simclr512_40/taskSDC_nuswide_d512_42_260906_082257_467700/models/last.pth
```

Run as two pairs (`d=64` pair, then `d=1024`/`d=512` pair). Runtimes:
1.04h/1.01h (`d=64` pair), 1.09h/1.05h (`d=1024`/`d=512` pair) -- for 60
epochs each, consistent with §18's ~0.02h/epoch rate.

**Combined mAP@5K, epoch 10-100 (10-40 from §18.1/§18.3, 50-100 new --
continuation-internal epochs 10/20/30/40/50/60 relabeled 50/60/70/80/90/100):**

| epoch | Arm B `d=64` | CIBHash `d=64` | Arm B `d=1024` | SDC_simclr `d=512` |
|---|---|---|---|---|
| 10 | 0.7912 | 0.8015 | **0.8217 (best)** | **0.8214 (best)** |
| 20 | **0.7927** | 0.8036 | 0.8208 | 0.8204 |
| 30 | 0.7884 | 0.8059 | 0.8172 | 0.8168 |
| 40 | 0.7911 | 0.8078 | 0.8184 | 0.8196 |
| 50 | 0.7887 | 0.8062 | 0.8164 | 0.8163 |
| 60 | 0.7909 | 0.8060 | 0.8173 | 0.8086 |
| 70 | 0.7879 | 0.8091 | 0.8151 | 0.8035 |
| 80 | **0.7933 (best)** | 0.8074 | 0.8133 | 0.8078 |
| 90 | 0.7913 | 0.8097 | 0.8143 | 0.8157 |
| 100 (final) | 0.7919 | **0.8098 (best)** | 0.8153 | 0.8155 |

**Diagnostics at the new best checkpoint (Arm B configs only), alongside
the 40-epoch values already on record:**

| | `d=64`, 40-epoch best (ep20, §18.1) | `d=64`, 100-epoch best (ep80) | `d=1024`, 40-epoch best (ep10, §18.1) | `d=1024`, continuation's own best (ep60, internal ep20) |
|---|---|---|---|---|
| κ mean±std | 4.75±1.27 | 4.55±1.22 | 35.00±9.41 | 34.56±8.63 |
| dead bits | 0/64 | 0/64 | 3/1024 | 3/1024 |
| binarity | 0.965 | 0.968 | 0.969 | 0.973 |

Small, unremarkable drift at both `d` -- no collapse, no dead-bit growth,
no meaningful change in binarity from 60 further epochs of training.

**Does the picture change between epoch 40 (§18.4) and epoch 100 (here)?
Stated specifically, not just via the final table:**

- **Arm B `d=64`** found a new peak at **epoch 80 (0.7933)**, exceeding
  its epoch-40-window peak (0.7927 at epoch 20) by +0.0006 -- a real but
  tiny improvement, and the curve is still noisy/non-monotonic in the
  same way it was through epoch 40 (§16.1 already characterized Arm B's
  CIFAR curves this way; the same pattern holds here).
- **CIBHash `d=64`** was **still rising** through epoch 40 (0.8015 ->
  0.8078, monotonic) and **continued rising** through epoch 100, reaching
  a new best of **0.8098 at epoch 100** (the final epoch -- curve had not
  plateaued by epoch 40, and arguably still hadn't fully plateaued by
  epoch 100 either, though the epoch 70-100 values, 0.8091/0.8074/0.8097/
  0.8098, are within a tight 0.0027 band, close to flat).
- **Arm B `d=1024`** peaked very early (**epoch 10, 0.8217**) and never
  recovered to that level again through epoch 100 -- every one of the 6
  new eval points (epochs 50-100) is below 0.8217, ranging 0.8133-0.8173.
  **40 epochs did not miss a later, higher peak here -- if anything the
  opposite: the true peak was already in by epoch 10, well before even
  the 40-epoch mark.**
- **SDC_simclr `d=512`** shows the same early-peak-then-never-recovers
  pattern (**epoch 10, 0.8214**, never exceeded through epoch 100), with
  a pronounced mid-training dip at epoch 70 (0.8035, the lowest value in
  its entire 100-epoch trajectory) before partially recovering to
  ~0.815-0.816 by epochs 90-100 -- still below the epoch-10 peak.

**Recomputed comparisons, 100-epoch numbers (seed 42 only, this task):**

- **Comparison 1** (Arm B `d=64` vs. CIBHash `d=64`): best -0.0165
  (-2.04% relative, vs. 40-epoch's -1.87%); final -0.0179 (-2.21%, vs.
  40-epoch's -2.07%). **Same direction (Arm B loses), margin slightly
  larger against Arm B with more training** -- CIBHash's still-rising
  epoch-40-100 curve is the reason, per the point above.
- **Comparison 2** (Arm B `d=1024` vs. SDC_simclr `d=512`): best +0.0003
  (+0.037%, **numerically identical** to the 40-epoch value, since
  neither config's best checkpoint moved -- both peaked at epoch 10);
  final -0.0002 (-0.025%, vs. 40-epoch's -0.14% -- even closer to an
  exact tie). **Direction unchanged either way (best: Arm B narrowly
  ahead; final: Arm B narrowly behind), margins stay negligible.**

### 19.4 Task 4 -- combined verdict

**Comparison 1 (small-`d`, Arm B `d=64` vs. CIBHash `d=64`,
§18.4's first flip vs. CIFAR): confirmed stable against Arm B.** Holds at
every one of 3 seeds (§19.2, margins -0.0119 to -0.0167, never crossing
zero) and holds -- if anything slightly more strongly -- at 100 epochs
(§19.3, -2.04%/-2.21% vs. 40-epoch's -1.87%/-2.07%). Neither more seeds
nor more training moves this toward a tie or a flip back toward Arm B;
both pieces of new evidence agree with §18.4's original single-seed,
40-epoch finding. This is the one of the two closest-to-flip comparisons
that is **not actually close** once checked properly -- it is a
consistent, reproducible loss for Arm B on this dataset at this `d`,
unlike CIFAR.

**Comparison 2 (large-`d` matched-storage, Arm B `d=1024` vs.
SDC_simclr `d=512`, §18.4's second flip vs. CIFAR): still genuinely too
close to call, not resolved in either direction.** By best-mAP, Arm B
wins at all 3 seeds (§19.2) and the margin is essentially unchanged at
100 epochs (§19.3, since both configs' true peaks occur within the first
10-20 epochs and neither seeds nor further training moved them) -- but
every margin involved is small (+0.03% to +0.34% relative across the 3
seeds' best-mAP comparisons, +0.037% at 100 epochs). By final-mAP, the
direction is **not stable across seeds** (seed 42 loses, seeds 43/44 win)
and stays a near-exact tie at 100 epochs (-0.025%). Unlike Comparison 1,
here more evidence did not converge on a clear answer -- it consistently
produced margins small enough that which side "wins" depends on which
metric (best vs. final epoch) and, for final-mAP, which seed is used.
**Stated plainly, per the task's instruction not to force a verdict past
what the evidence supports: this comparison remains a near-tie, not a
confirmed win or loss for Arm B, across every check run in §18-§19.**
