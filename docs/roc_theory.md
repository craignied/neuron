# ROC AUC and Signal Detection Theory in neuron 3.0

The engine's "statistical" ROC area is the **binormal method** of Thomas D.
Wickens, *Elementary Signal Detection Theory*, Oxford University Press, 2002,
pp. 60-74. This document gives the conceptual background, then maps the theory
onto the implementation.

## The central idea

Suppose you have two populations:

-   **Negative cases**: no disease / no signal
-   **Positive cases**: disease / signal present

Your test, algorithm, or neural network produces a continuous score. The
two populations therefore generate two distributions of scores. The
degree of separation between these distributions is the fundamental
quantity that signal detection theory is trying to describe.

## Where the ROC curve comes from

Place a threshold somewhere along the score axis. Everything to the
right is called positive.

Move the threshold from right to left, and you successively calculate:

-   sensitivity = true-positive rate
-   1 − specificity = false-positive rate

Plot one against the other and you get the ROC curve. The ROC curve
describes what happens as the **decision criterion changes**, while the
underlying ability to discriminate signal from noise remains
conceptually distinct from that criterion.

## The particularly useful part

Wickens provides a theoretical bridge between **separation of
distributions** and **ROC AUC**.

Under the classic equal-variance Gaussian signal-detection model,
discriminability is measured by:

$$
d' = \frac{\mu_1-\mu_0}{\sigma}
$$

where the positive and negative populations are assumed to have equal
standard deviations.

Then:

$$
AUC = \Phi\left(\frac{d'}{\sqrt{2}}\right)
$$

where $\Phi$ is the standard normal cumulative distribution function.

    $d'$   Approximate AUC
  ------ -----------------
       0              0.50
     0.5              0.64
     1.0              0.76
     1.5              0.86
     2.0              0.92
     3.0              0.98

This gives a very intuitive interpretation of AUC:

> **AUC is fundamentally a measure of how well separated the positive
> and negative score distributions are.**

There is an even more useful interpretation:

$$
AUC = P(X_{\text{positive}} > X_{\text{negative}})
$$

In words:

> **AUC is the probability that a randomly selected positive case will
> receive a higher score than a randomly selected negative case.**

That probabilistic interpretation is also the reason AUC is
mathematically related to the Mann--Whitney U statistic.

## Why Wickens might matter for your application

Wickens helps separate **three things that are very easy to conflate**:

### 1. Discriminability

How much information does the model actually contain about the outcome?

This is represented by $d'$, the ROC curve, or AUC.

### 2. Decision threshold

At what score do you declare a case positive?

Changing this threshold changes sensitivity and specificity but **does
not change the underlying AUC**.

### 3. Optimization criterion

Which threshold should you choose?

That depends upon what you value: Youden index, accuracy, cost of false
positives, cost of false negatives, clinical utility, etc.

This distinction is extremely important. A neural network can be trained
to produce a score with better discrimination---that is, a higher AUC.
You can then choose a threshold algorithmically. But **the threshold
itself does not determine the AUC**. The ranking/separation of positive
and negative observations determines the AUC.

## A model-based alternative

If you **do not have enough observations to construct a reliable
empirical ROC curve**, Wickens's framework suggests a model-based
alternative.

If you can reasonably model the positive and negative score
distributions---for example, as Gaussian distributions---you can
estimate their parameters and derive a **theoretical ROC curve and AUC**
from the signal-detection model.

For equal-variance Gaussian distributions:

$$
AUC = \Phi\left(
\frac{\mu_1-\mu_0}
{\sqrt{2}\sigma}
\right)
$$

More generally, for two independent Gaussian score distributions with
unequal variances:

$$
AUC =
\Phi\left(
\frac{\mu_1-\mu_0}
{\sqrt{\sigma_1^2+\sigma_0^2}}
\right)
$$

That may be where *Elementary Signal Detection Theory* becomes genuinely
useful rather than merely explanatory: it gives you a principled way to
think about estimating ROC performance from underlying distributions
rather than simply feeding observed predictions into an ROC/AUC
function.

## How neuron 3.0 implements this

`TwoSet::getStatROCarea()` (`src/twoset.cpp`, which cites Wickens directly):

1. Walk every observed score as a threshold, computing the false-alarm rate
   F = fp/(tn+fp) and hit rate H = tp/(tp+fn) at each — the empirical ROC.
2. Convert each interior point to z-coordinates: `zF = invZarea(F)`,
   `zH = invZarea(H)` (`stats::invZarea` is the inverse normal CDF, built on
   Takuya Ooura's 1996 inverse-erfc).
3. Fit the **zROC line** `zH = a + b·zF`. With enough points (≥ `binThresh`),
   the z-points are grouped into bins; each bin contributes its mean and
   standard deviation, and the line is fit with *errors in both coordinates*
   (`XY` in `src/stats.cpp`, Numerical Recipes' fitexy), yielding a χ² and
   p-value for the fit (`getStatChi2()`, `getStatP()`).
4. Report **Az = Φ( a / √(1 + b²) )** via `stats::Zarea`.

The equivalence to the Gaussian formulas above: under the binormal model the
zROC intercept is a = (μ₁−μ₀)/σ₁ and the slope is b = σ₀/σ₁, so

    Φ( a / √(1+b²) ) = Φ( (μ₁−μ₀) / √(σ₀²+σ₁²) )

— the unequal-variance AUC exactly. The equal-variance d′ case is b = 1.

When there are too few data points for the statistical method (< `calcThresh`,
default 10), the engine reports the empirical **trapezoidal** area instead —
that is the "Cannot calculate ROC statistically" message followed by
"By trapezoidal method, ROC area = ...".

## Confidence intervals on the ROC area

Every ROC area the engine prints carries a 95% confidence interval, by an
**analytic (parametric)** method in both cases — neither is a bootstrap:

- **Binormal Az → delta method.** The z-ROC line fit yields the intercept *a*
  and slope *b* with standard errors; since Az = Φ(a/√(1+b²)) is a smooth
  function of those two parameters, its first-order Taylor expansion propagates
  the parameter variances through the analytic gradient (`TwoSet::azSE`).
- **Trapezoidal area → Hanley–McNeil.** The closed-form variance for the
  empirical area, from its equivalence to the Mann–Whitney U statistic
  (Hanley & McNeil, *Radiology* 1982; `TwoSet::hmSE`).

### Methods-section language

> A 95% confidence interval for the binormal area under the ROC curve (Az) was
> obtained by the delta method, propagating the standard errors of the fitted
> z-ROC intercept and slope through Az = Φ(a/√(1+b²)). For empirical
> (trapezoidal) areas, the confidence interval used the Hanley–McNeil variance
> estimator based on the equivalence of the area to the Mann–Whitney U
> statistic.

### Caveat before relying on the binormal interval in print

The binormal delta-method interval has a known approximation in this
implementation: on the binned fit path (fitexy, errors in both coordinates),
the routine doesn't expose the a–b covariance, so the cross term is set to
zero. In the calibration test the delta-method SE ran a bit narrow (reported
SE ≈ 0.56× the empirical SD) — i.e., somewhat anti-conservative. The
Hanley–McNeil interval, by contrast, calibrated cleanly (ratio ≈ 1.09). So the
trapezoidal CI is the more trustworthy of the two as it stands; recover that
covariance term, or validate the binormal CI further, before leaning on it in
print. Both intervals' calibration is checked by simulation in
`tests/binormal/check_az.cpp` (the delta method under a generous window, the
Hanley–McNeil under a tight one).

**The delta-method SE scales with the bin count, not the sample size**
(observed 2026-07-15). Because the z-ROC line is fitted to *binned* points, the
delta SE reflects the uncertainty of a fit through a handful of bin means and
never sees the number of exemplars behind them. Two consequences, both real:

- Across datasets it barely moves with n. On the low-birth-weight set, n = 142
  and n = 47 gave SEs of 0.0102 and 0.0103 — while Hanley–McNeil correctly gave
  0.049 and 0.091, a ratio of 1.85 against the expected √(142/47) = 1.74.
- Within one dataset it moves with the binning alone. The same 142 exemplars
  gave SE 0.014 at 9 bins and 0.034 at 5 bins.

So a binormal Az is only interpretable **alongside the nBins that produced it**,
which is why `ROCarea` and the GUI stats panel both label every fit with its bin
count, and why `TwoSet::getBestPfit()`/`getBestAUCfit()` return `nBins` as part
of the `ROCfit`. Quote the bin count whenever you quote the Az. This makes the
"0.56× narrow" figure above a floor rather than a fixed correction: how narrow
depends on how many bins the search happened to land on.

**This is tested.** `tests/binormal/check_az.cpp` draws large Gaussian samples
with known (μ, σ), runs them through `getStatROCarea()`, and requires the
result to match Φ((μ₁−μ₀)/√(σ₀²+σ₁²)) within sampling tolerance — including an
unequal-variance case. It runs in CI on every push (`ctest`).

### Provenance

The implementation entered the neUROn2++ lineage via the standalone Windows
"roc" application (`code/C++/msvc/roc`, 2001-2005), an MFC GUI for ROC analysis
of assay data, and was merged into the engine before the 2.6 era. The engine's
copy is the maintained one (it carries the 2.6.x and 3.0 fixes); the roc app
additionally collected the ROC curve coordinates for plotting (`ROCx`/`ROCy`),
which is planned for porting when the 3.0 GUI needs to draw curves.
