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

1. Walk each **distinct** observed score as a threshold, computing the false-alarm
   rate F = fp/(tn+fp) and hit rate H = tp/(tp+fn) at each — the empirical ROC.
   One point per distinct threshold: a tied score repeated across exemplars gives
   the *same* operating point, and replicating it adds no information.
2. Convert each interior point to z-coordinates: `zF = invZarea(F)`,
   `zH = invZarea(H)` (`stats::invZarea` is the inverse normal CDF, built on
   Takuya Ooura's 1996 inverse-erfc).
3. Give each point Wickens' **binomial error bar**: a rate p is binomial with
   variance p(1−p)/N (Eq. 11.2, p. 202), and the delta method carries that
   through the z transform (Eq. 11.3, p. 202):

       σ²_z ≈ p(1−p)/N ÷ φ²(z)

   Points near 0 or 1 sit where the normal density φ is tiny, so they earn a
   huge error bar and little weight — which is right, and is the down-weighting
   no within-bin standard deviation could ever express.
4. Fit the **zROC line** `zH = a + b·zF` with *errors in both coordinates*
   (`XY` in `src/stats.cpp`, Numerical Recipes' fitexy), yielding a χ² and
   p-value for the fit (`getStatChi2()`, `getStatP()` — but read
   "What the fit χ² is worth" below before quoting them).
5. Report **Az = Φ( a / √(1 + b²) )** via `stats::Zarea`.

There is **no binning**. Steps 3–5 replaced it on 2026-07-15; the account of what
binning was and what it cost is kept below, because it is the reason this design
is what it is.

The equivalence to the Gaussian formulas above: under the binormal model the
zROC intercept is a = (μ₁−μ₀)/σ₁ and the slope is b = σ₀/σ₁, so

    Φ( a / √(1+b²) ) = Φ( (μ₁−μ₀) / √(σ₀²+σ₁²) )

— the unequal-variance AUC exactly. The equal-variance d′ case is b = 1.

When there are too few data points for the statistical method (< `calcThresh`,
default 10), the engine reports the empirical **trapezoidal** area instead —
that is the "Cannot calculate ROC statistically" message followed by
"By trapezoidal method, ROC area = ...".

**The trapezoidal area is exact, not a grid approximation** (since 2026-07-20).
`getTrapROCarea()` integrates trapezoids over the *same* operating points as
step 1 above — one per distinct score, cumulated in one sorted pass
(`operatingPoints()`) — so it is the exact non-parametric AUC, equal to the
Mann–Whitney U statistic divided by n₀·n₁ (ties contributing half-steps). That
is precisely the quantity whose variance the Hanley–McNeil estimator describes,
so the area and its interval now rest on the same object by construction.
Before 2026-07-20 it swept a fixed count of evenly spaced thresholds
(`nThresholds`, a user-settable parameter) and only approximated this area —
the parameter, its "Number thresholds" report line, and its CLI/GUI controls
are gone. The curve capture (`ROCx`/`ROCy`, the GUI's ROC plot) runs
(0,0) → (1,1) in ascending false-alarm rate.

## Confidence intervals on the ROC area

**Read this section before quoting an interval in print.** It records a 2026-07-15
audit that traced every ROC interval back to Wickens page by page, found the
original delta-method interval to be mis-specified, and replaced it. The
citation trail is preserved because the reasoning is not reconstructible from
the code alone.

### What kind of data neuron actually has

This distinction drives everything below, and getting it wrong is what produced
the defective interval.

Wickens describes two ways to obtain several points on one isosensitivity
contour:

- **Several independent bias conditions** (his Figure 4.1, p. 61): four separate
  conditions, 200 signal + 200 noise trials each. Each point comes from *its own*
  sample, so the points are **statistically independent**.
- **A single condition with graded (rating) responses** (§5.1–5.2, pp. 83–87):
  one sample, several criteria cutting the evidence axis at once. The points are
  formed by *cumulating* the same observations, so they are **not independent**
  (Wickens states this explicitly, pp. 87–88).

neuron sweeps ~100 thresholds across one continuous classifier score on one
sample. That is structurally the **rating case**, not the bias case. Wickens
ships two different programs for exactly this reason — FitRoc for independent
conditions, FitRating for rating data — and says to be sure to use the right one
for one's data (§5.3, p. 88).

Two consequences he states directly:

1. The dependence must be accommodated **in the fitting algorithm** (§5.3, p. 88).
2. The dependence **increases** the sampling variability of the parameter
   estimates relative to independent points (pp. 87–88).

So an interval computed as if the points were independent is not merely
imprecise — it is biased **narrow**, and Wickens says so before we ever measured
it.

### The estimator is faithful; the interval was not

The point estimate needs no apology. Wickens fits the same line to the same
cumulated z-points *by eye* in Example 5.1 (p. 89), obtaining
zH = 0.735·zF + 0.974, converts it to μ̂ₛ = a/b = 1.325 and σ̂ₛ = 1/b = 1.360
(p. 90), and reports Az = Φ(a/√(1+b²)) = 0.784 — then carries those very
estimates into the goodness-of-fit test of Example 11.8 (p. 214). He remarks
that a by-eye fit is nearly as accurate as a computer program (p. 89).
`getStatROCarea()` computes Az by his Eq. 4.7 (p. 68). That path is correct.

What was wrong was everything downstream of it:

- **The delta-method interval assumed independence.** It propagated the z-ROC
  fit's parameter SEs through Az, which is only valid if the fitted points are
  independent observations. They are cumulated (§5.2–5.3). No amount of
  recovering the a–b covariance term fixes a correlation structure that is
  absent from the model.
- **It was symmetric.** Wickens is explicit (§11.4, p. 206) that estimate ± z·se
  (his Eq. 11.9) is correct only when the standard error does not vary with the
  parameter, and that neither d′ nor Az has a constant SE (Figure 11.1, p. 205);
  the asymmetry for Az runs *opposite* to that for d′. His analytic remedy is to
  iterate — re-evaluate the SE at the interval endpoints and recompute (p. 207).
  The implementation instead clamped a symmetric interval to [0,1].
- **Analytic intervals run short anyway.** Real data are overdispersed relative
  to the multinomial, so calculations based on it underestimate variability and
  confidence intervals come out too short (§11.2, p. 201). Wickens calls such
  standard errors optimistically too small (p. 210).
- **Wickens never offered an analytic SE for this fit.** His closed forms
  (Eqs. 11.2–11.7, pp. 202–203) apply where the parameter count equals the data
  point count — the single-point case. Where data exceed parameters (rating data,
  several bias conditions) he routes estimation to the §3.6 fitting algorithms,
  which supply approximate sampling variances from the product-multinomial model
  (p. 205). Those algorithms are maximum likelihood, and he names the standard
  one: Dorfman & Alf (§3.6, pp. 57–58). **The SE was always meant to come out of
  the fit itself.** There is no sanctioned analytic interval for a least-squares
  z-ROC line, which is why the delta method had to be invented and why it failed.

Measured on the Hosmer–Lemeshow low-birth-weight data, the delta SE was ~5×
too narrow (≈0.010 where the bootstrap and Hanley–McNeil both give ≈0.05),
and it did not scale with n at all — see "The binning artifact" below.

### What neuron does now: bootstrap

The binormal Az interval is a **stratified bootstrap percentile interval**.
Exemplars are resampled with replacement within each class (so every resample
preserves the observed class sizes and always has an ROC), the **entire Wickens
procedure is re-run on each resample**, and the 2.5th/97.5th percentiles of the
resampled areas form the interval; the SD of the resampled areas is reported as
the standard error.

Why this is the right answer rather than a convenience:

- Resampling **cases** reproduces the cumulation-induced dependence
  automatically. That is precisely the accommodation §5.3 (p. 88) demands, without
  having to model the correlation.
- The percentile interval is **asymmetric by construction** — it delivers for
  free what Wickens' endpoint iteration (p. 207) exists to approximate.
- It does not rely on the multinomial variance that §11.2 (p. 201) warns is
  optimistic.

Implementation notes: draws come from `util::i_resample()`, a generator kept
**independent of the training stream**, so computing an interval can never
perturb weight initialisation or train/test splits; `--seed N` seeds both.
Resamples on which the fit cannot be computed are counted and reported rather
than silently dropped. B = 2000 by default (`setBootstrapResamples`; 0 disables).

**Validation.** Two independent lines of evidence:

- On the low-birth-weight data the bootstrap SE agrees with the independent
  Hanley–McNeil estimate to within ~3% (train: 0.0524 vs 0.0522 at n = 142; test:
  0.0844 vs 0.0870 at n = 47), and scales with n as it must. Two methods resting
  on different assumptions agreeing is the calibration evidence; the delta method
  agreed with neither. (Before the variance fix these read 0.0497 vs 0.0522 —
  the residual gap on the training set *was* the discard bias.)
- Against simulation, `tests/binormal/check_az.cpp` draws replicate samples from
  known Gaussians and compares the mean reported bootstrap SE with the empirical
  SD of A_z across those replicates: **ratio 1.00**. The delta method it replaced
  needed a tolerance window of 0.4–2.5 to pass the same check.

The trapezoidal area keeps its **Hanley–McNeil** closed-form interval (the
Mann–Whitney U equivalence; Hanley & McNeil 1982). It calibrated cleanly
(ratio ≈ 1.09) and is retained as an independent cross-check.

### What the fit χ² is worth (read before quoting it)

The χ² and its p are a **weak diagnostic on continuous data, and no more**. With
the binomial error bars in place, a swept continuous score gives χ² far *below*
its degrees of freedom — on the low-birth-weight training set, χ² = 15.1 against
127 points (df = 125), so p rounds to 1.000. That is not a perfect fit; it is the
cumulation dependence showing through. Adjacent operating points differ by one
exemplar, so the scatter of the points *about the line* is far smaller than each
point's own marginal sampling error, which is what the error bar describes.
Wickens says the dependence invalidates such calculations (p. 212).

Where the χ² **is** meaningful is the rating case with few categories: on Wickens'
Table 5.1 (six categories, five points) the engine reports χ² = 1.93 on 3 df,
p = 0.587 — a sensible goodness of fit, of the same character as the one Wickens
himself computes for these data in Example 11.8 (p. 214), though not the same
statistic (see `tests/binormal/check_wickens.cpp` on why his is not implementable
here). Quote it in that regime; read nothing into a p of 1.000 from a continuous
sweep. The area does not depend on it either way (§11.5, p. 217).

### The binning artifact — removed 2026-07-15, kept here as the reason why

**This section is history.** It is retained because it is the entire justification
for the design above, and because the measurements in it are the evidence that the
replacement works.

Until 2026-07-15, `getStatROCarea()` grouped the z-points into fixed-count bins and
fed fitexy each bin's **mean and within-bin standard deviation** as the point and
its error bar. **This had no counterpart in Wickens.** Table 5.3 (p. 90) carries no
error-bar column; Example 5.1 fits by eye. The within-bin SD was invented to give
fitexy something to weight with — the rating case supplies no independent
per-point sample the way the bias case does.

Because the bins were chunks of adjacent values from a sorted, monotone z
sequence, that "standard deviation" measured **bin width**, not sampling error.
Consequences, all measured 2026-07-15 before the fix:

- The delta SE tracked the **bin count**, not the sample size: n = 142 and n = 47
  gave 0.0102 and 0.0103, while Hanley–McNeil correctly gave 0.049 and 0.091.
- Within one dataset it moved with the binning alone: the same 142 exemplars gave
  SE 0.014 at 9 bins and 0.034 at 5 bins.
- Flat runs of the empirical ROC (thresholds where the hit rate does not move)
  give a bin whose z values are all **identical**. Their standard deviation is
  legitimately zero, and fitexy handles a zero error bar by construction
  (`chixy` weights such a point `BIG`). What did not handle it was
  `Population::var()`, which computed the variance by the one-pass
  `(Σx² − n·x̄²)/(n−1)`: for identical values those two terms agree to within
  roundoff, so the result came out at ±1e-16 with an essentially **random sign**.
  A negative one made `std()` return `sqrt` of a negative number — NaN — which
  propagated into the fit and surfaced as the legacy `gammq` failure
  ("a too large, ITMAX too small in gcf"); a positive one silently produced a
  **fake error bar of ~3e-08**, weighting that point ~1e15 in a fit that
  reported no error at all. This was fixed 2026-07-15 at its source (the
  two-pass form, which returns the exact zero), not at the symptom. It had been
  discarding **27% of bootstrap resamples** on the low-birth-weight training set
  — non-randomly, since it tracks ties — and the discards biased the interval
  narrow: removing them widened the 95% CI from 0.513–0.711 to 0.513–0.721 and
  moved the bootstrap SE from 0.0497 to 0.0524, against Hanley–McNeil's 0.0522.
  See "Goodness of fit does not gate the area" below for the second, independent
  guard added at the same time.

**Measured against the literature** (`tests/binormal/check_wickens.cpp`, 2026-07-15).
Wickens' own Table 5.1 (p. 84) has a published answer, A_z = 0.784 (p. 90), so the
binning artifact can be sized rather than argued:

| bins | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| A_z | .7815 | .7789 | .7785 | .7783 | .7893 | .7828 | — | — |
| fit p | .909 | .976 | .995 | .000 | .000 | .000 | *no fit* | *no fit* |

The bin count alone moves A_z by **0.011** — the size of the entire bootstrap SE
(0.0142) — around a truth of 0.784. And the two selection criteria both choose
badly: **best p** takes 5 bins (0.7785, among the furthest), **best AUC** takes 7
bins (0.7893, the furthest) on a fit whose p is 0.000 — maximising the area selects
for a poor fit by construction. The near-exact binning (8 bins, −0.0012) is one
neither criterion picks. The engine's *operating points*, by contrast, reproduce
Wickens' Table 5.3 exactly — the error is entirely in what the binning does to them.

The fit is also **numerically fragile in this regime**: the same source compiled at
-O0 and -O3 differs in the fourth decimal of A_z, and at 10 bins one build converges
where the other cannot bracket a root. Bins of identical z values have SD zero, which
`chixy` weights `BIG` = 1e30, so the objective is dominated by 1e30 terms and `brent`
balances on the last bits.

**The fix, landed 2026-07-15.** The principled error bar is Wickens' own: binomial
rate variance (Eq. 11.2, p. 202) pushed through the delta method (Eq. 11.3, p. 202)
to the z scale, σ²_z ≈ p(1−p)/N ÷ φ²(z). Once each point carries an analytic error
bar, **binning is unnecessary** — its only purpose was to have several values to
take an SD of. Sweeping distinct thresholds and fitting those points directly gives,
on Wickens' own data, **A_z = 0.7839 against his published 0.784**: the bin count no
longer moves the answer at all (it is now inert, asserted as a null test in
`check_wickens`), and the estimate is ~50× closer than either binned criterion
managed. The zero-SD/`BIG`-weight fragility goes with it, since no error bar is zero
any more.

Metz's LABROC categorisation — the natural categories of continuous data are the
**corners of the empirical ROC** (Metz, Herman & Shen 1998) — turned out **not to be
needed** for this estimator, and is not implemented. Its purpose is to categorise for
a maximum-likelihood fit; a least-squares line over the distinct operating points
needs no categories at all. It would return with ML (Phase 3).

Wickens independently condemns the small-expected-frequency regime that
fixed-count binning created: distrust these χ² tests when many expected
frequencies fall below four or five (§11.5, p. 216) — another reason the binning
had to go rather than be tuned.

### Errors in both coordinates

The choice to fit with errors in *both* z-coordinates (fitexy) is Wickens' own
and worth keeping. He calls the ordinary regression model — x exact, all
uncertainty in y — unsatisfactory for detection data, because both Z(h) and Z(f)
carry sampling error (p. 56). Regression to the mean flattens such a line, giving
too small a slope b and hence an overestimated σ̂ₛ, which with enough scatter can
get the equal-variance model rejected when it is appropriate.

Note the limit of that endorsement: he invokes the both-coordinates line to show
why regression is biased, then routes three-or-more-point data to maximum
likelihood (§3.6, p. 57). The 2-axis fit is better than regression; it is not the
estimator he recommends.

### Goodness of fit does not gate the area

The χ² fit probability is a **diagnostic**, never a precondition for reporting
Az. Wickens states that even had the model been strictly rejected on goodness of
fit, it would still be appropriate to use statistics like Az to describe
performance (§11.5, p. 217). A `gammq` failure or a poor fit therefore marks the
p-value unavailable ("p = not available") and leaves the area intact.

This is implemented as a guard, not as the fix for the discards described above:
the failing `gammq` calls were a *symptom* of a NaN already in the fit, and had
they merely been suppressed the engine would have reported a NaN area instead of
raising. The fix is at the variance; this rule is what makes an unconvergeable
fit probability cost the p-value alone — the report prints "p = not available"
and the area stands. (While the bin search existed, the same principle applied
at its level too: a failing binning was skipped rather than suppressing the
whole report and its bootstrap resample. The search is gone — see "The binning
artifact" — so only the fit-level rule survives.)

### Methods-section language

Every methodological claim below is cited to its source and is a property of the
*method*, so it travels to any dataset. Three things do **not** travel — fix them
before this goes near a manuscript:

1. **Check what your build reports.** ROADMAP 3 Phase 1 is complete as of
   2026-07-15: the bootstrap interval landed, and the resample discards that
   biased it narrow (~27% on the low-birth-weight training set) were traced to a
   variance bug and eliminated — a current build reports 2000 of 2000 resamples
   on that data. The engine still prints the effective resample count and the
   failure count: **report them**, and if failures are more than a few percent,
   say so — failures track ties rather than occurring at random, so they bias the
   interval narrow. A build *predating* 2026-07-15 reports the delta-method
   interval, which is mis-specified: do not publish an A_z interval from one; use
   the trapezoidal Hanley–McNeil interval. A build from 2026-07-15 but before the
   variance fix reports a bootstrap interval with a large failure count and is
   narrow by roughly the amount shown above. Describe the code you actually ran.
2. **"B = 2000 resamples"** is the default (`setBootstrapResamples`). If you
   changed it, change it here.
3. **"the two interval methods agreed to within ~5%"** is a measurement on the
   Hosmer–Lemeshow low-birth-weight data (0.0497 vs 0.0522 at n = 142; 0.0844 vs
   0.0870 at n = 47). It is **not** a property of the method and **not** a
   guarantee. Recompute it on your data or delete the sentence. Agreement can fail
   — the two estimators rest on different assumptions, which is the entire reason
   the comparison is worth reporting.

Then read **"What to disclose, honestly"** immediately below the block: it lists the
limitations a reviewer is entitled to know about, and they belong in the paper too.

> Discrimination was summarised by the area under the ROC curve. The binormal
> area A_z was estimated under the unequal-variance Gaussian signal-detection
> model following Wickens (2002): the empirical operating points were transformed
> to Gaussian coordinates, a straight isosensitivity contour zH = a + b·zF was
> fitted, and the area computed as A_z = Φ(a/√(1+b²)) (Wickens 2002, Eq. 4.7,
> p. 68). Because the operating points derive from a single sample and are formed
> by cumulating the same observations, they are not statistically independent
> (Wickens 2002, pp. 87–88); the line was therefore fitted treating both
> coordinates as subject to sampling error (Wickens 2002, p. 56; Press et al.
> 1992, fitexy), and no analytic standard error was derived from the fit.
>
> A 95% confidence interval for A_z was obtained by a stratified bootstrap
> percentile method (Efron & Tibshirani 1993) with B = 2000 resamples. Exemplars
> were resampled with replacement within each outcome class, preserving the
> observed class sizes, and the complete estimation procedure was re-run on each
> resample, so that the dependence among operating points is reproduced in the
> resampling distribution rather than modelled. Resampling was chosen over an
> analytic interval because the parametric variance underlying such intervals
> assumes independent operating points and, being multinomial-based, underestimates
> variability in overdispersed data (Wickens 2002, §11.2, p. 201, and p. 210), and
> because a symmetric estimate ± z·se interval is inappropriate for A_z, whose
> standard error varies with the parameter (Wickens 2002, §11.4, pp. 206–207).
> The percentile interval is asymmetric by construction.
>
> The empirical (trapezoidal) area was reported alongside A_z with a 95% interval
> from the Hanley–McNeil variance estimator, based on the equivalence of the
> empirical area to the Mann–Whitney U statistic (Hanley & McNeil 1982). The
> trapezoidal area is known to be negatively biased, underestimating the true area
> when operating points are few or unevenly spread (Wickens 2002, pp. 70–71); A_z
> is therefore the primary measure (Wickens 2002, p. 72), with the trapezoidal
> area serving as a distribution-light cross-check. On the present data the two
> interval methods agreed to within ~5%. **[← RECOMPUTE ON YOUR DATA OR DELETE —
> this figure is from the low-birth-weight set, not a property of the method]**
>
> Goodness of fit of the binormal model was assessed by a χ² statistic on the
> fitted contour; a poor fit was reported as a diagnostic but did not preclude
> reporting A_z (Wickens 2002, §11.5, p. 217).

If the analysis instead uses a maximum-likelihood binormal fit (see the roadmap),
substitute: *"A_z and its standard error were estimated by maximum likelihood
under the binormal model (Dorfman & Alf 1968; Metz, Herman & Shen 1998), which
accommodates the dependence among operating points within the estimation
algorithm (Wickens 2002, §3.6, p. 57)."*

### What to disclose, honestly

- The estimator is a **least-squares z-ROC line**, not maximum likelihood.
  Wickens routes multi-point data to ML (§3.6, p. 57) principally because ML
  supplies the standard errors — a need the bootstrap meets independently. The
  least-squares route is Chapter 5's own hand method (Example 5.1, p. 89) and
  Wickens carries its estimates into his own goodness-of-fit test (p. 214), so it
  is defensible; it is not the reference estimator. It is also validated here
  against his published A_z to ±0.0001 (`check_wickens`).
- **The fitted curve is "improper"** and nothing guards against it. Any binormal
  curve with slope b ≠ 1 crosses the chance line and hooks (Metz & Pan 1999).
  Measured on Wickens' own fit (a = 0.970, b = 0.729): H falls below F beyond
  F ≈ 0.9999 — far outside any observed operating point, so it does not affect
  A_z on data like this. It matters when b is far from 1 or the data are sparse
  or lopsided. PROPROC is the remedy, and it needs ML (see the roadmap's Phase 3).
- **Quote the operating-point count with the area** (the report's "Operating
  points fitted" line): an A_z from five points and one from five hundred are
  different claims. (Before 2026-07-15 the equivalent advice was to quote the
  bin count, because the binning was arbitrary and moved A_z — worth 0.011 on
  Wickens' own data. The binning is gone; the point count is what remains to
  disclose.)
- The bootstrap resamples exemplars, so it captures sampling variability of the
  cases. It does not capture model-selection variability from choices made
  outside the resampled procedure.

### References

| Source | Used for |
|---|---|
| Wickens, T. D. (2002). *Elementary Signal Detection Theory*. Oxford University Press. | The model and estimator throughout. Az = Φ(a/√(1+b²)) Eq. 4.7 p. 68; Az = Φ(d_a/√2) Eq. 4.8 p. 68; A_trap Eq. 4.9 p. 69 and its negative bias pp. 70–71; A_z preferred p. 72; errors in both coordinates p. 56; ML for 3+ points and ML supplies SEs §3.6 p. 57; rating model §5.2 pp. 85–87, Eqs. 5.3–5.5 p. 86; rating points not independent, dependence increases sampling variability pp. 87–88; fitting algorithm must accommodate dependence, use the correct program §5.3 p. 88; worked fit Example 5.1 p. 89 and Table 5.3 p. 90; seven parameters and GOF p. 91; overdispersion → intervals too short §11.2 p. 201; binomial rate variance Eq. 11.2 p. 202; delta method Eq. 11.3 p. 202 and σ²_z = σ²_p/φ²(z) p. 202; var(Az) Eq. 11.7 p. 203; large-sample caveats p. 203; se(Âz) worked Example 11.1 p. 204; SE varies with parameter Figure 11.1 p. 205; closed forms limited to parameters = data points, else use §3.6 algorithms p. 205; symmetric interval only valid for constant SE, endpoint iteration §11.4 pp. 206–207; SEs optimistically too small p. 210; dependence invalidates SEs p. 212; GOF df Eq. 11.12 p. 213; Example 11.8 X² = 5.93 vs 35.85 p. 214; distrust GOF at small expected frequencies p. 216; rejected model still permits A_z p. 217. |
| Dorfman, D. D., & Alf, E. (1968a, 1968b) | The standard ML algorithm for fitting the SDT model to several conditions, as cited by Wickens' reference notes, p. 58. (Commonly cited elsewhere as Dorfman & Alf 1969, *J. Math. Psych.* 6:487–496; Wickens gives 1968a/1968b.) |
| Hanley, J. A., & McNeil, B. J. (1982). *Radiology* 143:29–36. | Closed-form variance of the empirical (trapezoidal) area via the Mann–Whitney U equivalence; `TwoSet::hmSE`. |
| Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. | Bootstrap percentile intervals. |
| Metz, C. E., Herman, B. A., & Shen, J. H. (1998). *Statistics in Medicine* 17:1033–1053. | LABROC4/LABROC5; truth-state runs in rank-ordered data as the natural categorisation of continuous data, retaining the ROC's corners without loss of information relevant to ROC estimation. Considered for Phase 2 and found unnecessary — its purpose is to categorise for a maximum-likelihood fit, and the least-squares line over distinct operating points needs no categories. Returns with ML (Phase 3), or not at all. |
| Metz, C. E., & Pan, X. (1999). | The "proper" binormal model; degeneracy (fits of unphysical shape crossing the chance line) and its remedy. |
| Pesce, L. L., & Metz, C. E. (2007). | Reliable, computationally efficient ML estimation of proper binormal ROC curves. |
| Hsieh, F., & Turnbull, B. W. (1996). *Annals of Statistics* 24:25–40. | Generalized least squares for grouped data; minimum-distance estimator that avoids grouping entirely. |
| Swets, J. A. (1996). *Signal Detection Theory and ROC Analysis in Psychology and Diagnostics*. Lawrence Erlbaum. | Summary of published ROC-fitting programs, per Wickens' reference notes p. 58 (print only; not digitised). |
| Chakraborty, D. P. *Modeling ROC Data*, ch. 6. | Modern critique of least-squares z-ROC fitting (non-independence of cumulative points; error in the x-direction; rigid thresholds) and of degeneracy. Note its first objection targets *cumulative rating-type* data — it does not apply to Wickens' independent-bias-condition design. |
| Press, W. H., et al. (1992). *Numerical Recipes in C*, 2nd ed. | `fit` p. 665, `fitexy` pp. 668–670 (errors in both coordinates), `gammq`; the basis of `XY` in `src/stats.cpp`. |
| Metz ROC software, University of Chicago. | ROCFIT, LABROC4, PROPROC, CORROC, INDROC, ROCKIT, LABMRMC, DBM-MRMC — the living descendant of Swets' program summary. |
| Wickens' supplementary programs, `twickens.bol.ucla.edu/sdt.htm` (still live; last modified 11 April 2002). | GAUSSW, DPRIMEW, and — the relevant pair — **FitRoc** (a series of independent detection conditions) and **FitRating** (a single condition with rating-scale responses). Their existence as two separate programs is the clearest statement that neuron's data type has its own fitting algorithm. neuron's case is FitRating. |

### What is tested, and what is not

**`binormal_seed42` is the golden that covers this page.** It runs the
Hosmer–Lemeshow low-birth-weight data (189 rows, split 142/47) through a seeded
logistic fit and freezes the whole printed ROC report byte-for-byte: the A_z
with its operating-point count, the χ² and fit p, the bootstrap interval, and
the Hanley–McNeil interval, on both the training and test sets. Its resolution
is the report's own 6 decimals: a 0.01% change in A_z is caught, 1e-7 is not.

It exists because the other two goldens **do not reach this code at all**:
`xor_seed42` and `regress_seed42` have too few exemplars (`goodData <
calcThresh`), so they print "Cannot calculate ROC statistically" and contain
**zero** "By statistical method" lines; `verify_oracle.sh` likewise. That is
exactly how the delta method came to be replaced with every invariant green.
Demonstrated, not assumed: perturbing A_z by 0.01% fails `binormal_seed42` while
both older goldens still pass.

The rest of the coverage:

`tests/binormal/check_az.cpp` (ctest, runs in CI on every push):

- draws large Gaussian samples with known (μ, σ), runs them through
  `getStatROCarea()`, and requires the result to match Φ((μ₁−μ₀)/√(σ₀²+σ₁²))
  within sampling tolerance — including an unequal-variance case;
- asserts the variance of a set of identical values is zero rather than NaN or a
  roundoff residue (the bug behind the resample discards, asserted at its root);
- drives deliberately **tied** scores — the flat ROC runs that used to make the
  binning degenerate — and requires the fit to yield a finite area and a finite
  fit p, and the bootstrap to discard **no** resample and stay within 25% of
  Hanley–McNeil;
- calibrates the bootstrap SE against the empirical SD of A_z over replicate
  samples.

`tests/gui/smoke.sh` asserts the panel's binormal block is `null` when no fit is
possible (four exemplars) and carries the fit — with its operating-point count
and bootstrap interval — on the low-birth-weight data, which is large enough to
reach the statistical path. Both assertions were verified to fail against the
pre-fix binary.

**When Phase 2 moved A_z (2026-07-15), `binormal_seed42` was the record of what
moved** — its re-bless diff (commit `c219b17`) is where the change is visible
line by line, and it was re-blessed only after the literature acceptance test
(Wickens Table 5.1) passed. The same discipline applies to any future change
that moves the area: literature test first, then read the golden's diff.

### Provenance

The implementation entered the neUROn2++ lineage via the standalone Windows
"roc" application (`code/C++/msvc/roc`, 2001-2005), an MFC GUI for ROC analysis
of assay data, and was merged into the engine before the 2.6 era. The engine's
copy is the maintained one (it carries the 2.6.x and 3.0 fixes); the roc app
additionally collected the ROC curve coordinates for plotting (`ROCx`/`ROCy`),
ported into the engine 2026-07-13 (`getTrapROCarea()` fills them; the GUI's
ROC plot draws them).
