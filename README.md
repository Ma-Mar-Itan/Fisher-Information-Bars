# FIBars &mdash; Fisher Information Bars

**A new answer to the oldest question in quantitative finance: what is the right unit of observation?**

---

## The Problem with Every Bar You've Ever Used

Time bars assume markets produce information at a constant rate. They don't.

Tick bars assume each transaction carries equal statistical weight. It doesn't.

Volume and dollar bars are better — but they standardize the *mechanics* of exchange, not the *inferential content* of the data. A dollar bar in a quiet market and a dollar bar during a volatility spike both contain the same notional value. They do not contain the same amount of *knowledge*.

The downstream consequences are real: heteroskedastic inputs to ML models, unstable covariance estimates, sampling sequences where adjacent observations vary by orders of magnitude in statistical precision. These aren't preprocessing annoyances — they're structural biases baked into the training data of almost every systematic strategy in production.

---

## The Proposal

**fibars** implements Fisher Information Bars (FIBs) — a bar construction protocol grounded in information geometry rather than market mechanics.

The core idea: a bar should close when the data accumulated within it constitutes a *completed inferential unit* — a sampling interval representing constant metric volume on the statistical manifold of the local market model. The relevant criterion is not elapsed seconds, ticks traded, or dollars exchanged, but the curvature of the log-likelihood surface.

Formally, a bar closes at the first event $\tau_k$ such that the accumulated Fisher Information crosses a target quantum $\mathcal{I}^*$:

$$\tau_k = \inf\left\{t > \tau_{k-1} : \Phi\!\left(\,\mathcal{I}_k(t)\,\right) \geq \mathcal{I}^*\right\}$$

where the cumulative information path is built recursively from the outer product of score vectors — an O(1) update that imposes no computational overhead on a live tick feed:

$$\mathcal{I}_k(t) = \sum_{i=\tau_{k-1}+1}^{t} \nabla\ell_i(\hat\theta_{i-1})\;\nabla\ell_i(\hat\theta_{i-1})^\top$$

The result: a bar series where the estimation precision is, by construction, approximately constant across observations. Each bar carries the same amount of statistical information about the local market regime. The sampling sequence is no longer contaminated by the latent state of volatility or liquidity.

---

## Why This Is Non-Trivial

Connecting Fisher Information to financial sampling requires resolving several open problems simultaneously. This library addresses them:

**The scalarization problem.** The Fisher Information accumulates as a matrix $J \in \mathbb{R}^{p \times p}$. Collapsing it to a scalar threshold requires a choice of functional. The theoretically principled choice is the log-determinant — it corresponds to the volume element of the Riemannian metric and is invariant to smooth reparameterizations of the model. Trace and Frobenius norms are provided as alternatives, with known A-optimality and entry-wise norm interpretations respectively.

**The adaptive threshold problem.** A static Information Quantum would produce bars of wildly varying frequency as the market cycles through regimes. The threshold must adapt to the asset's long-run information rate. The solution is an EWMA estimator of the average information rate $\bar\lambda_\Phi$ across completed bars, yielding a target $\mathcal{I}^* = \eta \cdot \bar\lambda_\Phi \cdot \Delta_0$ that breathes with the market while maintaining constant inferential precision.

**The initialization problem.** Before any bar has completed, the EWMA rate is undefined and the threshold lives at $+\infty$ — a deadlock. The engine breaks this cycle by seeding the threshold from the first observed scalar, ensuring the first bar closes and the EWMA can begin accumulating.

**The model selection problem.** The information content of an observation depends entirely on the assumed local model. This library implements the framework across three levels of structural complexity, allowing empirical investigation of which statistical manifold best captures the local market geometry:

| Stage | Model | Manifold |
|---|---|---|
| I | Local Gaussian, $r_t \sim \mathcal{N}(\mu, \sigma^2)$ | 2D: location × scale |
| II | GARCH(1,1) quasi-likelihood | 3D: $[\log\omega,\; \text{logit}\,\alpha,\; \text{logit}\,\beta]$ |
| III | Hawkes process, $\lambda(t) = \mu + \alpha R(t)$ | 3D: $[\log\mu,\; \log\alpha,\; \log\beta]$ |

The Hawkes specification is of particular note: it captures the self-exciting, reflexive nature of order flow arrival, meaning the information clock accelerates precisely when event clustering occurs — which is exactly when the market is generating the most signal.

---

## What This Is Not

This is not a trading strategy. It is a **sampling protocol** — infrastructure that sits upstream of every model, feature, and signal in a systematic pipeline.

The hypothesis is that downstream models (return forecasting, regime classification, covariance estimation, execution algorithms) trained on FIB-sampled data will exhibit more stable out-of-sample behavior than models trained on conventional bars, because the inputs will have more homogeneous statistical conditioning. Testing that hypothesis rigorously against live tick data is the research agenda this codebase is designed to support.

---

## Architecture

The implementation is deliberately modular to support experimentation:

```
fibars/
  config.py              FIBConfig — all hyperparameters in one validated dataclass
  events.py              MarketEvent — atomic input record
  models/                GaussianModel, GARCHModel, HawkesModel
  information/           Scalarizers: logdet · trace · frobenius
  thresholds/            AdaptiveThresholdPolicy · TimeoutPolicy
  bars/                  FIBBuilder (core engine) · BarAggregator · FIBBar
  data/                  DataFrame ↔ MarketEvent adapters
  api/                   build_fib_bars (batch) · StreamingBuilder (live feeds)
```

The `FIBBuilder` is stateful and event-driven — designed for low-latency streaming as well as historical backtesting. The `StreamingBuilder` wrapper adds an optional callback interface and live introspection of the open bar's current information level and threshold.

---

## Baseline Comparisons

Four conventional bar types (time, tick, volume, dollar) ship with the same output schema, enabling direct empirical comparison of bar-by-bar dispersion in $n$, duration, and OHLCV statistics across construction methods.

---

## Output

Each `FIBBar` carries the standard OHLCV fields alongside the information geometry metadata needed for research:

- `information_scalar` — $\Phi(\mathcal{I}_k)$ at bar close
- `threshold_at_close` — the Information Quantum $\mathcal{I}^*$ in effect
- `timeout_flag` — distinguishes information-triggered from timeout-triggered closures
- `model_name`, `info_mode`, `scalarizer_name` — full provenance for experiment tracking
- `start_event_index`, `end_event_index` — linkage back to the raw tick stream

---

## Theoretical Lineage

This work sits at the intersection of five research threads that have, until now, operated in parallel:

**Information geometry** (Amari & Nagaoka, 2000; Amari, 2016) — the Fisher Information Matrix as Riemannian metric tensor; distance on the statistical manifold as the natural measure of inferential progress.

**Business time and subordinated processes** (Clark, 1973; Ané & Geman, 2000) — the insight that price evolution is governed by an endogenous economic clock, not physical time. This work proposes that the correct endogenous clock is the Fisher Information process.

**Event-based sampling** (Easley, López de Prado & O'Hara, 2012) — volume and dollar bars as proxies for information flow. This framework replaces the proxy with the quantity itself.

**Optimal sampling under microstructure noise** (Aït-Sahalia, Mykland & Zhang, 2005) — the proof that the optimal sampling frequency is finite and regime-dependent. This framework makes that frequency adaptive and information-theoretically grounded.

**Score-driven dynamics** (Creal, Koopman & Lucas, 2013) — observation-driven parameter updates via the local score and curvature, which provides the computational bridge from theoretical geometry to real-time, O(1) implementation.

The gap in the literature is precise: despite these threads existing independently for decades, no framework has previously used accumulated Fisher Information as the direct trigger for a financial sampling boundary. This codebase is the first operational implementation of that idea.

---

## Installation

```bash
pip install -e ".[dev]"   # Python 3.11+
```

Dependencies: `numpy`, `pandas`, `scipy`. No GPU required. No exotic dependencies.

---

*v1.0.0 — companion code for the working paper "Fisher Information Bars: An Information-Geometric Approach to Financial Sampling"*
