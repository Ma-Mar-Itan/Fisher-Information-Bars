# fibars

**Fisher Information Bars** — an information-geometric approach to financial sampling.

A bar closes not when a clock ticks or a volume bucket fills, but when the current unfinished bar has accumulated a target amount of *statistical information* about the local market process.

---

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+, numpy, pandas, scipy.

---

## Quickstart

```python
import pandas as pd
from fibars import build_fib_bars

# df must have columns: timestamp (unix seconds), price
# optional: size, bid, ask, side, event_type
bars = build_fib_bars(
    data=df,
    model="gaussian",       # "gaussian" | "garch" | "hawkes"
    info_mode="observed",   # "observed" | "expected"
    scalarizer="logdet",    # "logdet" | "trace" | "frobenius"
    eta=1.0,
    delta0_seconds=60.0,
    timeout_seconds=300.0,
)
print(bars[["open_time","close_time","n_events","information_scalar","timeout_flag"]])
```

Streaming usage:

```python
from fibars import FIBConfig, StreamingBuilder
from fibars.events import MarketEvent

builder = StreamingBuilder(config=FIBConfig(model="gaussian", eta=1.0))
for raw in tick_feed:
    event = MarketEvent(timestamp=raw.ts, price=raw.price, size=raw.qty)
    bar = builder.push(event)
    if bar is not None:
        handle_closed_bar(bar)
final = builder.flush()
```

---

## Conceptual Summary

Traditional bar types (time, tick, volume, dollar) standardise the *mechanics* of exchange but not the *inferential density* of observations. In a volatile regime, 100 ticks may carry far more information than in a quiet one — yet a tick bar treats them identically.

Fisher Information Bars solve this by defining a bar as a completed **inferential unit**. A new bar opens when the previous one closes; it closes when the scalarised Fisher Information accumulated over all events in the bar reaches a configurable **Information Quantum** `I*`.

```
I_k(t) = Σ Ψ_i(θ̂_{i-1})      (cumulative information path)
Ψ_i    = s_i s_i^T            (outer-product-of-gradients, O(1) per event)
close  when  Φ(I_k(t)) ≥ I*   (scalarised threshold)
```

---

## Architecture

```
fibars/
  config.py          FIBConfig — all hyperparameters in one place
  events.py          MarketEvent — input record
  models/            GaussianModel, GARCHModel, HawkesModel
  information/       Scalarizers (logdet, trace, frobenius)
  thresholds/        AdaptiveThresholdPolicy, TimeoutPolicy
  bars/              FIBBuilder (core engine), BarAggregator, FIBBar
  data/              DataFrame ↔ MarketEvent adapters
  api/               build_fib_bars (batch), StreamingBuilder
```

---

## Models

### Gaussian (`model="gaussian"`)
Local arithmetic returns are modelled as `r_t ~ N(μ, σ²)`.  
Parameters `θ = [μ, log σ²]` are updated online via Welford's algorithm.  
Analytic expected information: `diag(1/σ², 0.5)`.

### GARCH(1,1) (`model="garch"`)
Quasi-likelihood with recursive variance: `σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}`.  
Parameters `θ = [log ω, logit α, logit β]`. Persistence `α+β` is capped at `garch_persistence_max`.  
Score computed via O(1) finite differences on the log-parameterised space.

### Hawkes (`model="hawkes"`)
Univariate point process with exponential kernel: `λ(t) = μ + α R(t)`.  
Excitation state `R(t)` updated recursively: `R ← R·exp(−β Δt) + 1` at each event.  
Score computed via O(1) finite differences on `[log μ, log α, log β]`.

---

## Scalarizers

| Name | Formula | Property |
|------|---------|----------|
| `logdet` | `log det(J + εI)` | Reparameterisation-invariant (default) |
| `trace`  | `tr(J) + ε·d` | A-optimality (sum of eigenvalues) |
| `frobenius` | `‖J‖_F + ε` | Entry-wise norm |

---

## Adaptive Thresholding

The Information Quantum adapts to the asset's long-run information rate:

```
I* = η · λ̄_Φ · Δ₀
```

where `λ̄_Φ` is an EWMA of the scalarised information rate across completed bars, `Δ₀` is a reference duration (`delta0_seconds`), and `η` is a user multiplier. During warmup the threshold seeds from the last observed scalar.

---

## Timeout Rule

A bar closes via timeout (setting `timeout_flag=True`) when **any** of:
- elapsed seconds ≥ `timeout_seconds`
- event count ≥ `max_events_per_bar`
- seconds since last event ≥ `inactivity_timeout_seconds` (if set)

This prevents bar starvation during low-activity regimes.

---

## Output Schema

Each `FIBBar` contains:

| Field | Description |
|-------|-------------|
| `open_time`, `close_time` | Unix timestamps |
| `duration_seconds` | Bar wall-clock length |
| `n_events` | Events in bar |
| `open/high/low/close` | Price OHLC |
| `sum_volume`, `dollar_value` | Volume aggregates |
| `mean_spread` | Mean bid-ask spread (if available) |
| `information_scalar` | `Φ(I_k)` at close |
| `threshold_at_close` | `I*` when bar closed |
| `timeout_flag` | True if closed by timeout |
| `model_name`, `info_mode`, `scalarizer_name` | Provenance |
| `start_event_index`, `end_event_index` | Index into original data |

---

## Baseline Bars

For comparison, four standard bar types are included with the same output schema:

```python
from fibars import build_time_bars, build_tick_bars, build_volume_bars, build_dollar_bars
```

---

## Configuration Reference

All parameters live in `FIBConfig`:

```python
from fibars import FIBConfig
cfg = FIBConfig(
    model="gaussian",           # model choice
    info_mode="observed",       # "observed" | "expected"
    scalarizer="logdet",
    eta=1.0,                    # threshold multiplier
    delta0_seconds=60.0,        # reference bar duration
    ewma_alpha=0.05,            # EWMA smoothing for info rate
    eps_ridge=1e-6,             # ridge stabilisation
    timeout_seconds=300.0,
    max_events_per_bar=10_000,
    min_warmup_events=20,
    var_floor=1e-12,            # Gaussian variance floor
    garch_persistence_max=0.9999,
    hawkes_intensity_floor=1e-8,
)
```

---

## Limitations

- GARCH and Hawkes scores use O(1) finite-difference approximations rather than exact analytic gradients. This is stable and fast but slightly noisy.
- Parameters are not re-estimated via MLE between bars; they evolve via online Welford (Gaussian) or fixed initialisation (GARCH, Hawkes). Full online EM or GAS updates are a natural extension.
- The adaptive threshold requires a warmup period (`min_warmup_events` bars) before the EWMA is reliable; early bars may be large or small.
- Hawkes model fitness degrades if inter-arrival times are very heterogeneous at the start of the stream (before `R(t)` is warmed up).
