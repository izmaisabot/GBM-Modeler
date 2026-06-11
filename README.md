# Geometric Brownian Motion Modeler

An interactive modeler for modelling asset price paths under Geometric Brownian Motion (GBM) — the stochastic process underlying the Black-Scholes options pricing model and much of quantitative finance.

---

## What is Geometric Brownian Motion?

GBM is a continuous-time stochastic process used to model quantities that grow multiplicatively and are subject to random shocks — most famously, stock prices.

The defining property of GBM is that **returns are normally distributed**, while **prices are lognormally distributed**. This means prices can never go below zero (a stock can't have negative value), but can grow without bound — which matches observed market behaviour better than a simple random walk.

The continuous-time stochastic differential equation (SDE) is:

```
dS = μS dt + σS dW
```

Where:

| Symbol | Name | Meaning |
|--------|------|---------|
| `S` | Price | The asset price at time `t` |
| `dS` | Price increment | The infinitesimal change in price |
| `μ` | Drift | The expected annualised rate of return |
| `σ` | Volatility | The annualised standard deviation of returns |
| `dt` | Time increment | An infinitesimally small time step |
| `dW` | Wiener increment | A random shock drawn from `N(0, dt)` |

The term `μS dt` is the **deterministic drift** — the average direction prices move. The term `σS dW` is the **stochastic diffusion** — the random noise layered on top.

---

## The Original Python Implementation

```python
def gbm(num_years=10, num_scenarios=50, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100):
    dt = 1 / steps_per_year
    num_steps = num_years * steps_per_year
    r_p = np.random.normal(
        loc=((1 + mu) ** dt),
        scale=sigma * np.sqrt(dt),
        size=(num_steps, num_scenarios)
    )
    r_p = pd.concat(
        [pd.DataFrame(np.ones((1, num_scenarios))), pd.DataFrame(r_p)],
        ignore_index=True
    )
    return s_0 * pd.DataFrame(r_p).cumprod(0)
```

### Line-by-line breakdown

**`dt = 1 / steps_per_year`**
The size of each time step in years. With monthly steps, `dt = 1/12`. Smaller steps give more accurate simulations but are slower.

**`num_steps = num_years * steps_per_year`**
Total number of steps in the simulation. For 10 years at monthly frequency, this is 120 steps.

**`r_p = np.random.normal(loc=((1+mu)**dt), scale=sigma*np.sqrt(dt), ...)`**
Draws a matrix of random return multipliers of shape `(num_steps, num_scenarios)`. Each value represents the gross return over one time step for one scenario.

- `loc = (1 + mu) ** dt` — the mean of the distribution, i.e. the expected gross return per step. This discretises the drift by compounding over `dt`.
- `scale = sigma * sqrt(dt)` — the standard deviation of returns per step, which scales with the square root of time (a fundamental property of Brownian motion).

**`pd.concat([ones_row, r_p])`**
Prepends a row of ones to the return matrix. This represents time `t = 0`, where the cumulative product starts at 1 (i.e. no change yet from the initial price).

**`s_0 * pd.DataFrame(r_p).cumprod(0)`**
Takes the cumulative product along axis 0 (down the rows, i.e. across time). Each column accumulates returns to give one price path. Multiplying by `s_0` scales from a return index to actual prices.

### Note on the drift approximation

The original code draws returns from a normal distribution centred at `(1+μ)^dt`. This is a **discrete approximation** — it models compounded returns rather than using the exact lognormal formula. For small `dt` and moderate `μ`, the results are very close to the exact solution. The interactive HTML version uses the exact log-normal discretisation: `exp((μ − σ²/2)dt + σ√dt · Z)`.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_years` | 10 | Simulation horizon in years |
| `num_scenarios` | 50 | Number of independent price paths to generate |
| `mu` | 0.07 | Annual drift (expected return), e.g. 0.07 = 7% |
| `sigma` | 0.15 | Annual volatility, e.g. 0.15 = 15% |
| `steps_per_year` | 12 | Discretisation frequency (12 = monthly) |
| `s_0` | 100 | Starting price |

---

## References

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy.
- Merton, R. C. (1973). *Theory of Rational Option Pricing.* Bell Journal of Economics.
- Øksendal, B. (2003). *Stochastic Differential Equations: An Introduction with Applications.* Springer.
- Hull, J. C. (2017). *Options, Futures, and Other Derivatives.* Pearson.
