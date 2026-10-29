# zenn_posting_code

[日本語版 README](README.md)

Companion code and summary tables for the Zenn article "[なぜMLBポスティング選手は短期契約を選ぶのか？──ゲーム理論を使って分析してみる](https://zenn.dev/kakimoto0225/articles/zenn_posting_mlb)" (*Why Do Posted MLB Players Choose Short-Term Contracts? A Game-Theoretic Analysis*).

## Correspondence to the Article

| Article Section | Topic | Files |
|:---|:---|:---|
| Sections 2–4 | Extensive-form game and Nash bargaining model | `src/posting_model/` |
| Section 5 | Baseline simulation ($N = 20{,}000$) | `configs/publication.toml`, `scripts/run_policy_experiments.py`, `results/simulation/policy_comparison.csv` |
| Section 6 | Counterfactual analysis across 8 fee schedules | `scripts/run_policy_experiments.py`, `results/simulation/policy_comparison.csv` |
| Section 7 | Empirical analysis of 19 completed posting contracts (permutation test, OLS, Huber M-estimation, leave-one-out) | `scripts/fee_incidence_primary.py`, `results/empirical/table_fee_incidence_primary.csv` |

### Package Modules (`src/posting_model/`)

- `config.py` — Dataclasses and TOML loader for model parameters and posting fee policies (`FeePolicy`, `ModelConfig`)
- `primitives.py` — Mapping from player ability, age, and remaining club control to MLB contract value $R_T$, player continuation value $A_T$, and origin-club opportunity cost $C_N(x)$
- `fee_schedule.py` — Post-2017 piecewise-linear posting fee schedule $\phi(G)$ (kinks at \$25M and \$50M) and minimum-fee floor
- `bargaining.py` — Duration-specific Nash bargaining solver comparing interior optima across linear segments, kink points, and disagreement
- `equilibrium.py` — Global equilibrium mapping $\Psi^*(x, z; \Theta)$ selecting among short-term, long-term, and no-deal outcomes
- `permission.py` — Numerical integration of the origin club's ex-ante permission value $V_N^{post}(x)$ over ex-post market shocks $z$
- `welfare.py` — Realized payoffs for the origin club, player, and signing MLB club, three-party total surplus, and excess short-term contract indicators
- `batch.py` — Vectorized equilibrium, permission, and welfare evaluation for large artificial populations
- `simulation.py` — Sequential data-generating process with separated random number streams for ex-ante permission decisions and ex-post bargaining shocks

## Repository Structure

```text
.
├── configs/
│   └── publication.toml                 # Baseline parameters for Sections 5–6
├── results/
│   ├── empirical/
│   │   └── table_fee_incidence_primary.csv  # Summary table for Section 7
│   └── simulation/
│       ├── figures/
│       │   └── fig_policy_welfare.png       # Mean payoff by policy
│       ├── runs/
│       │   └── policy_manifest.json         # Execution metadata and config snapshot
│       └── policy_comparison.csv            # Simulation outputs for Sections 5–6
├── scripts/
│   ├── _simulation_bootstrap.py         # Path setup for src/
│   ├── fee_incidence_primary.py         # Empirical script for Section 7 (reference only)
│   └── run_policy_experiments.py        # Simulation runner for Sections 5–6
├── src/
│   └── posting_model/                   # Core game-theoretic model package
├── tests/                               # Unit test suite (19 tests)
├── pyproject.toml
└── requirements.txt
```

## Usage

Requires Python 3.11 or later.

```bash
pip install -r requirements.txt
```

### 1. Reproduce the Simulation and Counterfactual Analysis (Sections 5–6)

```bash
python scripts/run_policy_experiments.py
```

This script loads `configs/publication.toml`, evaluates the 8 institutional scenarios on a common artificial population of $N = 20{,}000$ players with 64 integration draws, and writes `results/simulation/policy_comparison.csv` and `results/simulation/figures/fig_policy_welfare.png` (runtime is approximately 2 minutes).

The figures reported in Sections 5 and 6 of the article correspond to the rows with `permission_model == "integrated_cost"` in `results/simulation/policy_comparison.csv`:

| CSV Column | Article Metric | Baseline (`current`) |
|:---|:---|---:|
| `permission_rate` | Permission rate | 0.3589 (35.9%) |
| `signing_rate` | Signing rate | 0.3501 (35.0%) |
| `short_rate` | Short-term contract rate | 0.09845 (9.8%) |
| `mean_guarantee` | Mean guaranteed value (\$M) | 12.883 |
| `mean_fee` | Mean posting fee (\$M) | 2.568 |
| `excess_short_rate` | Excess short-term rate | 0.02875 (2.9%) |
| `origin_loss_given_permission` | Ex-post origin loss rate given permission | 0.7308 (73.1%) |
| `origin_welfare` | Origin club mean surplus | −0.493 |
| `player_welfare` | Player mean surplus | 6.916 |
| `mlb_welfare` | MLB club mean surplus | 8.123 |
| `total_welfare` | Total surplus (sum of the three parties) | 14.545 |

### 2. Run the Unit Tests

```bash
pytest -q
```

### 3. Empirical Analysis Script (Section 7)

`scripts/fee_incidence_primary.py` computes the raw median difference in real posting fee per projected WAR between short-term ($< 4$ years) and long-term ($\ge 4$ years) contracts (with two-sided permutation $p$-value and percentile bootstrap 95% CI), conditional OLS (HC3), Huber M-estimation, and leave-one-out range across the 19 completed NPB/KBO-to-MLB posting contracts.

Because the input datasets (`data/posting_master.csv` and `data/transfer_player_stats.csv`) contain third-party projection data (Steamer and ZiPS WAR), the raw data files are not redistributed in this repository. Consequently, `scripts/fee_incidence_primary.py` is provided for methodological reference and cannot be executed from this repository alone. The resulting summary table is provided in `results/empirical/table_fee_incidence_primary.csv`.

## Notes on Interpretation

- The simulation results in Sections 5–6 are numerical experiments under the specified model structure and parameter values ($N = 20{,}000$), not empirical estimates of realized financial losses or policy treatment effects.
- `total_welfare` denotes the sum of payoffs of the three modeled agents (the origin club, the player, and the initial signing MLB club).
- The empirical comparisons in Section 7 describe an endogenously selected sample of 19 completed posting transfers and do not identify the causal effect of contract duration.
