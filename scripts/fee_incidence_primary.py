#!/usr/bin/env python3
"""Build primary fee-incidence robustness outputs for the manuscript.

記事の第7章（19件の実データによる並べ替え検定など）の表を計算する。
入力データは公開していないため、このリポジトリだけでは実行できない（README を参照）。

Inputs (not included in this repository):
    - data/posting_master.csv
    - data/transfer_player_stats.csv

Outputs:
    - results/empirical/table_fee_incidence_primary.csv
    - results/empirical/table_contract_definition_sensitivity.csv
    - results/empirical/figures/fig_fee_per_war_by_contract_type.(png|pdf)
    - results/empirical/figures/fig_fee_gap_loo.(png|pdf)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
RESULT_DIR = ROOT_DIR / "results" / "empirical"
FIG_DIR = RESULT_DIR / "figures"

INPUT_POSTING = DATA_DIR / "posting_master.csv"
INPUT_TRANSFER = DATA_DIR / "transfer_player_stats.csv"

OUTPUT_PRIMARY = RESULT_DIR / "table_fee_incidence_primary.csv"
OUTPUT_SENSITIVITY = RESULT_DIR / "table_contract_definition_sensitivity.csv"
OUTPUT_FIG_DIST_PNG = FIG_DIR / "fig_fee_per_war_by_contract_type.png"
OUTPUT_FIG_DIST_PDF = FIG_DIR / "fig_fee_per_war_by_contract_type.pdf"
OUTPUT_FIG_LOO_PNG = FIG_DIR / "fig_fee_gap_loo.png"
OUTPUT_FIG_LOO_PDF = FIG_DIR / "fig_fee_gap_loo.pdf"

PRIMARY_WAR_COL = "WAR_mean3"
BOOT_REPS = 10000
PERM_REPS = 10000


def posting_fee_from_schedule(contract_value_musd: float) -> float:
    """Return the posting fee implied by the post-2017 posting schedule.

    Args:
        contract_value_musd: Guaranteed first-contract value in nominal million USD.

    Returns:
        Posting fee in nominal million USD.
    """
    if not np.isfinite(contract_value_musd):
        return np.nan
    if contract_value_musd <= 25:
        return 0.20 * contract_value_musd
    if contract_value_musd <= 50:
        return 5.0 + 0.175 * (contract_value_musd - 25)
    return 9.375 + 0.15 * (contract_value_musd - 50)


def load_augmented_posting_data() -> pd.DataFrame:
    """Load posting data with age and fee fallbacks applied.

    Returns:
        Posting data augmented with age and posting-fee completions.
    """
    posting = pd.read_csv(INPUT_POSTING, low_memory=False)
    transfer = pd.read_csv(INPUT_TRANSFER, low_memory=False)

    posting_numeric = [
        "Year",
        "Contract_Years",
        "Contract_Value_M",
        "Posting_Fee_M",
        "Age_at_Post",
        "Inflation_Index",
        "Projected_WAR",
        "Steamer_WAR",
        "ZiPS_WAR",
        "WAR_mean3",
        "is_Signed",
        "is_LongTerm",
        "is_NPB",
        "is_Pitcher",
    ]
    transfer_numeric = ["Year", "Posting_Age", "Posting_Fee_Nominal_M"]

    for column in posting_numeric:
        if column in posting.columns:
            posting[column] = pd.to_numeric(posting[column], errors="coerce")
    for column in transfer_numeric:
        if column in transfer.columns:
            transfer[column] = pd.to_numeric(transfer[column], errors="coerce")

    transfer_use = transfer[["Player", "Year", "Posting_Age", "Posting_Fee_Nominal_M"]].copy()
    transfer_use = transfer_use.rename(
        columns={
            "Posting_Age": "Posting_Age_transfer",
            "Posting_Fee_Nominal_M": "Posting_Fee_M_transfer",
        }
    )
    df = posting.merge(transfer_use, on=["Player", "Year"], how="left")

    df["Age_at_Post_filled"] = df["Age_at_Post"].fillna(df["Posting_Age_transfer"])
    df["Posting_Fee_M_filled"] = df["Posting_Fee_M"].fillna(df["Posting_Fee_M_transfer"])

    need_schedule = (
        df["is_Signed"].eq(1)
        & df["Contract_Value_M"].notna()
        & df["Posting_Fee_M_filled"].isna()
    )
    if need_schedule.any():
        df.loc[need_schedule, "Posting_Fee_M_filled"] = df.loc[need_schedule, "Contract_Value_M"].map(
            posting_fee_from_schedule
        )

    df["Inflation_Index"] = df["Inflation_Index"].fillna(1.0)
    df["Posting_Fee_Real_M_filled"] = df["Posting_Fee_M_filled"] / df["Inflation_Index"]
    return df


def build_sample(df: pd.DataFrame, war_col: str = PRIMARY_WAR_COL) -> pd.DataFrame:
    """Construct the baseline signed sample used in the primary table.

    Args:
        df: Augmented posting data.
        war_col: WAR column used in the baseline specification.

    Returns:
        Sample restricted to signed cases with finite contract years and fee/WAR.
    """
    sample = df[df["is_Signed"].eq(1)].copy()
    sample[war_col] = pd.to_numeric(sample[war_col], errors="coerce")
    sample = sample.dropna(subset=["Contract_Years", war_col, "Posting_Fee_Real_M_filled"]).copy()
    sample["fee_per_war"] = sample["Posting_Fee_Real_M_filled"] / sample[war_col]
    sample["fee_per_war"] = sample["fee_per_war"].replace([np.inf, -np.inf], np.nan)
    sample = sample.dropna(subset=["fee_per_war"]).copy()
    sample["short_term"] = (sample["Contract_Years"] < 4).astype(int)
    sample["war"] = sample[war_col].astype(float)
    sample["war_sq"] = sample["war"] ** 2
    sample["age_missing"] = sample["Age_at_Post_filled"].isna().astype(int)
    sample["age"] = sample["Age_at_Post_filled"].fillna(sample["Age_at_Post_filled"].median())
    sample["posting_year"] = sample["Year"].astype(int)
    return sample


def compute_median_diff(sample: pd.DataFrame, short_col: str = "short_term") -> float:
    """Compute the short-minus-long median gap in fee per WAR.

    Args:
        sample: Sample containing fee_per_war and short/long indicator.
        short_col: Indicator column equal to one for the short-term group.

    Returns:
        Median(short) minus median(long).
    """
    short_fee = sample.loc[sample[short_col].eq(1), "fee_per_war"]
    long_fee = sample.loc[sample[short_col].eq(0), "fee_per_war"]
    return float(short_fee.median() - long_fee.median())


def bootstrap_median_diff(
    sample: pd.DataFrame,
    short_col: str = "short_term",
    n_boot: int = BOOT_REPS,
) -> tuple[float, float]:
    """Bootstrap the median difference in fee per WAR.

    Args:
        sample: Baseline sample.
        short_col: Indicator column for short-term contracts.
        n_boot: Number of bootstrap replications.

    Returns:
        Lower and upper percentile bootstrap bounds.
    """
    rng = np.random.default_rng(42)
    short_fee = sample.loc[sample[short_col].eq(1), "fee_per_war"].to_numpy(dtype=float)
    long_fee = sample.loc[sample[short_col].eq(0), "fee_per_war"].to_numpy(dtype=float)
    estimates = np.empty(n_boot, dtype=float)
    for rep in range(n_boot):
        short_draw = short_fee[rng.integers(0, len(short_fee), len(short_fee))]
        long_draw = long_fee[rng.integers(0, len(long_fee), len(long_fee))]
        estimates[rep] = np.median(short_draw) - np.median(long_draw)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def permutation_p_value(
    sample: pd.DataFrame,
    short_col: str = "short_term",
    n_perm: int = PERM_REPS,
) -> float:
    """Run a randomization test for the median fee/WAR gap.

    Args:
        sample: Baseline sample.
        short_col: Indicator column for short-term contracts.
        n_perm: Number of label permutations.

    Returns:
        Two-sided permutation p-value.
    """
    rng = np.random.default_rng(42)
    observed = compute_median_diff(sample, short_col=short_col)
    fee_values = sample["fee_per_war"].to_numpy(dtype=float)
    short_count = int(sample[short_col].sum())
    permuted = np.empty(n_perm, dtype=float)
    for rep in range(n_perm):
        draw = rng.permutation(len(fee_values))
        short_fee = fee_values[draw[:short_count]]
        long_fee = fee_values[draw[short_count:]]
        permuted[rep] = np.median(short_fee) - np.median(long_fee)
    return float(np.mean(np.abs(permuted) >= abs(observed)))


def design_matrix(sample: pd.DataFrame, use_short_term: bool = True) -> pd.DataFrame:
    """Build the regression design matrix for conditional comparisons.

    Args:
        sample: Baseline sample.
        use_short_term: Whether to include the short-term dummy.

    Returns:
        Regression design matrix with controls and year fixed effects.
    """
    columns = ["war", "war_sq", "age", "age_missing", "is_Pitcher", "is_NPB"]
    if use_short_term:
        columns = ["short_term"] + columns
    year_dummies = pd.get_dummies(sample["posting_year"], prefix="year", drop_first=True, dtype=float)
    matrix = pd.concat(
        [pd.Series(1.0, index=sample.index, name="const"), sample[columns].astype(float), year_dummies],
        axis=1,
    )
    return matrix


def fit_models(sample: pd.DataFrame) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, sm.robust.robust_linear_model.RLMResultsWrapper]:
    """Fit OLS and robust linear models for the fee/WAR outcome.

    Args:
        sample: Baseline sample.

    Returns:
        Tuple of OLS and robust model results.
    """
    y = sample["fee_per_war"].astype(float)
    x = design_matrix(sample, use_short_term=True)
    ols = sm.OLS(y, x).fit(cov_type="HC3")
    robust = sm.RLM(y, x, M=sm.robust.norms.HuberT()).fit()
    return ols, robust


def model_row(
    method: str,
    estimate: float,
    std_error: float | None,
    p_value: float | None,
    sample: pd.DataFrame,
    notes: str,
) -> dict[str, float | int | str]:
    """Format one output row for the primary robustness table."""
    ci_lo = np.nan
    ci_hi = np.nan
    if std_error is not None and np.isfinite(std_error):
        ci_lo = float(estimate - 1.96 * std_error)
        ci_hi = float(estimate + 1.96 * std_error)
    return {
        "method": method,
        "estimate": float(estimate),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": np.nan if p_value is None else float(p_value),
        "n_total": int(len(sample)),
        "n_short": int(sample["short_term"].sum()),
        "n_long": int((1 - sample["short_term"]).sum()),
        "notes": notes,
    }


def leave_one_out(sample: pd.DataFrame) -> pd.DataFrame:
    """Compute leave-one-out median-gap estimates.

    Args:
        sample: Baseline sample.

    Returns:
        DataFrame with one leave-one-out estimate per omitted player.
    """
    rows: list[dict[str, float | str]] = []
    for idx, row in sample.iterrows():
        reduced = sample.drop(index=idx)
        estimate = compute_median_diff(reduced)
        rows.append(
            {
                "omitted_player": str(row["Player"]),
                "omitted_year": int(row["posting_year"]),
                "estimate": float(estimate),
            }
        )
    return pd.DataFrame(rows).sort_values("estimate").reset_index(drop=True)


def definition_sensitivity(sample: pd.DataFrame) -> pd.DataFrame:
    """Evaluate alternative short-term definitions and a continuous-years model.

    Args:
        sample: Baseline sample.

    Returns:
        DataFrame summarizing discrete-threshold and continuous specifications.
    """
    rows: list[dict[str, float | int | str]] = []
    definitions = [
        ("Contract_Years <= 2", sample["Contract_Years"] <= 2),
        ("Contract_Years <= 3", sample["Contract_Years"] <= 3),
        ("Contract_Years < 4", sample["Contract_Years"] < 4),
        ("Contract_Years < 5", sample["Contract_Years"] < 5),
    ]

    for label, indicator in definitions:
        spec_sample = sample.copy()
        spec_sample["short_alt"] = indicator.astype(int)
        if spec_sample["short_alt"].nunique() < 2:
            continue
        estimate = compute_median_diff(spec_sample, short_col="short_alt")
        ci_lo, ci_hi = bootstrap_median_diff(spec_sample, short_col="short_alt")
        rows.append(
            {
                "definition": label,
                "estimate": float(estimate),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "p_value": np.nan,
                "n_short": int(spec_sample["short_alt"].sum()),
                "n_long": int((1 - spec_sample["short_alt"]).sum()),
                "notes": "Median fee/WAR difference (short - long)",
            }
        )

    x = pd.concat(
        [
            pd.Series(1.0, index=sample.index, name="const"),
            sample[["Contract_Years", "war", "war_sq", "age", "age_missing", "is_Pitcher", "is_NPB"]].astype(float),
            pd.get_dummies(sample["posting_year"], prefix="year", drop_first=True, dtype=float),
        ],
        axis=1,
    )
    model = sm.OLS(sample["fee_per_war"].astype(float), x).fit(cov_type="HC3")
    rows.append(
        {
            "definition": "Continuous contract years",
            "estimate": float(model.params["Contract_Years"]),
            "ci_lo": float(model.conf_int().loc["Contract_Years", 0]),
            "ci_hi": float(model.conf_int().loc["Contract_Years", 1]),
            "p_value": float(model.pvalues["Contract_Years"]),
            "n_short": int(sample["short_term"].sum()),
            "n_long": int((1 - sample["short_term"]).sum()),
            "notes": "OLS slope of fee/WAR on contract years with the baseline controls",
        }
    )
    return pd.DataFrame(rows)


def plot_distribution(sample: pd.DataFrame) -> None:
    """Plot fee/WAR distributions for short- and long-term contracts.

    Args:
        sample: Baseline sample.
    """
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    groups = [
        sample.loc[sample["short_term"].eq(1), "fee_per_war"].to_numpy(dtype=float),
        sample.loc[sample["short_term"].eq(0), "fee_per_war"].to_numpy(dtype=float),
    ]
    labels = ["Short-term", "Long-term"]
    colors = ["#c44e52", "#4c72b0"]

    box = ax.boxplot(groups, patch_artist=True, widths=0.48, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set(facecolor=color, alpha=0.35, edgecolor=color, linewidth=1.5)
    for median in box["medians"]:
        median.set(color="black", linewidth=1.8)

    rng = np.random.default_rng(42)
    for idx, values in enumerate(groups, start=1):
        x_jitter = idx + rng.uniform(-0.10, 0.10, size=len(values))
        ax.scatter(
            x_jitter,
            values,
            s=48,
            color=colors[idx - 1],
            alpha=0.80,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        ax.text(
            idx,
            np.median(values) + 0.18,
            f"Median = {np.median(values):.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks([1, 2], labels)
    ax.set_ylabel("Posting fee per projected WAR (real 2018 USD, $M)")
    ax.set_title("Quality-adjusted posting compensation by contract type")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIST_PNG, dpi=300)
    fig.savefig(OUTPUT_FIG_DIST_PDF)
    plt.close(fig)


def plot_leave_one_out(loo_df: pd.DataFrame, full_estimate: float) -> None:
    """Plot leave-one-out fee-gap estimates.

    Args:
        loo_df: Leave-one-out estimate table.
        full_estimate: Full-sample median gap.
    """
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    x_axis = np.arange(1, len(loo_df) + 1)
    ax.scatter(x_axis, loo_df["estimate"], s=46, color="#4c72b0", edgecolors="white", linewidths=0.6)
    ax.axhline(full_estimate, color="black", linewidth=1.8, linestyle="--", label=f"Full sample = {full_estimate:.2f}")
    ax.set_xticks(x_axis)
    ax.set_xticklabels([name.split()[-1] for name in loo_df["omitted_player"]], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Leave-one-out median fee/WAR gap (short - long)")
    ax.set_title("Leave-one-out stability of the fee-incidence gap")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_LOO_PNG, dpi=300)
    fig.savefig(OUTPUT_FIG_LOO_PDF)
    plt.close(fig)


def main() -> None:
    """Run the primary fee-incidence robustness workflow."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sample = build_sample(load_augmented_posting_data(), war_col=PRIMARY_WAR_COL)

    raw_estimate = compute_median_diff(sample)
    boot_lo, boot_hi = bootstrap_median_diff(sample)
    perm_p = permutation_p_value(sample)
    ols, robust = fit_models(sample)
    loo_df = leave_one_out(sample)
    sensitivity_df = definition_sensitivity(sample)

    robust_p = float(2 * stats.norm.sf(abs(robust.tvalues["short_term"])))

    rows = [
        {
            "method": "Raw median difference",
            "estimate": float(raw_estimate),
            "ci_lo": float(boot_lo),
            "ci_hi": float(boot_hi),
            "p_value": float(perm_p),
            "n_total": int(len(sample)),
            "n_short": int(sample["short_term"].sum()),
            "n_long": int((1 - sample["short_term"]).sum()),
            "notes": "Mean3 fee/WAR; p-value from a two-sided permutation test",
        },
        model_row(
            method="Conditional OLS",
            estimate=float(ols.params["short_term"]),
            std_error=float(ols.bse["short_term"]),
            p_value=float(ols.pvalues["short_term"]),
            sample=sample,
            notes="HC3 standard errors; controls for WAR, WAR^2, age, age-missing, pitcher, NPB, and posting-year fixed effects",
        ),
        model_row(
            method="Robust regression",
            estimate=float(robust.params["short_term"]),
            std_error=float(robust.bse["short_term"]),
            p_value=robust_p,
            sample=sample,
            notes="Huber M-estimation with the same controls as the OLS specification",
        ),
        {
            "method": "Leave-one-out range",
            "estimate": float(raw_estimate),
            "ci_lo": float(loo_df["estimate"].min()),
            "ci_hi": float(loo_df["estimate"].max()),
            "p_value": np.nan,
            "n_total": int(len(sample)),
            "n_short": int(sample["short_term"].sum()),
            "n_long": int((1 - sample["short_term"]).sum()),
            "notes": "Range of the raw median difference after omitting one signed case at a time",
        },
    ]
    primary_df = pd.DataFrame(rows)
    primary_df.to_csv(OUTPUT_PRIMARY, index=False)
    sensitivity_df.to_csv(OUTPUT_SENSITIVITY, index=False)

    plot_distribution(sample)
    plot_leave_one_out(loo_df, raw_estimate)

    print(f"saved {OUTPUT_PRIMARY}")
    print(f"saved {OUTPUT_SENSITIVITY}")
    print(f"saved {OUTPUT_FIG_DIST_PNG}")
    print(f"saved {OUTPUT_FIG_LOO_PNG}")


if __name__ == "__main__":
    main()
