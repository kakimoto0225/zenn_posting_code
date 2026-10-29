"""記事の第5章（基準シミュレーション）と第6章（8つの制度の反実仮想分析）の数値を計算する。

出力: results/simulation/policy_comparison.csv
記事の数値は permission_model == "integrated_cost" の行に対応する。
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import _simulation_bootstrap
import matplotlib.pyplot as plt
import pandas as pd

from posting_model.config import FeePolicy, load_config
from posting_model.paths import FIGURES, OUTPUTS, ensure_directories
from posting_model.reporting import write_manifest
from posting_model.simulation import simulate_sample


def summarize(
    policy: str, model: str, observed: pd.DataFrame, truth: pd.DataFrame
) -> dict[str, float | str]:
    """制度・許可モデル別の疑似結果を集計する。"""
    permitted = observed["permission"].astype(bool).to_numpy()
    conditional = truth.loc[permitted]

    def conditional_mean(column: str) -> float:
        return float(conditional[column].mean()) if len(conditional) else float("nan")

    def standard_error(series: pd.Series) -> float:
        return (
            float(series.std(ddof=1) / len(series) ** 0.5)
            if len(series) > 1
            else float("nan")
        )

    return {
        "policy": policy,
        "permission_model": model,
        "permission_rate": float(observed.permission.mean()),
        "permission_se": standard_error(observed.permission),
        "signing_rate": float(observed.outcome.isin(["S", "L"]).mean()),
        "short_rate": float(observed.outcome.eq("S").mean()),
        "short_se": standard_error(observed.outcome.eq("S").astype(float)),
        "mean_guarantee": float(observed.guarantee.mean()),
        "mean_fee": float(observed.posting_fee.mean()),
        "excess_short_rate": float(truth.realized_excess_short.mean()),
        "origin_loss_rate": float(truth.realized_origin_loss.mean()),
        "origin_loss_given_permission": conditional_mean("potential_origin_loss"),
        "origin_loss_given_permission_se": standard_error(
            conditional["potential_origin_loss"]
        ),
        "no_deal_loss_given_permission": conditional_mean("potential_no_deal_loss"),
        "short_loss_given_permission": conditional_mean(
            "potential_short_contract_loss"
        ),
        "long_loss_given_permission": conditional_mean("potential_long_contract_loss"),
        "excess_short_loss_given_permission": conditional_mean(
            "potential_excess_short_loss"
        ),
        "joint_adverse_rate": float(truth.realized_joint_adverse.mean()),
        "joint_adverse_se": standard_error(truth.realized_joint_adverse),
        "origin_welfare": float(truth.realized_origin_welfare.mean()),
        "player_welfare": float(truth.realized_player_welfare.mean()),
        "mlb_welfare": float(truth.realized_mlb_welfare.mean()),
        "total_welfare": float(truth.realized_total_welfare.mean()),
        "total_welfare_se": standard_error(truth.realized_total_welfare),
    }


def main() -> None:
    started = time.perf_counter()
    ensure_directories()
    base = load_config(_simulation_bootstrap.ROOT / "configs" / "publication.toml").replace(
        n=20000, integration_draws=64
    )
    output_path = OUTPUTS / "policy_comparison.csv"
    policies = [
        FeePolicy("current"),
        FeePolicy("no_fee", rates=(0, 0, 0)),
        FeePolicy("flat_15", rates=(0.15, 0.15, 0.15)),
        FeePolicy("flat_20", rates=(0.20, 0.20, 0.20)),
        FeePolicy("trailing_10", trailing_rate=0.10),
        FeePolicy("minimum_5", minimum_fee=5.0),
        FeePolicy("deferred", rates=(0.10, 0.10, 0.10), trailing_rate=0.10),
        FeePolicy("pullback", rates=(0.20, 0.175, 0.15)),
    ]
    if os.environ.get("SIM_POLICY_REUSE", "0") == "1" and output_path.exists():
        frame = pd.read_csv(output_path)
    else:
        rows = []
        for policy in policies:
            cfg = base.replace(policy=policy)
            integrated_observed, integrated_truth = simulate_sample(cfg)
            rows.append(
                summarize(
                    policy.name,
                    "integrated_cost",
                    integrated_observed,
                    integrated_truth,
                )
            )
            fixed_observed, fixed_truth = simulate_sample(
                cfg, permission_bargaining_cfg=base
            )
            rows.append(
                summarize(policy.name, "fixed_expectation", fixed_observed, fixed_truth)
            )
            deterministic_cfg = cfg.replace(sigma_kappa=0.0)
            deterministic_observed, deterministic_truth = simulate_sample(
                deterministic_cfg
            )
            rows.append(
                summarize(
                    policy.name,
                    "integrated_deterministic",
                    deterministic_observed,
                    deterministic_truth,
                )
            )
        frame = pd.DataFrame(rows)
        frame.to_csv(output_path, index=False)
    plot_frame = frame.loc[frame.permission_model.eq("integrated_cost")]
    plot = plot_frame.set_index("policy")[
        ["origin_welfare", "player_welfare", "mlb_welfare"]
    ]
    ax = plot.plot(kind="bar", figsize=(10, 5), color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set(
        title="Simulated welfare by policy under common random numbers",
        ylabel="Mean payoff",
        xlabel="Policy",
    )
    ax.legend(frameon=False)
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURES / "fig_policy_welfare.png", dpi=200)
    ax.figure.savefig(FIGURES / "fig_policy_welfare.pdf")
    plt.close(ax.figure)
    write_manifest(
        base,
        "run_policy_experiments",
        started,
        {
            "policies": len(policies),
            "permission_models": 3,
            "rows": len(frame),
            "common_random_numbers": True,
        },
        OUTPUTS / "runs" / "policy_manifest.json",
    )


if __name__ == "__main__":
    main()
