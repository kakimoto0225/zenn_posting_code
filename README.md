# zenn_posting_code

[English README](README_en.md)

Zenn記事「[なぜMLBポスティング選手は短期契約を選ぶのか？──ゲーム理論を使って分析してみる](https://zenn.dev/kakimoto0225/articles/zenn_posting_mlb)」の分析コードと出力表をまとめたリポジトリである。

## 記事との対応

| 記事の章 | 内容 | 対応するファイル |
|:---|:---|:---|
| 2〜4章 | 展開型ゲームとNash交渉解のモデル | `src/posting_model/` |
| 5章 | 基準シミュレーション（$N = 20{,}000$） | `configs/publication.toml`, `scripts/run_policy_experiments.py`, `results/simulation/policy_comparison.csv` |
| 6章 | 8つの制度シナリオによる反実仮想分析 | `scripts/run_policy_experiments.py`, `results/simulation/policy_comparison.csv` |
| 7章 | 19件の実データを用いた並べ替え検定・回帰・1件除外分析 | `scripts/fee_incidence_primary.py`, `results/empirical/table_fee_incidence_primary.csv` |

### モジュールの役割（`src/posting_model/`）

- `config.py` — モデルパラメータと譲渡金ルール（`FeePolicy`, `ModelConfig`）の読み込み
- `primitives.py` — 選手の能力・年齢・保有年数からMLB側の契約価値 $R_T$ と選手の将来価値 $A_T$、NPB球団の戦力喪失費用 $C_N(x)$ を計算
- `fee_schedule.py` — 2018年以降の区分線形な譲渡金関数 $\phi(G)$（折れ点 25・50百万ドル）および最低譲渡金の計算
- `bargaining.py` — 各線形区間での内点解と折れ点・不成立を比較する契約期間別ソルバー
- `equilibrium.py` — 短期契約・長期契約・不成立からNash積を最大化する均衡契約 $\Psi^*(x, z; \Theta)$ を決定
- `permission.py` — 事後ショック $z$ について期待値をとったNPB球団の許可価値 $V_N^{post}(x)$ の数値積分
- `welfare.py` — NPB球団・選手・MLB球団の実現利得と3者合計の余剰、および長期契約が3者合計で望ましいのに短期契約が選ばれる過剰短期の判定
- `batch.py` — $N = 20{,}000$ 人の仮想母集団に対するベクトル化ソルバー
- `simulation.py` — 許可判断用の乱数と事後交渉ショック用の乱数を分離した標本生成（事後撤回権 `pullback` の処理を含む）

## ディレクトリ構成

```text
.
├── configs/
│   └── publication.toml                 # 記事5〜6章の基準パラメータ設定
├── results/
│   ├── empirical/
│   │   └── table_fee_incidence_primary.csv  # 記事7章の集計表
│   └── simulation/
│       ├── figures/
│       │   └── fig_policy_welfare.png       # 制度別の平均余剰の図
│       ├── runs/
│       │   └── policy_manifest.json         # 実行環境と設定の記録
│       └── policy_comparison.csv            # 記事5〜6章のシミュレーション結果表
├── scripts/
│   ├── _simulation_bootstrap.py         # src/ へのパス設定
│   ├── fee_incidence_primary.py         # 記事7章の実データ分析スクリプト（参照用）
│   └── run_policy_experiments.py        # 記事5〜6章のシミュレーション実行スクリプト
├── src/
│   └── posting_model/                   # ゲーム理論モデル本体
├── tests/                               # 単体テスト（19件）
├── pyproject.toml
└── requirements.txt
```

## 実行方法

Python 3.11 以上を前提とする。

```bash
pip install -r requirements.txt
```

### 1. シミュレーションと反実仮想分析（記事5〜6章）の再現

```bash
python scripts/run_policy_experiments.py
```

実行すると、`configs/publication.toml` の設定を $N = 20{,}000$、数値積分ドロー数 64 に置き換えて8つの制度シナリオを共通乱数で計算し、`results/simulation/policy_comparison.csv` と `results/simulation/figures/fig_policy_welfare.png` を出力する（所要時間は手元の環境で約2分）。

記事5章および6章の表に掲載した数値は、`results/simulation/policy_comparison.csv` のうち `permission_model == "integrated_cost"` の行に対応する。主な列の対応は以下の通りである。

| CSVの列名 | 記事での表記 | 現行制度（`current`）の値 |
|:---|:---|---:|
| `permission_rate` | 許可率 | 0.3589（35.9%） |
| `signing_rate` | 成立率 | 0.3501（35.0%） |
| `short_rate` | 短期契約率 / 短期率 | 0.09845（9.8%） |
| `mean_guarantee` | 平均保証額（百万ドル） | 12.883（約1287〜1288万ドル） |
| `mean_fee` | 平均譲渡金（百万ドル） | 2.568（約257万ドル） |
| `excess_short_rate` | 過剰短期率 | 0.02875（2.9%） |
| `origin_loss_given_permission` | 許可時の損失率 / NPB損失率 | 0.7308（73.1%） |
| `origin_welfare` | NPB球団の平均余剰 | −0.493 |
| `player_welfare` | 選手の平均余剰 | 6.916 |
| `mlb_welfare` | MLB球団の平均余剰 | 8.123 |
| `total_welfare` | 総余剰（3者合計） | 14.545 |

### 2. テストの実行

```bash
pytest -q
```

### 3. 実データ分析コード（記事7章）について

`scripts/fee_incidence_primary.py` は、記事7章で示した19件のポスティング契約について、WARあたり実質譲渡金の単純中央値差（並べ替え検定とブートストラップ信頼区間）、条件付きOLS（HC3）、Huber型ロバスト回帰、および1件除外（Leave-One-Out）範囲を計算するスクリプトである。

入力となる `data/posting_master.csv` と `data/transfer_player_stats.csv` には第三者が提供する成績予測（Steamer・ZiPS等の予測WAR）が含まれるため、本リポジトリには元データを同梱していない。そのため、このリポジトリ単体では `scripts/fee_incidence_primary.py` を再実行できない。手法の確認用にコードのみを公開し、出力された集計表を `results/empirical/table_fee_incidence_primary.csv` に収録している。

## 結果の解釈に関する留意点

- 5〜6章の数値は、明記した構造モデルとパラメータ設定のもとで生成した仮想母集団（$N = 20{,}000$）に対する数値実験の結果であり、現実のNPB・KBO球団における赤字率や政策効果を直接推定した値ではない。
- `total_welfare`（記事中の「総余剰」）は、モデル内の3主体（移籍元球団・選手・初回契約を結ぶMLB球団）の利得合計を指す。
- 7章の表は、ポスティングを許可されて契約が成立した19件の観測標本における記述的な比較であり、契約期間の無作為割当や因果効果の識別に基づくものではない。
