"""Credit card default PD scorecard on the UCI "Default of Credit Card Clients" dataset.

Pipeline: feature engineering -> quantile binning -> WOE / IV -> feature selection ->
logistic regression on WOE -> validation (AUC, KS, Gini, PSI) -> points-based scorecard ->
cutoff analysis (approval rate vs. bad rate).

Protected attributes (SEX, MARRIAGE, AGE) are excluded from the model on fair-lending grounds.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
TARGET = "default"
N_BINS = 6
IV_MIN = 0.02
PDO, BASE_SCORE, BASE_ODDS = 20, 600, 50  # 600 points = 50:1 good:bad odds; +20 points doubles the odds


def load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "credit_default.csv")
    df = df.rename(columns={"default payment next month": TARGET, "PAY_0": "PAY_1"})
    return df.drop(columns=["ID"])


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    pay = [f"PAY_{i}" for i in range(1, 7)]
    bill = [f"BILL_AMT{i}" for i in range(1, 7)]
    amt = [f"PAY_AMT{i}" for i in range(1, 7)]
    f = pd.DataFrame(index=df.index)
    f["limit_bal"] = df["LIMIT_BAL"]
    f["pay_status_latest"] = df["PAY_1"]
    f["max_delinq_6m"] = df[pay].max(axis=1)
    f["months_delinq_6m"] = (df[pay] > 0).sum(axis=1)
    f["util_latest"] = (df["BILL_AMT1"] / df["LIMIT_BAL"]).clip(-1, 3)
    f["util_avg_6m"] = (df[bill].mean(axis=1) / df["LIMIT_BAL"]).clip(-1, 3)
    bill_pos = df[bill].clip(lower=1)
    f["pay_ratio_latest"] = (df["PAY_AMT1"] / bill_pos["BILL_AMT1"]).clip(0, 5)
    f["pay_ratio_avg_6m"] = (df[amt].sum(axis=1) / bill_pos.sum(axis=1)).clip(0, 5)
    f["bill_trend_6m"] = ((df["BILL_AMT1"] - df["BILL_AMT6"]) / df["LIMIT_BAL"]).clip(-3, 3)
    f["education"] = df["EDUCATION"].where(df["EDUCATION"].isin([1, 2, 3]), 4)
    f[TARGET] = df[TARGET]
    return f


def bin_edges(x: pd.Series) -> np.ndarray:
    if x.nunique() <= N_BINS:
        vals = np.sort(x.unique())
        return np.concatenate([[-np.inf], (vals[:-1] + vals[1:]) / 2, [np.inf]])
    q = np.unique(np.quantile(x, np.linspace(0, 1, N_BINS + 1)[1:-1]))
    return np.concatenate([[-np.inf], q, [np.inf]])


def woe_table(x: pd.Series, y: pd.Series, edges: np.ndarray) -> pd.DataFrame:
    b = pd.cut(x, edges, include_lowest=True)
    t = pd.DataFrame({"bin": b, "y": y}).groupby("bin", observed=False)["y"].agg(["count", "sum"])
    t.columns = ["n", "bad"]
    t["good"] = t["n"] - t["bad"]
    dist_bad = (t["bad"] + 0.5) / (t["bad"].sum() + 0.5)
    dist_good = (t["good"] + 0.5) / (t["good"].sum() + 0.5)
    t["bad_rate"] = t["bad"] / t["n"].clip(lower=1)
    t["woe"] = np.log(dist_good / dist_bad)
    t["iv"] = (dist_good - dist_bad) * t["woe"]
    return t


def to_woe(x: pd.Series, edges: np.ndarray, table: pd.DataFrame) -> pd.Series:
    b = pd.cut(x, edges, include_lowest=True)
    return b.map(table["woe"]).astype(float)


def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def psi(expected, actual, bins=10) -> float:
    q = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    e = np.histogram(expected, q)[0] / len(expected)
    a = np.histogram(actual, q)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def main():
    data = engineer(load())
    feats = [c for c in data.columns if c != TARGET]
    train, test = train_test_split(data, test_size=0.3, stratify=data[TARGET], random_state=42)

    edges, tables, iv = {}, {}, {}
    for c in feats:
        edges[c] = bin_edges(train[c])
        tables[c] = woe_table(train[c], train[TARGET], edges[c])
        iv[c] = float(tables[c]["iv"].sum())
    iv = dict(sorted(iv.items(), key=lambda kv: -kv[1]))
    selected = [c for c, v in iv.items() if v >= IV_MIN]

    Xtr = pd.DataFrame({c: to_woe(train[c], edges[c], tables[c]) for c in selected})
    Xte = pd.DataFrame({c: to_woe(test[c], edges[c], tables[c]) for c in selected})
    model = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, train[TARGET])
    p_tr, p_te = model.predict_proba(Xtr)[:, 1], model.predict_proba(Xte)[:, 1]

    # Points: score = offset - factor * ln(odds_bad); WOE is defined as ln(good/bad), so coefficients are negative.
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    n = len(selected)
    points = {}
    for c, beta in zip(selected, model.coef_[0]):
        t = tables[c].copy()
        t["points"] = np.round(-(beta * t["woe"] + model.intercept_[0] / n) * factor + offset / n).astype(int)
        points[c] = t
    def score(df):
        return sum(pd.cut(df[c], edges[c], include_lowest=True).map(points[c]["points"]).astype(int) for c in selected)
    s_tr, s_te = score(train), score(test)

    metrics = {
        "rows": len(data), "default_rate": round(float(data[TARGET].mean()), 4),
        "features_engineered": len(feats), "features_selected": selected,
        "auc_train": round(roc_auc_score(train[TARGET], p_tr), 4),
        "auc_test": round(roc_auc_score(test[TARGET], p_te), 4),
        "ks_test": round(ks_stat(test[TARGET], p_te), 4),
        "gini_test": round(2 * roc_auc_score(test[TARGET], p_te) - 1, 4),
        "psi_score_train_vs_test": round(psi(s_tr, s_te), 4),
        "iv": {k: round(v, 4) for k, v in iv.items()},
    }

    # Cutoff analysis on the test set: approve everyone at or above the score cutoff.
    rows = []
    for cut in range(int(np.percentile(s_te, 5)), int(np.percentile(s_te, 95)) + 1, 5):
        appr = s_te >= cut
        rows.append({"cutoff": cut, "approval_rate": appr.mean(), "bad_rate_approved": test.loc[appr, TARGET].mean(),
                     "bads_avoided_pct": 1 - test.loc[appr, TARGET].sum() / test[TARGET].sum()})
    cut_df = pd.DataFrame(rows)
    cut_df.to_csv(OUT / "cutoff_analysis.csv", index=False)
    base_bad = test[TARGET].mean()
    at80 = cut_df.iloc[(cut_df["approval_rate"] - 0.80).abs().argmin()]
    metrics["cutoff_at_80pct_approval"] = {
        "cutoff": int(at80["cutoff"]), "approval_rate": round(float(at80["approval_rate"]), 3),
        "bad_rate_approved": round(float(at80["bad_rate_approved"]), 4),
        "portfolio_bad_rate": round(float(base_bad), 4),
        "relative_bad_rate_reduction": round(float(1 - at80["bad_rate_approved"] / base_bad), 3),
        "bads_avoided_pct": round(float(at80["bads_avoided_pct"]), 3),
    }

    # Score bands for monitoring.
    bands = pd.qcut(s_te, 10, duplicates="drop")
    band_df = test.groupby(bands, observed=False)[TARGET].agg(["count", "mean"]).rename(columns={"mean": "bad_rate"})
    band_df.to_csv(OUT / "score_deciles.csv")
    pd.concat({c: points[c][["n", "bad_rate", "woe", "iv", "points"]] for c in selected}).to_csv(OUT / "scorecard_points.csv")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    fpr, tpr, _ = roc_curve(test[TARGET], p_te)
    ax[0].plot(fpr, tpr, label=f"AUC {metrics['auc_test']:.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="grey")
    ax[0].set(title="ROC (test)", xlabel="False positive rate", ylabel="True positive rate")
    ax[0].legend()
    ax[1].plot(cut_df["approval_rate"], cut_df["bad_rate_approved"])
    ax[1].axhline(base_bad, ls="--", color="grey", label="Portfolio bad rate")
    ax[1].set(title="Approval rate vs. bad rate (test)", xlabel="Approval rate", ylabel="Bad rate of approved")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "validation.png", dpi=150)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
