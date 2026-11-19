"""Figures computed from record-level model outputs and analysis tables."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def curves(tables, path):
    fig, (roc, pr) = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, rows in tables.items():
        rows = rows.dropna(subset=["score"])
        if rows.label.nunique() < 2:
            continue
        fp, tp, _ = roc_curve(rows.label, rows.score)
        precision, recall, _ = precision_recall_curve(rows.label, rows.score)
        width = 2.5 if name == "full" else 1.4
        roc.plot(fp, tp, label=name, linewidth=width)
        pr.plot(recall, precision, label=name, linewidth=width)
    roc.plot([0, 1], [0, 1], ":", color="0.6")
    roc.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    pr.set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    pr.legend(fontsize=8)
    for ax in (roc, pr):
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
    save(fig, path)


def intervals(hours, path):
    fig, (hist, ecdf) = plt.subplots(1, 2, figsize=(8, 3.4))
    hist.hist(hours, bins="auto", color="#387988")
    hist.set(xlabel="Hours from ECG to arrest", ylabel="Cases")
    x = np.sort(hours)
    ecdf.step(np.r_[0, x], np.arange(len(x) + 1) / len(x), where="post", color="#255a70")
    ecdf.set(xlabel="Hours from ECG to arrest", ylabel="Cumulative fraction", ylim=(0, 1.03))
    save(fig, path)


def subgroups(table, path):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for i, row in table.reset_index(drop=True).iterrows():
        if np.isfinite(row.get("auroc", np.nan)):
            ax.plot([row.auroc_low, row.auroc_high], [i, i], color="#396f82")
            ax.plot(row.auroc, i, "o", color="#254958")
    ax.set_yticks(range(len(table)), table.subgroup)
    ax.invert_yaxis()
    ax.axvline(0.5, ls=":", color="0.6")
    ax.set(xlim=(0, 1.02), xlabel="AUROC with 95% interval")
    save(fig, path)


def prevalence(table, path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for cohort, frame in table.groupby("cohort"):
        for ax, key in zip(axes, ("ppv", "npv")):
            ax.plot(frame.incidence_per_1000, 100 * frame[key], ".-", label=cohort)
            ax.set(xlabel="IHCA events per 1,000 admissions", ylabel=key.upper() + " (%)")
    axes[-1].legend(fontsize=8)
    save(fig, path)


def ratios(table, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for key in ("auroc", "auprc"):
        ax.plot(table.controls_per_case, table[key], ".-", label=key.upper())
        ax.fill_between(table.controls_per_case, table[key + "_low"], table[key + "_high"], alpha=0.12)
    ax.set(
        xlabel="Controls per case",
        ylabel="Discrimination",
        ylim=(0, 1.03),
        xticks=list(table.controls_per_case),
    )
    ax.legend()
    save(fig, path)


def attribution(rank, individual, inputs, path):
    frame = rank.head(10).iloc[::-1]
    fig, (ax, detail) = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    positions = np.arange(len(frame))
    ax.barh(positions, frame.mean_abs_shap, color="#397887")
    ax.set_yticks(positions, frame.feature)
    ax.set_xlabel("Mean absolute SHAP value")
    rng = np.random.default_rng(0)
    for i, feature in enumerate(frame.feature):
        raw = inputs.loc[individual.index, feature].to_numpy()
        width = np.ptp(raw)
        colors = (raw - raw.min()) / width if width else np.full(len(raw), 0.5)
        scatter = detail.scatter(
            individual[feature],
            i + rng.uniform(-0.21, 0.21, len(raw)),
            c=colors,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=11,
            alpha=0.65,
        )
    detail.axvline(0, color="0.6", lw=1)
    detail.set_xlabel("Signed SHAP value (score)")
    fig.colorbar(scatter, ax=detail, label="Relative feature value", fraction=0.05)
    save(fig, path)
