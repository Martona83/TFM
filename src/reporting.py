from __future__ import annotations

from pathlib import Path
from typing import Any
import textwrap
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc


def configure_plot_style() -> None:
    """Apply a consistent, publication-friendly Viridis style using matplotlib."""
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linestyle": "-",
        "axes.titleweight": "semibold",
        "axes.titlesize": 11.5,
        "axes.titlepad": 11,
        "axes.labelsize": 9.8,
        "xtick.labelsize": 8.3,
        "ytick.labelsize": 8.3,
        "legend.fontsize": 8.2,
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
        "figure.constrained_layout.use": True,
    })


def _cmap(n: int):
    cmap = plt.get_cmap("viridis")
    if n <= 1:
        return [cmap(0.62)]
    return [cmap(float(i) / max(1, n - 1)) for i in range(n)]


def _short_label(value: Any, max_chars: int = 24, wrap: int | None = None) -> str:
    text = str(value)
    text = text.replace("_", " ")
    if len(text) > max_chars:
        text = text[: max(4, max_chars - 1)] + "…"
    if wrap:
        text = "\n".join(textwrap.wrap(text, width=wrap, break_long_words=False))
    return text


def _short_labels(values, max_chars: int = 24, wrap: int | None = None) -> list[str]:
    return [_short_label(v, max_chars=max_chars, wrap=wrap) for v in values]


def _style_axes(ax, *, xlabel: str | None = None, ylabel: str | None = None, title: str | None = None) -> None:
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_table(df: pd.DataFrame, paths: dict[str, Any], filename: str) -> Path:
    path = Path(paths["tables"]) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _running_in_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and "IPKernelApp" in getattr(shell, "config", {})
    except Exception:
        return False


def show_markdown(text: str) -> None:
    if _running_in_notebook():
        try:
            from IPython.display import Markdown, display
            display(Markdown(text))
            return
        except Exception:
            pass
    print(text)


def show_table(title: str, df: pd.DataFrame, max_rows: int | None = None) -> None:
    show_markdown(f"### {title}")
    shown = df.head(max_rows) if max_rows else df
    if _running_in_notebook():
        try:
            from IPython.display import display
            display(shown)
            return
        except Exception:
            pass
    print(shown.to_string(index=False))


def display_image(path: Path, caption: str | None = None) -> None:
    if caption:
        show_markdown(f"**{caption}**  \n`{path}`")
    try:
        from IPython.display import Image, display
        display(Image(filename=str(path)))
    except Exception:
        print(f"Figure saved: {path}")




def _finalize_layout(fig, *, bottom: float = 0.08, top: float = 0.94) -> None:
    """Compatibility helper for older plotting aliases."""
    try:
        fig.tight_layout(rect=(0, bottom, 1, top))
    except Exception:
        pass


def _save_fig(fig, paths: dict[str, Any], filename: str, display: bool = True, caption: str | None = None) -> Path:
    configure_plot_style()
    path = Path(paths["figures"]) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.set_constrained_layout(True)
    except Exception:
        pass
    fig.savefig(path, dpi=190, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    if display:
        display_image(path, caption or filename)
    return path


def _empty_figure(paths: dict[str, Any], filename: str, message: str, display: bool = True, caption: str | None = None) -> Path:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    ax.axis("off")
    return _save_fig(fig, paths, filename, display, caption or message)


def plot_variable_type_counts(schema_df: pd.DataFrame, paths: dict[str, Any], display: bool = True) -> Path:
    if schema_df.empty or "logical_type" not in schema_df.columns:
        return _empty_figure(paths, "fig_01_variable_type_counts.png", "No schema table available", display)
    counts = schema_df["logical_type"].value_counts().sort_values(ascending=True)
    colors = _cmap(len(counts))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.barh(_short_labels(counts.index, 26), counts.values, color=colors)
    _style_axes(ax, xlabel="Number of columns", title="Automatically extracted variable types")
    maxv = max(counts.values) if len(counts) else 1
    for i, v in enumerate(counts.values):
        ax.text(v + maxv * 0.02, i, str(int(v)), va="center", fontsize=9)
    return _save_fig(fig, paths, "fig_01_variable_type_counts.png", display, "Variable extraction: logical type counts")


def plot_missingness(schema_df: pd.DataFrame, paths: dict[str, Any], display: bool = True) -> Path:
    if schema_df.empty or "missing_rate" not in schema_df.columns:
        return _empty_figure(paths, "fig_02_missingness_by_column.png", "No missingness table available", display)
    df = schema_df.sort_values("missing_rate", ascending=False).head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.barh(_short_labels(df["column"], 28), df["missing_rate"].astype(float), color=plt.get_cmap("viridis")(0.58))
    _style_axes(ax, xlabel="Missing-value rate", title="Missingness by column — top 25")
    ax.set_xlim(0, max(0.02, min(1.0, float(df["missing_rate"].max()) * 1.15 if len(df) else 1)))
    return _save_fig(fig, paths, "fig_02_missingness_by_column.png", display, "Variable extraction: missingness")


def plot_dataset_flow(eligibility_df: pd.DataFrame, paths: dict[str, Any], display: bool = True) -> Path:
    if eligibility_df.empty:
        return _empty_figure(paths, "fig_03_dataset_flow.png", "No cohort-flow table available", display)
    labels = eligibility_df["rule"].astype(str).tolist()
    values = eligibility_df["n"].astype(float).tolist()
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.barh(y, values, color=_cmap(len(labels)))
    ax.set_yticks(y)
    ax.set_yticklabels(_short_labels(labels, 40, wrap=28))
    ax.invert_yaxis()
    _style_axes(ax, xlabel="Number of rows", title="Analytic-cohort / target construction")
    maxv = max(values) if values else 1
    for i, v in enumerate(values):
        ax.text(v + maxv * 0.01, i, f"{int(v):,}", va="center", fontsize=9)
    return _save_fig(fig, paths, "fig_03_dataset_flow.png", display, "Dataset flow")


def plot_target_distribution(analytic: pd.DataFrame, paths: dict[str, Any], display: bool = True) -> Path:
    if "target" not in analytic.columns:
        return _empty_figure(paths, "fig_04_target_distribution.png", "No target column has been configured yet", display)
    counts = analytic["target"].astype(int).value_counts().sort_index()
    labels = ["Negative / non-event", "Positive / event"]
    fig, ax = plt.subplots(figsize=(6.9, 4.4))
    colors = _cmap(len(counts))
    x = np.arange(len(counts))
    ax.bar(x, counts.values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[int(i)] if int(i) < len(labels) else str(i) for i in counts.index])
    _style_axes(ax, ylabel="Rows", title="Target distribution")
    maxv = max(counts.values) if len(counts) else 1
    for i, v in enumerate(counts.values):
        ax.text(i, v + maxv * 0.025, f"{int(v):,}\n{v / counts.sum():.1%}", ha="center", fontsize=9)
    return _save_fig(fig, paths, "fig_04_target_distribution.png", display, "EDA: target distribution")


def plot_numeric_distributions(analytic: pd.DataFrame, numeric_summary: pd.DataFrame, paths: dict[str, Any], display: bool = True) -> Path:
    if numeric_summary.empty or "feature" not in numeric_summary.columns:
        return _empty_figure(paths, "fig_05_numeric_distributions.png", "No numeric variables available", display)
    cols = [c for c in numeric_summary["feature"].tolist() if c in analytic.columns][:9]
    if not cols:
        return _empty_figure(paths, "fig_05_numeric_distributions.png", "No numeric variables available", display)
    n = len(cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.15 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    colors = _cmap(len(cols))
    for idx, (ax, col) in enumerate(zip(axes, cols)):
        x = pd.to_numeric(analytic[col], errors="coerce").dropna()
        ax.hist(x, bins=min(30, max(8, int(np.sqrt(max(len(x), 1))))), color=colors[idx], alpha=0.92)
        _style_axes(ax, ylabel="Count", title=_short_label(col, 26))
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Numeric variable distributions", y=0.995, fontsize=13, fontweight="semibold")
    return _save_fig(fig, paths, "fig_05_numeric_distributions.png", display, "EDA: numeric distributions")


def plot_categorical_distributions(analytic: pd.DataFrame, categorical_summary: pd.DataFrame, paths: dict[str, Any], display: bool = True) -> Path:
    if categorical_summary.empty or "feature" not in categorical_summary.columns:
        return _empty_figure(paths, "fig_06_categorical_distributions.png", "No categorical or discrete variables available", display)
    cols = [c for c in categorical_summary["feature"].tolist() if c in analytic.columns][:9]
    if not cols:
        return _empty_figure(paths, "fig_06_categorical_distributions.png", "No categorical or discrete variables available", display)
    n = len(cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.35 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    colors = _cmap(8)
    for ax, col in zip(axes, cols):
        counts = analytic[col].fillna("Missing").astype(str).value_counts().head(8).iloc[::-1]
        ax.barh(_short_labels(counts.index, 24), counts.values, color=colors[: len(counts)])
        _style_axes(ax, xlabel="Count", title=_short_label(col, 26))
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Categorical/discrete variable distributions", y=0.995, fontsize=13, fontweight="semibold")
    return _save_fig(fig, paths, "fig_06_categorical_distributions.png", display, "EDA: categorical distributions")


def plot_correlation_heatmap(analytic: pd.DataFrame, feature_cols: list[str], paths: dict[str, Any], display: bool = True) -> Path:
    numeric = [c for c in feature_cols if c in analytic.columns and pd.api.types.is_numeric_dtype(analytic[c])]
    if len(numeric) < 2:
        return _empty_figure(paths, "fig_07_numeric_correlation_heatmap.png", "Fewer than two numeric variables available", display)
    numeric = numeric[:22]
    corr = analytic[numeric].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(7.2, 0.48 * len(numeric)), max(6.2, 0.44 * len(numeric))))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(numeric)))
    ax.set_xticklabels(_short_labels(numeric, 18), rotation=45, ha="right")
    ax.set_yticks(np.arange(len(numeric)))
    ax.set_yticklabels(_short_labels(numeric, 20))
    _style_axes(ax, title="Numeric-variable correlation heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    return _save_fig(fig, paths, "fig_07_numeric_correlation_heatmap.png", display, "EDA: numeric correlation heatmap")


def plot_event_rates_by_sensitive(sensitive_summary: pd.DataFrame, paths: dict[str, Any], display: bool = True) -> Path:
    if sensitive_summary.empty:
        return _empty_figure(paths, "fig_08_event_rate_by_sensitive_group.png", "No sensitive attributes configured", display)
    df = sensitive_summary.copy()
    df["label"] = df["attribute"].astype(str) + "\n" + df["group"].astype(str)
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    x = np.arange(len(df))
    ax.bar(x, df["event_rate"].astype(float), color=_cmap(len(df)))
    ax.set_xticks(x)
    ax.set_xticklabels(_short_labels(df["label"], 24), rotation=45, ha="right")
    _style_axes(ax, ylabel="Positive target / event rate", title="Target rate by sensitive-attribute group")
    for i, row in df.iterrows():
        ax.text(i, float(row["event_rate"]) + 0.002, f"n={int(row['n'])}", ha="center", va="bottom", fontsize=8)
    return _save_fig(fig, paths, "fig_08_event_rate_by_sensitive_group.png", display, "EDA: event rate by sensitive group")


def plot_feature_target_associations(feature_policy_df: pd.DataFrame, paths: dict[str, Any], display: bool = True) -> Path:
    if feature_policy_df.empty or "target_association_strength" not in feature_policy_df.columns:
        return _empty_figure(paths, "fig_09_feature_target_associations.png", "No feature-association table available", display)
    df = feature_policy_df[feature_policy_df["selected"]].sort_values("target_association_strength", ascending=False).head(20).iloc[::-1]
    if df.empty:
        return _empty_figure(paths, "fig_09_feature_target_associations.png", "No selected features available", display)
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ax.barh(_short_labels(df["feature"], 30), df["target_association_strength"].astype(float), color=plt.get_cmap("viridis")(0.68))
    _style_axes(ax, xlabel="Association strength with target", title="Selected predictors ranked by target association")
    return _save_fig(fig, paths, "fig_09_feature_target_associations.png", display, "EDA: feature-target association")


def plot_proxy_heatmap(feature_policy_df: pd.DataFrame, paths: dict[str, Any], display: bool = True) -> Path:
    if feature_policy_df.empty or "max_sensitive_proxy_strength" not in feature_policy_df.columns:
        return _empty_figure(paths, "fig_10_sensitive_proxy_screen.png", "No proxy-screen table available", display)
    df = feature_policy_df[feature_policy_df["selected"]].sort_values("max_sensitive_proxy_strength", ascending=False).head(20).iloc[::-1]
    if df.empty:
        return _empty_figure(paths, "fig_10_sensitive_proxy_screen.png", "No selected features available", display)
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    vals = df["max_sensitive_proxy_strength"].astype(float).fillna(0)
    ax.barh(_short_labels(df["feature"], 30), vals, color=plt.get_cmap("viridis")(0.50))
    _style_axes(ax, xlabel="Maximum association with any sensitive attribute", title="Sensitive-proxy risk screen among selected predictors")
    return _save_fig(fig, paths, "fig_10_sensitive_proxy_screen.png", display, "EDA: sensitive proxy screen")


def plot_validation_model_comparison(validation_results: pd.DataFrame, paths: dict[str, Any], display: bool = True) -> Path:
    if validation_results.empty:
        return _empty_figure(paths, "fig_11_validation_model_comparison.png", "No validation results available", display)
    df = validation_results.sort_values("balanced_accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(_short_labels(df["model"], 18), df["balanced_accuracy"], color=_cmap(len(df)))
    _style_axes(ax, ylabel="Balanced accuracy", title="Validation balanced accuracy by model")
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    return _save_fig(fig, paths, "fig_11_validation_model_comparison.png", display, "Model training: validation comparison")


def plot_test_performance(test_performance: pd.DataFrame, paths: dict[str, Any], display: bool = True) -> Path:
    if test_performance.empty:
        return _empty_figure(paths, "fig_12_test_performance_comparison.png", "No test-performance results available", display)
    df = test_performance.sort_values("balanced_accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    width = 0.38
    x = np.arange(len(df))
    ax.bar(x - width / 2, df["balanced_accuracy"], width, label="Balanced accuracy", color=plt.get_cmap("viridis")(0.35))
    ax.bar(x + width / 2, df["roc_auc"], width, label="ROC-AUC", color=plt.get_cmap("viridis")(0.75))
    ax.set_xticks(x)
    ax.set_xticklabels(_short_labels(df["model"], 18), rotation=25, ha="right")
    _style_axes(ax, ylabel="Score", title="Held-out test performance")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    return _save_fig(fig, paths, "fig_12_test_performance_comparison.png", display, "Held-out test performance")


def plot_roc_curves(test_df: pd.DataFrame, test_probs: dict[str, np.ndarray], paths: dict[str, Any], display: bool = True) -> Path:
    if not test_probs:
        return _empty_figure(paths, "fig_13_roc_curves.png", "No predicted probabilities available", display)
    y_true = test_df["target"].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    colors = _cmap(len(test_probs))
    for (idx, (model, prob)) in enumerate(test_probs.items()):
        try:
            fpr, tpr, _ = roc_curve(y_true, prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, color=colors[idx], label=f"{_short_label(model, 18)} (AUC={roc_auc:.3f})")
        except Exception:
            continue
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", lw=1)
    _style_axes(ax, xlabel="False positive rate", ylabel="True positive rate", title="ROC curves on held-out test set")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=max(1, min(3, len(test_probs))), frameon=False)
    return _save_fig(fig, paths, "fig_13_roc_curves.png", display, "Visual diagnostics: ROC curves")


def plot_confusion_matrices(models: dict[str, Any], test_df: pd.DataFrame, test_probs: dict[str, np.ndarray], paths: dict[str, Any], display: bool = True) -> Path:
    if not models:
        return _empty_figure(paths, "fig_14_confusion_matrices.png", "No models available", display)
    names = list(models.keys())[:6]
    n = len(names)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.15 * ncols, 3.75 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    y_true = test_df["target"].astype(int).to_numpy()
    for ax, name in zip(axes, names):
        model = models[name]
        prob = test_probs[name]
        pred = (prob >= model.threshold).astype(int)
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        im = ax.imshow(cm, cmap="viridis")
        ax.set_title(_short_label(name, 20))
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.grid(False)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes[:len(names)], fraction=0.030, pad=0.04)
    if len(names) == 1:
        axes[0].set_title(f"Confusion matrix — {_short_label(names[0], 28)}", pad=14, fontsize=12.5, fontweight="semibold")
    else:
        fig.suptitle("Confusion matrices at selected thresholds", y=1.015, fontsize=13, fontweight="semibold")
    return _save_fig(fig, paths, "fig_14_confusion_matrices.png", display, "Visual diagnostics: confusion matrices")


def plot_probability_histograms(test_df: pd.DataFrame, test_probs: dict[str, np.ndarray], paths: dict[str, Any], display: bool = True) -> Path:
    if not test_probs:
        return _empty_figure(paths, "fig_15_probability_histograms.png", "No probabilities available", display)
    names = list(test_probs.keys())[:6]
    n = len(names)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.35 * ncols, 3.65 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    y = test_df["target"].astype(int).to_numpy()
    for ax, name in zip(axes, names):
        prob = np.asarray(test_probs[name])
        ax.hist(prob[y == 0], bins=20, alpha=0.72, label="True 0", color=plt.get_cmap("viridis")(0.28))
        ax.hist(prob[y == 1], bins=20, alpha=0.72, label="True 1", color=plt.get_cmap("viridis")(0.78))
        _style_axes(ax, xlabel="Predicted probability", ylabel="Count", title=_short_label(name, 20))
        ax.legend(frameon=False)
    for ax in axes[len(names):]:
        ax.axis("off")
    if len(names) == 1:
        axes[0].set_title(f"Predicted probabilities — {_short_label(names[0], 28)}", pad=14, fontsize=12.5, fontweight="semibold")
    else:
        fig.suptitle("Predicted-probability distributions", y=1.015, fontsize=13, fontweight="semibold")
    return _save_fig(fig, paths, "fig_15_probability_histograms.png", display, "Visual diagnostics: probability histograms")


def plot_threshold_sweep(threshold_sweep_df: pd.DataFrame, champion_model: str, paths: dict[str, Any], display: bool = True) -> Path:
    if threshold_sweep_df.empty or not champion_model:
        return _empty_figure(paths, "fig_16_threshold_sweep.png", "No threshold sweep available", display)
    df = threshold_sweep_df[threshold_sweep_df["model"] == champion_model]
    if df.empty:
        df = threshold_sweep_df.copy()
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(df["threshold"], df["balanced_accuracy"], marker="o", ms=3, label="Balanced accuracy", color=plt.get_cmap("viridis")(0.25))
    ax.plot(df["threshold"], df["f1"], marker="o", ms=3, label="F1", color=plt.get_cmap("viridis")(0.65))
    ax.plot(df["threshold"], df["selection_rate"], marker="o", ms=3, label="Selection rate", color=plt.get_cmap("viridis")(0.90))
    _style_axes(ax, xlabel="Decision threshold", ylabel="Metric value", title=f"Threshold sweep — {_short_label(champion_model, 22)}")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    return _save_fig(fig, paths, "fig_16_threshold_sweep.png", display, "Visual diagnostics: threshold sweep")


def plot_threshold_selection_all_models(threshold_sweep_df: pd.DataFrame, paths: dict[str, Any], display: bool = True) -> Path:
    """Display how the validation decision threshold was selected for every trained model."""
    if threshold_sweep_df.empty or "selected_threshold" not in threshold_sweep_df.columns:
        return _empty_figure(paths, "fig_16b_threshold_selection_all_models.png", "No per-model threshold-selection sweep available", display)
    names = list(pd.unique(threshold_sweep_df["model"]))[:8]
    if not names:
        return _empty_figure(paths, "fig_16b_threshold_selection_all_models.png", "No per-model threshold-selection sweep available", display)
    n = len(names)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.6 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    cmap = plt.get_cmap("viridis")
    for ax, name in zip(axes, names):
        df = threshold_sweep_df[threshold_sweep_df["model"] == name].sort_values("threshold")
        selected = float(df["selected_threshold"].iloc[0])
        selected_rows = df.loc[np.isclose(df["threshold"].astype(float), selected)]
        if selected_rows.empty:
            selected_y = float(df.iloc[(df["threshold"].astype(float) - selected).abs().argmin()]["balanced_accuracy"])
        else:
            selected_y = float(selected_rows.iloc[0]["balanced_accuracy"])
        ax.plot(df["threshold"], df["balanced_accuracy"], marker="o", ms=2.4, lw=1.4, label="Balanced accuracy", color=cmap(0.25))
        ax.plot(df["threshold"], df["f1"], marker="o", ms=2.4, lw=1.2, label="F1", color=cmap(0.65))
        ax.plot(df["threshold"], df["selection_rate"], marker="o", ms=2.0, lw=1.0, label="Selection rate", color=cmap(0.88), alpha=0.82)
        ax.axvline(selected, color=cmap(0.98), lw=1.8, ls="--", label="Selected threshold")
        ax.scatter([selected], [selected_y], s=56, color=cmap(0.98), edgecolor="black", linewidth=0.6, zorder=4)
        ax.text(selected, min(0.98, selected_y + 0.07), f"{selected:.3f}", ha="center", va="bottom", fontsize=8, color="#333333")
        subplot_title = _short_label(name, 26) if len(names) > 1 else f"Validation threshold selection — {_short_label(name, 28)}"
        _style_axes(ax, xlabel="Validation threshold", ylabel="Metric", title=subplot_title)
        ax.set_ylim(0, 1)
        ax.legend(frameon=True, framealpha=0.88, loc="lower right", fontsize=7.8)
    for ax in axes[len(names):]:
        ax.axis("off")
    if len(names) > 1:
        fig.suptitle("Validation threshold selection for each trained model", y=1.015, fontsize=13, fontweight="semibold")
    return _save_fig(fig, paths, "fig_16b_threshold_selection_all_models.png", display, "Visual diagnostics: per-model threshold selection")

def plot_fairness_gaps(gap_df: pd.DataFrame, champion_model: str, paths: dict[str, Any], display: bool = True) -> Path:
    if gap_df.empty:
        return _empty_figure(paths, "fig_17_baseline_fairness_gaps.png", "No fairness gaps available", display)
    df = gap_df.copy()
    if champion_model:
        df = df[df["model"] == champion_model]
    if df.empty:
        df = gap_df.copy()
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    x = np.arange(len(df))
    ax.bar(x - 0.18, df["fpr_gap"], 0.36, label="FPR gap", color=plt.get_cmap("viridis")(0.32))
    ax.bar(x + 0.18, df["fnr_gap"], 0.36, label="FNR gap", color=plt.get_cmap("viridis")(0.72))
    ax.set_xticks(x)
    ax.set_xticklabels(_short_labels(df["attribute"], 20), rotation=30, ha="right")
    _style_axes(ax, ylabel="Absolute gap", title="Baseline fairness gaps by sensitive attribute")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    return _save_fig(fig, paths, "fig_17_baseline_fairness_gaps.png", display, "Fairness audit: FPR/FNR gaps")


def plot_fairness_gap_heatmap(gap_df: pd.DataFrame, paths: dict[str, Any], display: bool = True) -> Path:
    if gap_df.empty:
        return _empty_figure(paths, "fig_18_fairness_gap_heatmap.png", "No fairness gaps available", display)
    pivot = gap_df.pivot_table(index="model", columns="attribute", values="combined_fpr_fnr_gap", aggfunc="mean")
    if pivot.empty:
        return _empty_figure(paths, "fig_18_fairness_gap_heatmap.png", "No fairness gaps available", display)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.1 * len(pivot.columns)), max(4.8, 0.55 * len(pivot.index))))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(_short_labels(pivot.columns, 18), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(_short_labels(pivot.index, 20))
    _style_axes(ax, title="Combined FPR+FNR fairness-gap heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Combined gap")
    return _save_fig(fig, paths, "fig_18_fairness_gap_heatmap.png", display, "Fairness audit: gap heatmap")


def plot_mitigation_summary(mitigation_summary: pd.DataFrame, paths: dict[str, Any], display: bool = True) -> Path:
    if mitigation_summary.empty:
        return _empty_figure(paths, "fig_19_mitigation_gap_change.png", "No mitigation summary available", display)
    df = mitigation_summary.sort_values("delta_combined_gap").head(30)
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    labels = df["model"].astype(str) + " / " + df.get("mitigation_method", pd.Series([""] * len(df))).astype(str) + " / " + df["attribute"].astype(str)
    ax.barh(_short_labels(labels, 34), df["delta_combined_gap"].astype(float), color=plt.get_cmap("viridis")(0.62))
    ax.axvline(0, color="#555555", lw=1)
    _style_axes(ax, xlabel="Mitigated minus original combined gap", title="Mitigation effect on combined fairness gap")
    return _save_fig(fig, paths, "fig_19_mitigation_gap_change.png", display, "Mitigation: gap change")


def plot_mitigation_group_rates(mitigation_group_df: pd.DataFrame, paths: dict[str, Any], display: bool = True) -> Path:
    if mitigation_group_df.empty:
        return _empty_figure(paths, "fig_20_mitigation_fpr_fnr_group_rates.png", "No mitigation group comparison available", display)
    df = mitigation_group_df.copy()
    for col in ["delta_fpr", "delta_fnr"]:
        if col not in df.columns:
            df[col] = 0.0
    df["absolute_rate_change"] = df["delta_fpr"].abs() + df["delta_fnr"].abs()
    df = df.sort_values("absolute_rate_change", ascending=False).head(24).reset_index(drop=True)
    def _method_caption(value: str) -> str:
        v = str(value)
        if "equalized_odds" in v:
            return "EO threshold"
        if "equal_opportunity" in v:
            return "EqOpp threshold"
        if "demographic_parity" in v:
            return "DP threshold"
        if "balanced_accuracy" in v:
            return "BA threshold"
        if "reweigh" in v:
            return "Reweighing"
        if "smoteenn" in v:
            return "SMOTEENN"
        if "smotenc" in v:
            return "SMOTENC"
        if "smoten" in v:
            return "SMOTEN"
        if "smote" in v:
            return "SMOTE"
        if "oversampling" in v:
            return "Oversampling"
        if "expgrad" in v:
            return "Fairlearn EG"
        return _short_label(v.replace("postprocess_", "post-").replace("preprocess_", "pre-"), 18)

    labels = (
        df["model"].astype(str).map(lambda x: _short_label(x, 14)) + "\n" +
        df["mitigation_method"].astype(str).map(_method_caption) + "\n" +
        df["attribute"].astype(str).map(lambda x: _short_label(x, 14)) + "=" +
        df["group"].astype(str).map(lambda x: _short_label(x, 14))
    )
    x = np.arange(len(df))
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(max(12.5, 0.55 * len(df)), 8.2), sharex=True, constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    axes[0].bar(x - width / 2, df["fpr_original"], width, label="Original FPR", color=cmap(0.22))
    axes[0].bar(x + width / 2, df["fpr_mitigated"], width, label="Mitigated FPR", color=cmap(0.55))
    _style_axes(axes[0], ylabel="FPR", title="FPR before and after mitigation")
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=True, loc="upper right", ncol=2)
    axes[1].bar(x - width / 2, df["fnr_original"], width, label="Original FNR", color=cmap(0.36))
    axes[1].bar(x + width / 2, df["fnr_mitigated"], width, label="Mitigated FNR", color=cmap(0.82))
    _style_axes(axes[1], ylabel="FNR", title="FNR before and after mitigation")
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=True, loc="upper right", ncol=2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(_short_labels(labels, 60, wrap=18), rotation=35, ha="right")
    return _save_fig(fig, paths, "fig_20_mitigation_fpr_fnr_group_rates.png", display, "Mitigation: subgroup FPR/FNR rates")

def plot_mitigation_heatmap(mitigation_summary: pd.DataFrame, paths: dict[str, Any], display: bool = True) -> Path:
    if mitigation_summary.empty:
        return _empty_figure(paths, "fig_21_mitigation_delta_heatmap.png", "No mitigation summary available", display)
    df = mitigation_summary.copy()
    if "mitigation_method" in df.columns:
        df["attribute_method"] = df["attribute"].astype(str) + " / " + df["mitigation_method"].astype(str)
        pivot = df.pivot_table(index="model", columns="attribute_method", values="delta_combined_gap", aggfunc="mean")
    else:
        pivot = df.pivot_table(index="model", columns="attribute", values="delta_combined_gap", aggfunc="mean")
    if pivot.empty:
        return _empty_figure(paths, "fig_21_mitigation_delta_heatmap.png", "No mitigation summary available", display)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.1 * len(pivot.columns)), max(4.8, 0.55 * len(pivot.index))))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(_short_labels(pivot.columns, 18), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(_short_labels(pivot.index, 20))
    _style_axes(ax, title="Mitigation delta heatmap: lower values mean smaller gap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Delta combined gap")
    return _save_fig(fig, paths, "fig_21_mitigation_delta_heatmap.png", display, "Mitigation: delta heatmap")


def build_manifest(paths: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for folder_key in ["tables", "figures"]:
        folder = Path(paths[folder_key])
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file():
                rows.append({"type": folder_key[:-1], "filename": path.name, "path": str(path), "size_bytes": int(path.stat().st_size)})
    return pd.DataFrame(rows)


def create_results_archive(paths: dict[str, Any], archive_name: str | None = None) -> Path:
    """Create a ZIP archive containing every exported table and figure."""
    import zipfile
    root = Path(paths["root"]).resolve()
    base = Path(paths.get("base", root.parent)).resolve()
    archive_path = base / (archive_name or f"{root.name}.zip")
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.resolve() != archive_path.resolve():
                zf.write(path, arcname=path.relative_to(root))
    return archive_path


def display_archive_link(archive_path: Path) -> None:
    """Display a downloadable link in Jupyter/Kaggle/local notebooks and trigger Colab download when available."""
    archive_path = Path(archive_path)
    show_markdown(f"**Results archive created:** `{archive_path}`")
    try:
        from IPython.display import FileLink, display
        display(FileLink(str(archive_path)))
    except Exception:
        print(f"Results archive: {archive_path}")
    try:
        from google.colab import files  # type: ignore
        files.download(str(archive_path))
    except Exception:
        # Local Jupyter and Kaggle expose the FileLink/sidebar output instead of browser-triggered download.
        pass



def export_results_archive(paths: dict[str, Any], display: bool = True, archive_name: str | None = None) -> Path:
    """Create a ZIP archive and display a notebook download link."""
    archive_path = create_results_archive(paths, archive_name=archive_name)
    if display:
        display_archive_link(archive_path)
    return archive_path





def plot_model_performance_summary(summary_df: pd.DataFrame, paths: dict[str, Any], display: bool = True) -> Path:
    """Plot the consolidated validation/test/fairness model summary."""
    if summary_df is None or summary_df.empty:
        return _empty_figure(paths, "fig_18b_model_performance_fairness_summary.png", "No consolidated model summary available", display)
    df = summary_df.copy().head(12)
    model_col = "model" if "model" in df.columns else df.columns[0]
    labels = _short_labels(df[model_col].astype(str), 24, wrap=16)
    x = np.arange(len(df))
    perf = pd.to_numeric(df.get("test_balanced_accuracy", df.get("validation_balanced_accuracy", pd.Series(np.nan, index=df.index))), errors="coerce")
    auc_values = pd.to_numeric(df.get("test_roc_auc", df.get("validation_roc_auc", pd.Series(np.nan, index=df.index))), errors="coerce")
    gap = pd.to_numeric(df.get("worst_attribute_combined_fpr_fnr_gap", df.get("max_combined_fpr_fnr_gap", pd.Series(np.nan, index=df.index))), errors="coerce")

    fig, axes = plt.subplots(2, 1, figsize=(max(9.8, 0.76 * len(df)), 8.4), sharex=True, constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    width = 0.34
    axes[0].bar(x - width / 2, perf, width, label="Balanced accuracy", color=cmap(0.30))
    axes[0].bar(x + width / 2, auc_values, width, label="ROC-AUC", color=cmap(0.68))
    axes[0].set_ylim(0, 1)
    _style_axes(axes[0], ylabel="Metric", title="Held-out performance")
    axes[0].legend(loc="upper right", ncol=1, frameon=True, framealpha=0.88)

    ymax = float(np.nanmax(gap)) if np.isfinite(gap).any() else 1.0
    axes[1].bar(x, gap, width=0.54, label="Worst combined FPR+FNR gap", color=cmap(0.82))
    axes[1].set_ylim(0, max(0.05, ymax * 1.15))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    _style_axes(axes[1], ylabel="Gap", title="Worst baseline fairness gap")
    axes[1].legend(loc="upper right", ncol=1, frameon=True, framealpha=0.88)
    # No global suptitle here: panel titles are clearer and avoid overlap in narrow notebook views.
    return _save_fig(fig, paths, "fig_18b_model_performance_fairness_summary.png", display, "Fairness audit: consolidated performance/fairness profile")


def plot_mitigation_combination_summary(combination_df: pd.DataFrame, paths: dict[str, Any], display: bool = True) -> Path:
    """Visual summary of the best model × mitigation-method candidates."""
    if combination_df is None or combination_df.empty:
        return _empty_figure(paths, "fig_22_mitigation_combination_ranking.png", "No mitigation-combination summary available", display)
    df = combination_df.copy().head(20).iloc[::-1]
    model = df.get("model", pd.Series([""] * len(df), index=df.index)).astype(str)
    family = df.get("mitigation_family", pd.Series([""] * len(df), index=df.index)).astype(str)
    method = df.get("mitigation_method", pd.Series([""] * len(df), index=df.index)).astype(str)
    labels = model + " | " + family + "\n" + method
    score_candidates = [
        ("selection_score", "Selection score: gap reduction penalised by accuracy loss"),
        ("candidate_score", "Candidate score: gap reduction penalised by accuracy loss"),
        ("combined_gap_reduction_sum", "Combined FPR+FNR gap reduction"),
        ("guarded_gap_utility_score", "Guarded fairness-utility score"),
        ("decision_score", "Decision score: gap reduction penalised by accuracy loss"),
        ("combined_gap_improvement", "Combined FPR+FNR gap improvement"),
    ]
    metric = next((m for m, _ in score_candidates if m in df.columns), None)
    xlabel = next((x for m, x in score_candidates if m == metric), "Mitigation-combination score")
    values = pd.to_numeric(df[metric], errors="coerce").fillna(0) if metric else pd.Series(0, index=df.index)
    colors = [plt.get_cmap("viridis")(0.72 if bool(v) else 0.45) for v in df.get("recommended_combination", pd.Series(False, index=df.index)).fillna(False)]
    fig, ax = plt.subplots(figsize=(11.4, max(5.2, 0.40 * len(df))), constrained_layout=True)
    ax.barh(_short_labels(labels, 48, wrap=28), values, color=colors)
    ax.axvline(0, color="#666666", lw=1)
    _style_axes(ax, xlabel=xlabel, title="Best mitigation/model combinations")
    return _save_fig(fig, paths, "fig_22_mitigation_combination_ranking.png", display, "Mitigation: combination ranking")


def plot_mitigation_combination_ranking(combination_df: pd.DataFrame, paths: dict[str, Any], display: bool = True) -> Path:
    """Backward-compatible alias used by older workflow cells."""
    return plot_mitigation_combination_summary(combination_df, paths, display)


def plot_mitigation_leaderboard(leaderboard_df: pd.DataFrame, paths: dict[str, Any], display: bool = True) -> Path:
    """Backward-compatible alias used by older workflow cells."""
    return plot_mitigation_combination_summary(leaderboard_df, paths, display)
