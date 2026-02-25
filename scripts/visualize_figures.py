# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

from umap import UMAP

# Reuse utilities from pipeline
from run_pipeline import (
    DATA_DIR,
    FIG_DIR,
    OUTPUT_DIR,
    normalize_missing,
    parse_docx_mappings,
    apply_mapping,
    compute_missing_bins,
    configure_plot_style,
    prepare_features,
    build_ic_scores,
    feature_importance_shap,
)


def load_font():
    font_paths = [
        r"C:\\Windows\\Fonts\\msyh.ttc",
        r"C:\\Windows\\Fonts\\msyhbd.ttc",
        r"C:\\Windows\\Fonts\\simhei.ttf",
        r"C:\\Windows\\Fonts\\simsun.ttc",
        r"C:\\Windows\\Fonts\\NotoSansSC-VF.ttf",
    ]
    chosen = None
    for p in font_paths:
        if Path(p).exists():
            try:
                font_manager.fontManager.addfont(p)
                chosen = p
                break
            except Exception:
                continue
    return font_manager.FontProperties(fname=chosen) if chosen else None


def apply_font(ax, font):
    if font is None:
        return
    ax.title.set_fontproperties(font)
    ax.xaxis.label.set_fontproperties(font)
    ax.yaxis.label.set_fontproperties(font)
    for t in ax.get_xticklabels():
        t.set_fontproperties(font)
    for t in ax.get_yticklabels():
        t.set_fontproperties(font)


def plot_bar(series: pd.Series, title: str, path: Path, xlabel: str = "", ylabel: str = "Count", font=None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=series.index.astype(str), y=series.values, color="#4C78A8", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    apply_font(ax, font)
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def load_raw():
    xlsx = next(DATA_DIR.glob("*.xlsx"))
    docx = next(DATA_DIR.glob("*.docx"), None)
    df = pd.read_excel(xlsx)
    df = normalize_missing(df)
    mappings = parse_docx_mappings(docx, df.columns.tolist())
    return df, mappings


def plot_d1(df: pd.DataFrame, mappings: dict[str, dict[str, str]], font=None):
    # Missing bins
    missing_rate = df.isna().mean().sort_values(ascending=False)
    missing_bins = compute_missing_bins(missing_rate)
    plot_bar(missing_bins, "Missing Rate Distribution", FIG_DIR / "D1_missing_bins.png", ylabel="Columns", font=font)

    # Categorical distributions
    cat_fields = ["性别", "婚姻状况", "文化程度", "受试来源", "民族", "经济状况", "家庭类型", "健康状态"]
    for col in cat_fields:
        if col not in df.columns:
            continue
        s = df[col].copy()
        if col in mappings:
            s = apply_mapping(s, mappings[col])
        s = s.fillna("Missing").astype(str)
        freq = s.value_counts(normalize=True)
        rare = freq[freq < 0.005].index
        s = s.where(~s.isin(rare), "Other")
        vc = s.value_counts().head(12)
        plot_bar(vc, f"{col} distribution", FIG_DIR / f"D1_{col}_dist.png", ylabel="Count", font=font)

    # Region distribution
    for col in ["省", "市", "区"]:
        if col not in df.columns:
            continue
        s = df[col].fillna("Missing").astype(str)
        vc = s.value_counts().head(15)
        plot_bar(vc, f"Top {col}", FIG_DIR / f"D1_{col}_top15.png", ylabel="Count", font=font)

    # Assessment time
    if "出生日期" in df.columns and "年龄" in df.columns:
        birth = pd.to_datetime(df["出生日期"], errors="coerce")
        age = pd.to_numeric(df["年龄"], errors="coerce")
        assess = birth + pd.to_timedelta(age * 365.25, unit="D")
        assess = assess.dropna()
        if not assess.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            assess.dt.to_period("M").value_counts().sort_index().plot(kind="bar", color="#59A14F", ax=ax)
            ax.set_title("Estimated Assessment Month (Birthdate + Age)")
            ax.set_xlabel("Month")
            ax.set_ylabel("Count")
            apply_font(ax, font)
            plt.tight_layout()
            fig.savefig(FIG_DIR / "D1_assessment_time.png", dpi=200)
            plt.close(fig)


def plot_d3(font=None):
    emb_path = OUTPUT_DIR / "D3_embeddings.parquet"
    assign_path = OUTPUT_DIR / "D3_cluster_assignments.csv"
    ic_path = OUTPUT_DIR / "D2_with_IC.csv"

    if not emb_path.exists() or not assign_path.exists() or not ic_path.exists():
        return

    emb = pd.read_parquet(emb_path)
    assign = pd.read_csv(assign_path)
    ic_df = pd.read_csv(ic_path)

    cluster_id = assign["cluster_id"].values
    emb_vecs = emb.drop(columns=["编号"], errors="ignore").values

    umap = UMAP(n_components=2, random_state=42)
    umap_2d = umap.fit_transform(emb_vecs)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(x=umap_2d[:, 0], y=umap_2d[:, 1], hue=cluster_id, palette="tab10", s=12, legend="full", ax=ax)
    ax.set_title("UMAP of Embeddings (colored by cluster)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, prop=font)
    apply_font(ax, font)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "D3_umap_clusters.png", dpi=200)
    plt.close(fig)

    # Radar plot for IC domains
    radar_cols = ["IC_sensory", "IC_vitality", "IC_locomotion", "IC_cognition", "IC_psychological"]
    radar = ic_df[radar_cols].groupby(cluster_id).mean()
    labels = radar_cols
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    for idx in radar.index:
        values = radar.loc[idx].tolist()
        values += values[:1]
        ax.plot(angles, values, label=f"Cluster {idx}")
        ax.fill(angles, values, alpha=0.1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("IC Domain Means by Cluster", fontproperties=font)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), prop=font)
    for t in ax.get_xticklabels():
        t.set_fontproperties(font)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "D3_radar_by_cluster.png", dpi=200)
    plt.close(fig)


def plot_d4(df: pd.DataFrame, mappings: dict[str, dict[str, str]], font=None):
    # recompute features for SHAP plots
    ic_df = build_ic_scores(df)
    X, _, _, _ = prepare_features(ic_df, mappings)

    _ = feature_importance_shap(X, ic_df["IC_level"], "IC")
    _ = feature_importance_shap(X, pd.read_csv(OUTPUT_DIR / "D3_cluster_assignments.csv")["cluster_id"], "cluster")

    # top feature bar (IC)
    imp_ic = pd.read_csv(OUTPUT_DIR / "D4_feature_importance_IC.csv")
    top = imp_ic.head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=top, y="feature", x="mean_abs_shap", color="#E15759", ax=ax)
    ax.set_title("Top Features by Mean |SHAP| (IC_level)")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_ylabel("Feature")
    apply_font(ax, font)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "D4_top_features_bar.png", dpi=200)
    plt.close(fig)


def main():
    configure_plot_style()
    font = load_font()
    df, mappings = load_raw()
    plot_d1(df, mappings, font=font)
    plot_d3(font=font)
    plot_d4(df, mappings, font=font)


if __name__ == "__main__":
    main()
