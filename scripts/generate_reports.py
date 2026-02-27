# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import davies_bouldin_score, adjusted_rand_score
from sklearn.cluster import KMeans

import shap
from xgboost import XGBClassifier

from run_pipeline import (
    DATA_DIR,
    OUTPUT_DIR,
    REPORT_DIR,
    FIG_DIR,
    normalize_missing,
    parse_docx_mappings,
    apply_mapping,
    prepare_features,
    build_ic_scores,
    get_font_properties,
    apply_font_to_axes,
)

RANDOM_SEED = 42


def load_raw():
    xlsx = next(p for p in DATA_DIR.glob("*.xlsx") if not p.name.startswith("~$"))
    docx = next(DATA_DIR.glob("*.docx"), None)
    df = pd.read_excel(xlsx)
    df = normalize_missing(df)
    mappings = parse_docx_mappings(docx, df.columns.tolist())
    return df, mappings


def write_data_cleaning_spec(df: pd.DataFrame, mappings: dict[str, dict[str, str]]):
    # categorize columns similarly to pipeline
    X, numeric_cols, categorical_cols, _ = prepare_features(build_ic_scores(df), mappings)

    lines = []
    lines.append("# DATA_CLEANING_SPEC")
    lines.append("")
    lines.append("## Missing Encoding (Hard Rules)")
    lines.append("- Missing tokens treated as NA: ND, UK, NA, N/A, N\\A, N-D, N.D, UNK, UNKNOWN, empty string.")
    lines.append("- Literal string \"Missing\" is NOT treated as NA in raw data; it is a generated category label for categorical imputation.")
    lines.append("- Numeric value 0 is treated as valid unless explicitly missing (never converted to NA).")
    lines.append("")
    lines.append("## Missing Handling Strategy (Current Pipeline)")
    lines.append("- Columns with missing rate >= 95% are dropped before modeling.")
    lines.append("- Numeric features: median imputation.")
    lines.append("- Categorical features: Missing kept as its own category (""Missing"").")
    lines.append("- Missingness indicator variables: NOT used in current pipeline.")
    lines.append("")
    lines.append("## Alternative Strategies (Not Implemented)")
    lines.append("- Numeric: KNN imputation or model-based reconstruction (masked autoencoder).")
    lines.append("- Categorical: mode imputation instead of Missing category.")
    lines.append("- Add missingness indicator flags for high-missing features.")
    lines.append("")
    lines.append("## Categorical Encoding Rules")
    lines.append("- Encoding: one-hot (dummy variables).")
    lines.append("- Low-frequency categories (<0.5%) are merged into 'Other'.")
    lines.append("- One-hot naming rule: `<column>=<category>`.")
    lines.append("- Region fields (省/市/区): missing -> 'Missing' category.")
    lines.append("")
    lines.append("## Columns Treated as Categorical (post-cleaning)")
    lines.append("Total: %d" % len(categorical_cols))
    lines.append("\n".join([f"- {c}" for c in categorical_cols]))
    lines.append("")
    lines.append("## Columns Treated as Numeric (post-cleaning)")
    lines.append("Total: %d" % len(numeric_cols))
    lines.append("\n".join([f"- {c}" for c in numeric_cols]))
    lines.append("")
    lines.append("## Categorical Value Sets (after mapping + rare merge)")
    for col in categorical_cols:
        s = df[col].copy() if col in df.columns else None
        if s is None:
            continue
        if col in mappings:
            s = apply_mapping(s, mappings[col])
        s = s.fillna("Missing").astype(str)
        freq = s.value_counts(normalize=True)
        rare = freq[freq < 0.005].index
        s = s.where(~s.isin(rare), "Other")
        vals = sorted(s.unique())
        lines.append(f"### {col}")
        lines.append(", ".join(vals))

    (REPORT_DIR / "DATA_CLEANING_SPEC.md").write_text("\n".join(lines), encoding="utf-8")


def write_ic_rules_spec():
    lines = []
    lines.append("# IC_RULES_SPEC")
    lines.append("")
    lines.append("## IC Domains (5)")
    lines.append("- Sensory / Vitality / Locomotion / Cognition / Psychological")
    lines.append("")
    lines.append("## Trigger Rules (Proxy)")
    lines.append("**Sensory**: hearing/vision impairment (听力障碍=1 or 视力障碍=1) OR failed screening (感知-听力=0 or 感知-视力=0) OR impairment impacts daily life (听力/视力障碍是否影响日常=1).")
    lines.append("**Vitality**: 活力-营养描述结果 <=1 OR 摄食减少<=1 OR 体重下降情况<=1 OR BMI<18.5 OR 小腿围<33.")
    lines.append("**Locomotion**: 步态异常=1 OR 250m步行困难=1 OR 肌少症-步行困难>=1 OR 运动-总分<=9 OR 握力低 (男<28, 女<18).")
    lines.append("**Cognition**: 认知-总分 < 24.")
    lines.append("**Psychological**: 心理-总分 >= 5 OR 是否焦虑抑郁症 = 1.")
    lines.append("")
    lines.append("## Missing Handling")
    lines.append("- If a domain's contributing fields are missing, no impairment is triggered by that rule (conservative: not impaired).")
    lines.append("- Imputation used only for modeling; IC proxy uses raw values with missing-safe logic.")
    lines.append("")
    lines.append("## IC_total")
    lines.append("IC_total = Σ_d IC_domain_d, d in {sensory, vitality, locomotion, cognition, psychological}.")
    lines.append("")
    lines.append("## IC_level (Standard Risk)")
    lines.append("- 0–1: Low risk")
    lines.append("- 2–3: Medium risk")
    lines.append("- 4–5: High risk")
    lines.append("Rationale: severity increases with the count of impaired IC domains.")

    (REPORT_DIR / "IC_RULES_SPEC.md").write_text("\n".join(lines), encoding="utf-8")


def write_embedding_model_card(df: pd.DataFrame, mappings: dict[str, dict[str, str]]):
    ic_df = build_ic_scores(df)
    X, numeric_cols, categorical_cols, _ = prepare_features(ic_df, mappings)

    lines = []
    lines.append("# EMBEDDING_MODEL_CARD")
    lines.append("")
    lines.append("## Input Features")
    lines.append(f"- Total features after encoding: {X.shape[1]}")
    lines.append(f"- Numeric features: {len(numeric_cols)}")
    lines.append(f"- Categorical features (one-hot): {len(categorical_cols)}")
    lines.append("")
    lines.append("## Preprocessing")
    lines.append("- Numeric: z-score standardization (mean=0, std=1).")
    lines.append("- Categorical: one-hot encoding; rare categories (<0.5%) -> 'Other'.")
    lines.append("- Missing: numeric median; categorical 'Missing'.")
    lines.append("")
    lines.append("## Model")
    lines.append("- AutoEncoder (MLP)")
    lines.append("- Encoder: input -> 256 -> 128 -> latent")
    lines.append("- Decoder: latent -> 128 -> 256 -> input")
    lines.append("- Latent dimension: 32")
    lines.append("- Objective: reconstruction (MSE)")
    lines.append("- Epochs: 40; batch size: 256; lr: 1e-3")
    lines.append("- Random seed: 42")
    lines.append("")
    lines.append("## Stability (Current)")
    lines.append("- Cluster stability is evaluated via ARI across multiple random seeds in KMeans (see D3_cluster_metrics.csv).")
    lines.append("- Full embedding stability across multiple AutoEncoder trainings is not yet computed in this run.")

    (REPORT_DIR / "EMBEDDING_MODEL_CARD.md").write_text("\n".join(lines), encoding="utf-8")


def write_cluster_model_card(emb_path: Path):
    lines = []
    lines.append("# CLUSTER_MODEL_CARD")
    lines.append("")
    lines.append("## Embedding")
    lines.append("- Source: outputs/D3_embeddings.parquet (32D AutoEncoder embedding)")
    lines.append("")
    lines.append("## K Selection")
    metrics_path = OUTPUT_DIR / "D3_cluster_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        lines.append("- Metrics used: silhouette, Calinski–Harabasz, stability ARI.")
        lines.append("- K chosen: best within 3–6 by silhouette + stability.")
        lines.append("\n" + metrics.to_markdown(index=False))
    else:
        lines.append("- Metrics file not found.")

    if emb_path.exists():
        emb = pd.read_parquet(emb_path)
        X = emb.drop(columns=["编号"], errors="ignore").values
        rows = []
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
            labels = km.fit_predict(X)
            db = davies_bouldin_score(X, labels)
            rows.append([k, db])
        db_df = pd.DataFrame(rows, columns=["k", "davies_bouldin"])
        lines.append("")
        lines.append("## Davies–Bouldin Index (lower is better)")
        lines.append(db_df.to_markdown(index=False))

    assign_path = OUTPUT_DIR / "D3_cluster_assignments.csv"
    if assign_path.exists():
        assign = pd.read_csv(assign_path)
        vc = assign["cluster_id"].value_counts().sort_index().rename("count").reset_index().rename(columns={"index":"cluster_id"})
        lines.append("")
        lines.append("## Cluster Sizes")
        lines.append(vc.to_markdown(index=False))
        small = vc[vc["count"] < 50]
        if not small.empty:
            lines.append("\n## Small-Cluster Policy")
            lines.append("- Clusters with n<50 should be treated as rare/edge subtypes.")
            lines.append("- Avoid causal over-interpretation; consider merging by distance if needed for reporting.")

    (REPORT_DIR / "CLUSTER_MODEL_CARD.md").write_text("\n".join(lines), encoding="utf-8")


def leakage_and_sensitivity(df: pd.DataFrame, mappings: dict[str, dict[str, str]]):
    # leakage field list: any fields used in IC rules
    leakage_fields = [
        "听力障碍", "视力障碍", "听力障碍是否影响日常", "视力障碍是否影响日常",
        "感知-听力", "感知-视力",
        "活力-营养描述结果", "活力-过去三个月内有没有因为食欲不振、消化问题、咀嚼或吞咽困难而摄食减少",
        "活力-过去三个月内体重下降情况", "活力-BMI值", "小腿围",
        "步态异常-编码", "衰弱快速筛查量表-1您能步行250米么？",
        "肌少症评估-2步行穿过房间是否存在困难，是否需要帮助？", "运动-总分",
        "Fried衰弱表型评估-握力左手最大值", "Fried衰弱表型评估-握力右手最大值",
        "认知-总分", "心理-总分", "是否焦虑抑郁症",
        "活力-总分", "运动-总分", "Fried衰弱表型评估-总分", "肌少症评估-总分", "衰弱快速筛查量表-总分",
    ]

    # full model
    ic_df = build_ic_scores(df)
    X, _, _, _ = prepare_features(ic_df, mappings)

    y = ic_df["IC_level"]
    y_cat = pd.Categorical(y)
    y_codes = y_cat.codes

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(y_cat.categories),
        random_state=RANDOM_SEED,
        eval_metric="mlogloss",
    )
    model.fit(X, y_codes)

    sample_idx = np.random.choice(len(X), size=min(1000, len(X)), replace=False)
    X_sample = X.iloc[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values_agg = np.mean(np.stack(shap_values, axis=0), axis=0)
    else:
        shap_values_agg = shap_values
    if shap_values_agg.ndim == 3:
        shap_values_agg = shap_values_agg.mean(axis=2)

    imp_full = np.abs(shap_values_agg).mean(axis=0).ravel()
    if imp_full.shape[0] != X.shape[1]:
        imp_full = imp_full[: X.shape[1]]
        shap_values_agg = shap_values_agg[:, : X.shape[1]]
    imp_full_df = pd.DataFrame({"feature": X.columns, "mean_abs_shap": imp_full}).sort_values("mean_abs_shap", ascending=False)

    # remove leakage fields from features only (labels remain from full IC)
    df_noleak = df.drop(columns=[c for c in leakage_fields if c in df.columns], errors="ignore")
    ic_df_noleak = build_ic_scores(df_noleak)
    X_nl, _, _, _ = prepare_features(ic_df_noleak, mappings)

    y_nl = ic_df["IC_level"]
    y_nl_cat = pd.Categorical(y_nl)
    y_nl_codes = y_nl_cat.codes

    model_nl = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(y_nl_cat.categories),
        random_state=RANDOM_SEED,
        eval_metric="mlogloss",
    )
    model_nl.fit(X_nl, y_nl_codes)

    sample_idx = np.random.choice(len(X_nl), size=min(1000, len(X_nl)), replace=False)
    X_sample_nl = X_nl.iloc[sample_idx]

    explainer_nl = shap.TreeExplainer(model_nl)
    shap_values_nl = explainer_nl.shap_values(X_sample_nl)
    if isinstance(shap_values_nl, list):
        shap_values_nl = np.mean(np.stack(shap_values_nl, axis=0), axis=0)
    if shap_values_nl.ndim == 3:
        shap_values_nl = shap_values_nl.mean(axis=2)

    imp_nl = np.abs(shap_values_nl).mean(axis=0).ravel()
    if imp_nl.shape[0] != X_nl.shape[1]:
        imp_nl = imp_nl[: X_nl.shape[1]]
        shap_values_nl = shap_values_nl[:, : X_nl.shape[1]]
    imp_nl_df = pd.DataFrame({"feature": X_nl.columns, "mean_abs_shap": imp_nl}).sort_values("mean_abs_shap", ascending=False)

    imp_nl_df.to_csv(OUTPUT_DIR / "D4_feature_importance_IC_noleak.csv", index=False)

    # compare top20 overlap
    top_full = set(imp_full_df.head(20)["feature"])
    top_nl = set(imp_nl_df.head(20)["feature"])
    overlap = len(top_full.intersection(top_nl))
    jaccard = overlap / max(1, len(top_full.union(top_nl)))

    # write report
    lines = []
    lines.append("# LEAKAGE_CHECK")
    lines.append("")
    lines.append("## What Leakage Means (Plain Language)")
    lines.append("Leakage happens when the model uses variables that are already part of the definition of the label. This makes importance rankings look strong but circular, and reduces real-world interpretability.")
    lines.append("")
    lines.append("## IC Construction Fields")
    lines.append("\n".join([f"- {c}" for c in leakage_fields]))
    lines.append("")
    lines.append("## Sensitivity Analysis")
    lines.append("- Model A: all features (current pipeline)")
    lines.append("- Model B: remove IC construction fields and re-run importance")
    lines.append(f"- Top-20 overlap: {overlap} / 20, Jaccard={jaccard:.3f}")
    lines.append("")
    lines.append("### Top 20 (Model A)")
    lines.append(imp_full_df.head(20).to_markdown(index=False))
    lines.append("")
    lines.append("### Top 20 (Model B - No Leak)")
    lines.append(imp_nl_df.head(20).to_markdown(index=False))

    (REPORT_DIR / "LEAKAGE_CHECK.md").write_text("\n".join(lines), encoding="utf-8")

    return imp_full_df, X_sample, shap_values_agg, leakage_fields


def cluster_importance_noleak(df: pd.DataFrame, mappings: dict[str, dict[str, str]], leakage_fields: list[str]):
    # remove leakage fields from features
    df_nl = df.drop(columns=[c for c in leakage_fields if c in df.columns], errors="ignore")
    ic_df = build_ic_scores(df_nl)
    X, _, _, _ = prepare_features(ic_df, mappings)

    assign = pd.read_csv(OUTPUT_DIR / "D3_cluster_assignments.csv")
    y = assign["cluster_id"]
    y_cat = pd.Categorical(y)
    y_codes = y_cat.codes

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(y_cat.categories),
        random_state=RANDOM_SEED,
        eval_metric="mlogloss",
    )
    model.fit(X, y_codes)

    sample_idx = np.random.choice(len(X), size=min(1000, len(X)), replace=False)
    X_sample = X.iloc[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = np.mean(np.stack(shap_values, axis=0), axis=0)
    if shap_values.ndim == 3:
        shap_values = shap_values.mean(axis=2)

    importance = np.abs(shap_values).mean(axis=0).ravel()
    if importance.shape[0] != X.shape[1]:
        importance = importance[: X.shape[1]]

    imp_df = pd.DataFrame({"feature": X.columns, "mean_abs_shap": importance}).sort_values("mean_abs_shap", ascending=False)
    imp_df.to_csv(OUTPUT_DIR / "D4_feature_importance_cluster_noleak.csv", index=False)
    return imp_df


def dependence_plots(imp_df: pd.DataFrame, X: pd.DataFrame, shap_values: np.ndarray, top_n: int = 10):
    font = get_font_properties()
    top_feats = imp_df.head(top_n)["feature"].tolist()
    lines = []
    lines.append("# SHAP Dependence Notes (IC_level)")
    lines.append("")

    for i, feat in enumerate(top_feats, 1):
        if feat not in X.columns:
            continue
        idx = list(X.columns).index(feat)
        vals = X[feat].values
        shap_f = shap_values[:, idx]
        corr = np.corrcoef(vals, shap_f)[0, 1]
        direction = "positive" if corr >= 0 else "negative"

        # plot
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(vals, shap_f, s=8, alpha=0.5)
        ax.set_title(f"Dependence: {feat}")
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP value")
        apply_font_to_axes(ax, font)
        plt.tight_layout()
        fig_path = FIG_DIR / f"D4_dependence_IC_{i:02d}.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)

        lines.append(f"- {feat}: corr(feature, SHAP) = {corr:.3f} ({direction} association). See {fig_path.name}.")

    (REPORT_DIR / "SHAP_DEPENDENCE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def write_shap_guide():
    lines = []
    lines.append("# SHAP_INTERPRETATION_GUIDE")
    lines.append("")
    lines.append("## What is SHAP?")
    lines.append("SHAP (SHapley Additive exPlanations) assigns each feature a contribution to the model's prediction, based on game-theoretic Shapley values.")
    lines.append("")
    lines.append("## Mean |SHAP|")
    lines.append("mean(|SHAP|) = (1/m) * Σ_i |SHAP_{i,f}|. Larger values indicate stronger overall influence.")
    lines.append("")
    lines.append("## Scale Caution")
    lines.append("SHAP values are on the model-output scale (e.g., log-odds or probability). They are not fixed to 0–1. Compare within the same model only.")
    lines.append("")
    lines.append("## Interpretation Boundaries")
    lines.append("High importance for region or center indicates association, not causality. It may reflect center composition, sampling, or true underlying differences.")
    lines.append("")
    lines.append("## Directional Explanations")
    lines.append("See `reports/SHAP_DEPENDENCE_SUMMARY.md` and `figures/D4_dependence_IC_*.png` for directionality plots.")

    (REPORT_DIR / "SHAP_INTERPRETATION_GUIDE.md").write_text("\n".join(lines), encoding="utf-8")


def write_llm_spec():
    lines = []
    lines.append("# LLM_USAGE_SPEC")
    lines.append("")
    lines.append("## Tab2Text Template (Deterministic)")
    lines.append("```")
    lines.append("ID: {编号}")
    lines.append("Age: {年龄}")
    lines.append("Sex: {性别}")
    lines.append("IC_total: {IC_total}")
    lines.append("IC_level: {IC_level}")
    lines.append("Impaired domains: {domain_list}")
    lines.append("Cluster: {cluster_id}")
    lines.append("```")
    lines.append("")
    lines.append("## LLM Boundary")
    lines.append("- LLM is only used for naming/summarizing clusters based on statistical summaries.")
    lines.append("- No clinical advice, diagnosis, or treatment suggestions allowed.")
    lines.append("- Outputs must disclose: 'Based on statistical summaries, not clinical judgment.'")

    (REPORT_DIR / "LLM_USAGE_SPEC.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    df, mappings = load_raw()

    write_data_cleaning_spec(df, mappings)
    write_ic_rules_spec()
    write_embedding_model_card(df, mappings)
    write_cluster_model_card(OUTPUT_DIR / "D3_embeddings.parquet")

    imp_full_df, X_sample, shap_values, leakage_fields = leakage_and_sensitivity(df, mappings)
    dependence_plots(imp_full_df, X_sample, shap_values, top_n=10)
    _ = cluster_importance_noleak(df, mappings, leakage_fields)
    write_shap_guide()
    write_llm_spec()


if __name__ == "__main__":
    main()
