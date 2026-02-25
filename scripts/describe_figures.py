# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from run_pipeline import DATA_DIR, OUTPUT_DIR, normalize_missing, parse_docx_mappings, apply_mapping, build_ic_scores

ROOT = Path(__file__).resolve().parents[1]
FIG_MD = ROOT / "figures" / "FIGURE_NOTES.md"


def load_raw():
    xlsx = next(DATA_DIR.glob("*.xlsx"))
    docx = next(DATA_DIR.glob("*.docx"), None)
    df = pd.read_excel(xlsx)
    df = normalize_missing(df)
    mappings = parse_docx_mappings(docx, df.columns.tolist())
    return df, mappings


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_markdown(index=False)


def main():
    df, mappings = load_raw()
    n = len(df)

    lines = []
    lines.append("# Figure Notes (Text + Math)")
    lines.append("")
    lines.append("This file provides full interpretability for each figure. If fonts fail in images, use these tables and formulas.")
    lines.append("")

    # D1 Missing bins
    lines.append("## D1_missing_bins.png")
    lines.append("**Definition:** For column j, missing_rate_j = (1/n) * Σ_i 1[x_ij is missing].")
    lines.append("Columns are binned by missing_rate_j into intervals: <=5%, 5–20%, 20–50%, 50–80%, 80–95%, 95–100%.")
    miss_rate = df.isna().mean()
    bins = [0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0]
    labels = ["<=5%", "5-20%", "20-50%", "50-80%", "80-95%", "95-100%"]
    miss_bin = pd.cut(miss_rate, bins=bins, labels=labels, include_lowest=True).value_counts().reindex(labels, fill_value=0)
    lines.append("**Counts by bin:**")
    lines.append(md_table(miss_bin.rename("columns").reset_index().rename(columns={"index": "bin"})))
    lines.append("")

    # D1 categorical distributions
    cat_fields = ["性别", "婚姻状况", "文化程度", "受试来源", "民族", "经济状况", "家庭类型", "健康状态"]
    for col in cat_fields:
        if col not in df.columns:
            continue
        lines.append(f"## D1_{col}_dist.png")
        lines.append("**Definition:** count_c = Σ_i 1[x_i = c], proportion p_c = count_c / n.")
        s = df[col].copy()
        if col in mappings:
            s = apply_mapping(s, mappings[col])
        s = s.fillna("Missing").astype(str)
        freq = s.value_counts(normalize=True)
        rare = freq[freq < 0.005].index
        s = s.where(~s.isin(rare), "Other")
        vc = s.value_counts().rename("count").reset_index().rename(columns={"index": "category"})
        vc["proportion"] = vc["count"] / n
        lines.append(md_table(vc))
        lines.append("")

    # D1 region top15
    for col in ["省", "市", "区"]:
        if col not in df.columns:
            continue
        lines.append(f"## D1_{col}_top15.png")
        lines.append("**Definition:** count_r = Σ_i 1[x_i = r], proportion p_r = count_r / n.")
        s = df[col].fillna("Missing").astype(str)
        vc = s.value_counts().head(15).rename("count").reset_index().rename(columns={"index": col})
        vc["proportion"] = vc["count"] / n
        lines.append(md_table(vc))
        lines.append("")

    # D1 assessment time
    lines.append("## D1_assessment_time.png")
    if "出生日期" in df.columns and "年龄" in df.columns:
        birth = pd.to_datetime(df["出生日期"], errors="coerce")
        age = pd.to_numeric(df["年龄"], errors="coerce")
        assess = birth + pd.to_timedelta(age * 365.25, unit="D")
        assess = assess.dropna()
        lines.append("**Definition:** estimated_assessment_date_i = birthdate_i + age_i * 365.25 days.")
        lines.append("Histogram over month buckets.")
        vc = assess.dt.to_period("M").value_counts().sort_index().rename("count").reset_index().rename(columns={"index": "month"})
        vc["proportion"] = vc["count"] / vc["count"].sum()
        lines.append(md_table(vc))
    else:
        lines.append("Birthdate/age columns missing.")
    lines.append("")

    # D3 cluster sizes
    lines.append("## D3_umap_clusters.png")
    assign_path = OUTPUT_DIR / "D3_cluster_assignments.csv"
    if assign_path.exists():
        assign = pd.read_csv(assign_path)
        vc = assign["cluster_id"].value_counts().sort_index().rename("count").reset_index().rename(columns={"index": "cluster_id"})
        vc["proportion"] = vc["count"] / vc["count"].sum()
        lines.append("**Definition:** count_k = Σ_i 1[cluster_i = k], proportion p_k = count_k / n.")
        lines.append(md_table(vc))
    lines.append("")

    # D3 radar
    lines.append("## D3_radar_by_cluster.png")
    ic_path = OUTPUT_DIR / "D2_with_IC.csv"
    if ic_path.exists() and assign_path.exists():
        ic_df = pd.read_csv(ic_path)
        assign = pd.read_csv(assign_path)
        radar_cols = ["IC_sensory", "IC_vitality", "IC_locomotion", "IC_cognition", "IC_psychological"]
        radar = ic_df[radar_cols].groupby(assign["cluster_id"]).mean().reset_index()
        lines.append("**Definition:** mean_k,d = (1/n_k) * Σ_{i:cluster_i=k} IC_{i,d}.")
        lines.append(md_table(radar))
    lines.append("")

    # D4 SHAP summary (IC)
    lines.append("## D4_shap_summary_IC.png")
    imp_ic_path = OUTPUT_DIR / "D4_feature_importance_IC.csv"
    if imp_ic_path.exists():
        imp = pd.read_csv(imp_ic_path)
        lines.append("**Definition:** mean_abs_shap_f = (1/m) * Σ_i |SHAP_{i,f}|.")
        lines.append("Top features:")
        lines.append(md_table(imp.head(20)))
    lines.append("")

    # D4 SHAP summary (cluster)
    lines.append("## D4_shap_summary_cluster.png")
    imp_cl_path = OUTPUT_DIR / "D4_feature_importance_cluster.csv"
    if imp_cl_path.exists():
        imp = pd.read_csv(imp_cl_path)
        lines.append("**Definition:** mean_abs_shap_f = (1/m) * Σ_i |SHAP_{i,f}|.")
        lines.append("Top features:")
        lines.append(md_table(imp.head(20)))
    lines.append("")

    FIG_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
