# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score

from umap import UMAP

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import shap
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
CONFIG_DIR = ROOT / "configs"
REPORT_DIR = ROOT / "reports"
OUTPUT_DIR = ROOT / "outputs"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

MISSING_TOKENS = {"ND", "UK", "NA", "N/A", "N\\A", "N-D", "N.D", "UNK", "UNKNOWN"}

NUMERIC_KEYWORDS = [
    "年龄", "BMI", "体重", "身高", "围", "血压", "血糖", "胆固醇", "甘油", "糖化",
    "次数", "数量", "年数", "得分", "总分", "时间", "速度", "评分", "剂量", "kg", "cm", "mm",
]

DROP_COLS = [
    "序号", "编号", "患者编码", "详细地址", "备注",
    "基本信息", "慢病收集", "辅助检查", "认知", "心理", "感知", "活力", "运动",
]

IC_CONFIG = {
    "cognition_cutoff": 24,
    "psych_cutoff": 5,
    "vitality_bmi_low": 18.5,
    "calf_circumference_low": 33,
    "sppb_cutoff": 9,
    "grip_male_low": 28,
    "grip_female_low": 18,
}


def ensure_dirs() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    CONFIG_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def configure_plot_style() -> None:
    # Force a CJK-capable font by registering known font files
    font_files = [
        r"C:\\Windows\\Fonts\\msyh.ttc",
        r"C:\\Windows\\Fonts\\msyhbd.ttc",
        r"C:\\Windows\\Fonts\\simhei.ttf",
        r"C:\\Windows\\Fonts\\simsun.ttc",
        r"C:\\Windows\\Fonts\\NotoSansSC-VF.ttf",
        r"C:\\Windows\\Fonts\\NotoSerifSC-VF.ttf",
    ]
    for f in font_files:
        try:
            if Path(f).exists():
                font_manager.fontManager.addfont(f)
        except Exception:
            pass

    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans SC", "Noto Serif SC", "SimSun"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in preferred if f in available), None)
    if chosen:
        plt.rcParams["font.family"] = chosen
        plt.rcParams["font.sans-serif"] = [chosen]
    else:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = preferred
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid")


def get_font_properties():
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


def apply_font_to_axes(ax, font):
    if font is None or ax is None:
        return
    ax.title.set_fontproperties(font)
    ax.xaxis.label.set_fontproperties(font)
    ax.yaxis.label.set_fontproperties(font)
    for t in ax.get_xticklabels():
        t.set_fontproperties(font)
    for t in ax.get_yticklabels():
        t.set_fontproperties(font)


def find_data_files():
    xlsx = next(DATA_DIR.glob("*.xlsx"))
    docx = next(DATA_DIR.glob("*.docx"), None)
    pdf = next(DATA_DIR.glob("*.pdf"), None)
    return xlsx, docx, pdf


def read_docx_text(docx_path: Path) -> str:
    with zipfile.ZipFile(docx_path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml)
    return "".join(texts)


def _extract_block(text: str, field: str, fields: list[str]) -> str | None:
    key = field + "："
    idx = text.find(key)
    if idx == -1:
        return None
    start = idx + len(key)
    next_idx = len(text)
    for f in fields:
        if f == field:
            continue
        j = text.find(f + "：", start)
        if j != -1 and j < next_idx:
            next_idx = j
    return text[start:next_idx]


def _parse_label_code(block: str) -> dict[str, str]:
    # label then code (e.g., 女0 男1)
    block = block.replace("，", ",").replace("、", " ").replace("：", " ")
    block = re.sub(r"\s+", " ", block)
    pairs = re.findall(r"([^0-9,]+?)\s*(\d+)", block)
    return {code.strip(): label.strip() for label, code in pairs if label.strip()}


def _parse_code_label(block: str) -> dict[str, str]:
    # code then label (e.g., 01"汉族")
    block = block.replace("，", ",")
    pairs = re.findall(r"(\d{1,3})\s*\"?([^\",]+?)\"?(?=,|$)", block)
    return {code.strip(): label.strip() for code, label in pairs if label.strip()}


def parse_docx_mappings(docx_path: Path, columns: list[str]) -> dict[str, dict[str, str]]:
    if docx_path is None:
        return {}

    text = read_docx_text(docx_path)
    fields = [
        "民族", "性别", "曾经职业", "一生主要从事", "人均可支配月收入水平", "经济状况", "受试来源",
        "婚姻状况", "家庭类型", "文化程度", "健康状态", "是否高血压", "血压治疗状态",
        "听力/视力障碍", "是否影响生活", "嗅觉", "症状", "感知", "营养", "步行是否完成", "步态异常",
    ]

    mappings: dict[str, dict[str, str]] = {}

    block = _extract_block(text, "民族", fields)
    if block:
        m = _parse_code_label(block)
        if "民族" in columns:
            mappings["民族"] = m

    for f in [
        "性别", "曾经职业", "一生主要从事", "经济状况", "受试来源", "婚姻状况", "家庭类型",
        "文化程度", "健康状态", "是否高血压", "血压治疗状态",
    ]:
        block = _extract_block(text, f, fields)
        if block:
            m = _parse_label_code(block)
            if f in columns:
                mappings[f] = m

    # income levels (labels include digits)
    block = _extract_block(text, "人均可支配月收入水平", fields)
    if block and "人均可支配月收入水平" in columns:
        block = block.replace("，", ",")
        pairs = re.findall(r"([<>≥=\d\-]+)\s*(\d+)", block)
        mappings["人均可支配月收入水平"] = {code.strip(): label.strip() for label, code in pairs}

    # hearing/vision impairment
    block = _extract_block(text, "听力/视力障碍", fields)
    if block:
        m = _parse_label_code(block)
        for col in ["听力障碍", "视力障碍"]:
            if col in columns:
                mappings[col] = m

    # impact on daily life
    block = _extract_block(text, "是否影响生活", fields)
    if block:
        m = _parse_label_code(block)
        for col in columns:
            if "是否影响日常" in col:
                mappings[col] = m

    # smell
    block = _extract_block(text, "嗅觉", fields)
    if block and "自我感觉嗅觉" in columns:
        mappings["自我感觉嗅觉"] = _parse_label_code(block)

    # symptom
    block = _extract_block(text, "症状", fields)
    if block:
        m = _parse_label_code(block)
        for col in columns:
            if "慢病症状" in col:
                mappings[col] = m

    # sensory screening
    block = _extract_block(text, "感知", fields)
    if block:
        # expected: 视力：通过1 不通过0 听力：通过1 不通过0
        if "感知-视力" in columns:
            mappings["感知-视力"] = {"1": "通过", "0": "不通过"}
        if "感知-听力" in columns:
            mappings["感知-听力"] = {"1": "通过", "0": "不通过"}

    # nutrition status
    block = _extract_block(text, "营养", fields)
    if block and "活力-营养描述结果" in columns:
        mappings["活力-营养描述结果"] = _parse_label_code(block)

    # gait abnormal
    block = _extract_block(text, "步态异常", fields)
    if block and "步态异常-编码" in columns:
        mappings["步态异常-编码"] = _parse_label_code(block)

    return mappings


def normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            s = df[col].astype(str).str.strip()
            mask = s.str.upper().isin(MISSING_TOKENS)
            df.loc[mask, col] = np.nan
            df.loc[s == "", col] = np.nan
    return df


def classify_columns(df: pd.DataFrame) -> dict[str, str]:
    types: dict[str, str] = {}
    for col in df.columns:
        s = df[col]
        if "日期" in col:
            dt = pd.to_datetime(s, errors="coerce")
            if dt.notna().mean() > 0.6:
                types[col] = "date"
                continue
        num = pd.to_numeric(s, errors="coerce")
        num_ratio = num.notna().mean()
        if num_ratio > 0.9:
            uniq = num.dropna().unique()
            if len(uniq) <= 10 and not any(k in col for k in NUMERIC_KEYWORDS):
                types[col] = "categorical"
            else:
                types[col] = "numeric"
        else:
            if s.nunique(dropna=True) <= 20:
                types[col] = "categorical"
            else:
                types[col] = "text"
    return types


def apply_mapping(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    return series.apply(lambda x: mapping.get(str(x).strip(), x))


def compute_missing_bins(missing_rate: pd.Series) -> pd.Series:
    bins = [0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0]
    labels = ["<=5%", "5-20%", "20-50%", "50-80%", "80-95%", "95-100%"]
    return pd.cut(missing_rate, bins=bins, labels=labels, include_lowest=True).value_counts().reindex(labels, fill_value=0)


def plot_bar(series: pd.Series, title: str, path: Path, xlabel: str = "", ylabel: str = "Count") -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=series.index.astype(str), y=series.values, color="#4C78A8", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    apply_font_to_axes(ax, get_font_properties())
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def data_audit_report(df: pd.DataFrame, mappings: dict[str, dict[str, str]], report_path: Path) -> dict:
    report = []
    n_rows, n_cols = df.shape

    id_col = "编号" if "编号" in df.columns else ("序号" if "序号" in df.columns else None)
    id_unique = df[id_col].nunique() if id_col else None
    id_missing = df[id_col].isna().sum() if id_col else None

    types = classify_columns(df)
    type_counts = pd.Series(types).value_counts().to_dict()

    report.append("# D1 Data Audit Report")
    report.append("")
    report.append("## Overview")
    report.append(f"- Rows: {n_rows}")
    report.append(f"- Columns: {n_cols}")
    if id_col:
        report.append(f"- ID column: `{id_col}` (unique={id_unique}, missing={id_missing})")
    report.append(f"- Column types: {type_counts}")

    # Missing analysis
    missing_rate = df.isna().mean().sort_values(ascending=False)
    top30 = missing_rate.head(30)
    missing_bins = compute_missing_bins(missing_rate)

    plot_bar(missing_bins, "Missing Rate Distribution", FIG_DIR / "D1_missing_bins.png", ylabel="Columns")

    report.append("")
    report.append("## Missingness")
    report.append("- Explicit missing tokens normalized: ND/UK/NA and empty strings")
    report.append("- Top 30 columns by missing rate:")
    report.append("")
    report.append(top30.to_frame("missing_rate").to_markdown())
    report.append("")
    report.append("- Missing-rate bins saved to `figures/D1_missing_bins.png`")

    # Outlier analysis for count-like fields
    count_keywords = ["次数", "数量", "年数"]
    count_cols = [c for c in df.columns if any(k in c for k in count_keywords)]
    for special in ["过去一年住院次数", "过去一年急诊次数", "服用药物数量", "慢病数量"]:
        if special in df.columns and special not in count_cols:
            count_cols.append(special)

    outlier_rows = []
    for col in count_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        p95 = s.quantile(0.95)
        p99 = s.quantile(0.99)
        maxv = s.max()
        count_gt = (s > p99).sum()
        outlier_rows.append([col, float(p95), float(p99), float(maxv), int(count_gt)])

    report.append("")
    report.append("## Outliers / Extreme Values")
    if outlier_rows:
        outlier_df = pd.DataFrame(outlier_rows, columns=["field", "p95", "p99", "max", "count_>p99"])
        report.append(outlier_df.to_markdown(index=False))
    else:
        report.append("No count-like fields found for outlier scan.")

    # Range checks for scale-like fields
    scale_cols = [c for c in df.columns if any(k in c for k in ["得分", "总分", "量表", "评分"])]
    range_rows = []
    for col in scale_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        minv, maxv = s.min(), s.max()
        if maxv <= 10:
            lo, hi = 0, 10
        elif maxv <= 100:
            lo, hi = 0, 100
        else:
            lo, hi = None, None
        if lo is not None:
            out = ((s < lo) | (s > hi)).sum()
            range_rows.append([col, minv, maxv, f"{lo}-{hi}", int(out)])
        else:
            range_rows.append([col, minv, maxv, "not_checked", "-"])

    report.append("")
    report.append("## Scale Range Checks")
    if range_rows:
        range_df = pd.DataFrame(range_rows, columns=["field", "min", "max", "expected_range", "out_of_range"])
        report.append(range_df.to_markdown(index=False))
    else:
        report.append("No scale-like fields found.")

    # Categorical distributions
    cat_fields = ["性别", "婚姻状况", "文化程度", "受试来源", "民族", "经济状况", "家庭类型", "健康状态"]
    report.append("")
    report.append("## Categorical Distributions")
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
        plot_bar(vc, f"{col} distribution", FIG_DIR / f"D1_{col}_dist.png", ylabel="Count")
        report.append(f"- `{col}` distribution saved to `figures/D1_{col}_dist.png`")

    # Region distribution
    report.append("")
    report.append("## Region Distribution")
    for col in ["省", "市", "区"]:
        if col not in df.columns:
            continue
        s = df[col].fillna("Missing").astype(str)
        missing_rate_col = (s == "Missing").mean()
        vc = s.value_counts().head(15)
        plot_bar(vc, f"Top {col}", FIG_DIR / f"D1_{col}_top15.png", ylabel="Count")
        report.append(f"- `{col}` missing rate: {missing_rate_col:.3f}, top 15 saved to `figures/D1_{col}_top15.png`")

    # Time inference
    report.append("")
    report.append("## Assessment Time Inference")
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
            apply_font_to_axes(ax, get_font_properties())
            plt.tight_layout()
            fig.savefig(FIG_DIR / "D1_assessment_time.png", dpi=200)
            plt.close(fig)
            report.append("- Estimated assessment month histogram saved to `figures/D1_assessment_time.png`")
        else:
            report.append("- Unable to infer assessment time (no valid birthdate/age).")
    else:
        report.append("- Birthdate/age columns not found for time inference.")

    report_path.write_text("\n".join(report), encoding="utf-8")

    return {
        "missing_rate": missing_rate,
        "type_counts": type_counts,
        "id_col": id_col,
    }


def build_ic_scores(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    def num(col):
        return pd.to_numeric(d[col], errors="coerce") if col in d.columns else pd.Series([np.nan] * len(d))

    # Sensory
    hearing_impair = (num("听力障碍") == 1) if "听力障碍" in d.columns else False
    vision_impair = (num("视力障碍") == 1) if "视力障碍" in d.columns else False
    hearing_screen = (num("感知-听力") == 0) if "感知-听力" in d.columns else False
    vision_screen = (num("感知-视力") == 0) if "感知-视力" in d.columns else False
    hearing_impact = (num("听力障碍是否影响日常") == 1) if "听力障碍是否影响日常" in d.columns else False
    vision_impact = (num("视力障碍是否影响日常") == 1) if "视力障碍是否影响日常" in d.columns else False
    sensory = hearing_impair | vision_impair | hearing_screen | vision_screen | hearing_impact | vision_impact

    # Vitality
    mna = num("活力-营养描述结果")
    appetite = num("活力-过去三个月内有没有因为食欲不振、消化问题、咀嚼或吞咽困难而摄食减少")
    weight_loss = num("活力-过去三个月内体重下降情况")
    bmi = num("活力-BMI值")
    calf = num("小腿围")

    vitality = (
        (mna <= 1) |
        (appetite <= 1) |
        (weight_loss <= 1) |
        (bmi < IC_CONFIG["vitality_bmi_low"]) |
        (calf < IC_CONFIG["calf_circumference_low"])
    )

    # Locomotion
    gait = (num("步态异常-编码") == 1) if "步态异常-编码" in d.columns else False
    walk_250 = (num("衰弱快速筛查量表-1您能步行250米么？") == 1) if "衰弱快速筛查量表-1您能步行250米么？" in d.columns else False
    sarc_walk = (num("肌少症评估-2步行穿过房间是否存在困难，是否需要帮助？") >= 1) if "肌少症评估-2步行穿过房间是否存在困难，是否需要帮助？" in d.columns else False
    sppb = (num("运动-总分") <= IC_CONFIG["sppb_cutoff"]) if "运动-总分" in d.columns else False

    # Grip strength
    if "Fried衰弱表型评估-握力左手最大值" in d.columns or "Fried衰弱表型评估-握力右手最大值" in d.columns:
        left = num("Fried衰弱表型评估-握力左手最大值")
        right = num("Fried衰弱表型评估-握力右手最大值")
        grip = pd.concat([left, right], axis=1).max(axis=1)
        sex = num("性别")
        grip_low = ((sex == 1) & (grip < IC_CONFIG["grip_male_low"])) | ((sex == 0) & (grip < IC_CONFIG["grip_female_low"]))
    else:
        grip_low = False

    locomotion = gait | walk_250 | sarc_walk | sppb | grip_low

    # Cognition
    cog = (num("认知-总分") < IC_CONFIG["cognition_cutoff"]) if "认知-总分" in d.columns else False

    # Psychological
    psych_score = (num("心理-总分") >= IC_CONFIG["psych_cutoff"]) if "心理-总分" in d.columns else False
    psych_dx = (num("是否焦虑抑郁症") == 1) if "是否焦虑抑郁症" in d.columns else False
    psych = psych_score | psych_dx

    d["IC_sensory"] = sensory.astype(int)
    d["IC_vitality"] = vitality.astype(int)
    d["IC_locomotion"] = locomotion.astype(int)
    d["IC_cognition"] = cog.astype(int)
    d["IC_psychological"] = psych.astype(int)

    d["IC_total"] = d[["IC_sensory", "IC_vitality", "IC_locomotion", "IC_cognition", "IC_psychological"]].sum(axis=1)

    def level_standard(x: int) -> str:
        if x <= 1:
            return "Low"
        if x <= 3:
            return "Medium"
        return "High"

    def level_inverse(x: int) -> str:
        if x <= 1:
            return "High"
        if x <= 3:
            return "Medium"
        return "Low"

    d["IC_level"] = d["IC_total"].apply(level_standard)
    d["IC_level_inv"] = d["IC_total"].apply(level_inverse)

    return d


def write_ic_rules(path: Path) -> None:
    rules = []
    rules.append("# D2 IC Proxy Rules (v0.1)")
    rules.append("")
    rules.append("## Overview")
    rules.append("This IC proxy uses 5 domains (sensory, vitality, locomotion, cognition, psychological). Each domain is flagged as impaired (1) if any proxy rule triggers. IC_total is the sum of impaired domains (0–5).")
    rules.append("")
    rules.append("## Domain Rules")
    rules.append("- Sensory: hearing/vision impairment (听力障碍/视力障碍 = 1) OR failed screening (感知-听力/感知-视力 = 0) OR impairment impacts daily life (听力/视力障碍是否影响日常 = 1).")
    rules.append("- Vitality: malnutrition or risk (活力-营养描述结果 <= 1) OR reduced intake (活力-过去三个月内...摄食减少 <= 1) OR weight loss (活力-过去三个月内体重下降情况 <= 1) OR BMI < 18.5 OR calf circumference < 33 cm.")
    rules.append("- Locomotion: gait abnormal (步态异常-编码 = 1) OR cannot walk 250m (衰弱快速筛查量表-1 = 1) OR sarcopenia walking difficulty (肌少症评估-2 >= 1) OR SPPB total (运动-总分 <= 9) OR low grip strength (male < 28, female < 18).")
    rules.append("- Cognition: cognitive total score (认知-总分 < 24).")
    rules.append("- Psychological: GDS-15 total (心理-总分 >= 5) OR diagnosis (是否焦虑抑郁症 = 1).")
    rules.append("")
    rules.append("## IC Level Mapping")
    rules.append("- IC_level: standard risk mapping by impairment count (0–1 Low, 2–3 Medium, 4–5 High).")
    rules.append("- IC_level_inv: inverse mapping (0–1 High, 2–3 Medium, 4–5 Low) retained for stakeholder comparison.")

    path.write_text("\n".join(rules), encoding="utf-8")


def prepare_features(df: pd.DataFrame, mappings: dict[str, dict[str, str]]):
    d = df.copy()

    # drop non-feature columns
    drop_cols = [c for c in DROP_COLS if c in d.columns]
    # drop IC-derived columns to avoid leakage
    ic_cols = [c for c in d.columns if c.startswith("IC_")] + [c for c in ["IC_total", "IC_level", "IC_level_inv"] if c in d.columns]
    drop_cols.extend(ic_cols)
    d = d.drop(columns=drop_cols, errors="ignore")

    # drop high-missing columns
    miss = d.isna().mean()
    d = d.loc[:, miss < 0.95]

    # identify numeric vs categorical
    numeric_cols = []
    categorical_cols = []
    for col in d.columns:
        s = d[col]
        num = pd.to_numeric(s, errors="coerce")
        num_ratio = num.notna().mean()
        if num_ratio > 0.9 and any(k in col for k in NUMERIC_KEYWORDS):
            numeric_cols.append(col)
        elif num_ratio > 0.9 and num.nunique(dropna=True) > 10:
            numeric_cols.append(col)
        else:
            # treat small integer codes as categorical
            if s.nunique(dropna=True) <= 20:
                categorical_cols.append(col)
            else:
                # fallback to numeric if convertible
                if num_ratio > 0.5:
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)

    # numeric processing
    num_df = d[numeric_cols].apply(pd.to_numeric, errors="coerce")
    num_df = num_df.fillna(num_df.median())
    scaler = StandardScaler()
    num_scaled = pd.DataFrame(scaler.fit_transform(num_df), columns=numeric_cols, index=d.index)

    # categorical processing
    cat_df = d[categorical_cols].copy()
    for col in cat_df.columns:
        if col in mappings:
            cat_df[col] = apply_mapping(cat_df[col], mappings[col])
    cat_df = cat_df.fillna("Missing").astype(str)

    # collapse rare categories
    for col in cat_df.columns:
        freq = cat_df[col].value_counts(normalize=True)
        rare = freq[freq < 0.005].index
        cat_df.loc[cat_df[col].isin(rare), col] = "Other"

    cat_dummies = pd.get_dummies(cat_df, prefix=cat_df.columns, prefix_sep="=")

    X = pd.concat([num_scaled, cat_dummies], axis=1)
    X = X.astype(float)

    return X, numeric_cols, categorical_cols, scaler


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z


def train_autoencoder(X: np.ndarray, latent_dim: int = 32, epochs: int = 40, batch_size: int = 256, lr: float = 1e-3):
    device = torch.device("cpu")
    model = AutoEncoder(X.shape[1], latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon, _ = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        _, z = model(torch.tensor(X, dtype=torch.float32).to(device))
    return z.cpu().numpy()


def cluster_and_visualize(df: pd.DataFrame, X: pd.DataFrame, ic_df: pd.DataFrame):
    # Baseline PCA embedding
    pca = PCA(n_components=min(50, X.shape[1]), random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X)

    # Deep embedding (AutoEncoder)
    emb = train_autoencoder(X.values, latent_dim=32, epochs=40)

    # Choose K by silhouette (KMeans)
    ks = range(2, 11)
    metrics = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(emb)
        sil = silhouette_score(emb, labels)
        ch = calinski_harabasz_score(emb, labels)
        # stability via multiple seeds
        aris = []
        for seed in [0, 1, 2, 3, 4]:
            km2 = KMeans(n_clusters=k, random_state=seed, n_init=10)
            lab2 = km2.fit_predict(emb)
            aris.append(adjusted_rand_score(labels, lab2))
        metrics.append([k, sil, ch, float(np.mean(aris))])

    metrics_df = pd.DataFrame(metrics, columns=["k", "silhouette", "calinski_harabasz", "stability_ari"])
    # Prefer 3-6 clusters to align with study design; fallback to global best if unavailable
    candidate = metrics_df[(metrics_df["k"] >= 3) & (metrics_df["k"] <= 6)]
    if not candidate.empty:
        best_k = candidate.sort_values(["silhouette", "stability_ari"], ascending=False).iloc[0]["k"]
    else:
        best_k = metrics_df.sort_values(["silhouette", "stability_ari"], ascending=False).iloc[0]["k"]

    km_final = KMeans(n_clusters=int(best_k), random_state=RANDOM_SEED, n_init=10)
    cluster_id = km_final.fit_predict(emb)

    # Save embeddings
    emb_df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    if "编号" in df.columns:
        emb_df.insert(0, "编号", df["编号"].values)
    emb_df.to_parquet(OUTPUT_DIR / "D3_embeddings.parquet", index=False)

    # UMAP visualization
    umap = UMAP(n_components=2, random_state=RANDOM_SEED)
    umap_2d = umap.fit_transform(emb)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(x=umap_2d[:, 0], y=umap_2d[:, 1], hue=cluster_id, palette="tab10", s=12, legend="full", ax=ax)
    ax.set_title("UMAP of Embeddings (colored by cluster)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, prop=get_font_properties())
    apply_font_to_axes(ax, get_font_properties())
    plt.tight_layout()
    fig.savefig(FIG_DIR / "D3_umap_clusters.png", dpi=200)
    plt.close(fig)

    # Cluster assignments
    assign = pd.DataFrame({
        "编号": df["编号"] if "编号" in df.columns else df.index,
        "cluster_id": cluster_id,
        "IC_total": ic_df["IC_total"],
        "IC_level": ic_df["IC_level"],
        "IC_level_inv": ic_df["IC_level_inv"],
    })
    assign.to_csv(OUTPUT_DIR / "D3_cluster_assignments.csv", index=False)

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
    ax.set_title("IC Domain Means by Cluster", fontproperties=get_font_properties())
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), prop=get_font_properties())
    for t in ax.get_xticklabels():
        t.set_fontproperties(get_font_properties())
    plt.tight_layout()
    fig.savefig(FIG_DIR / "D3_radar_by_cluster.png", dpi=200)
    plt.close(fig)

    # Cluster profiles (numeric + categorical)
    profile_path = OUTPUT_DIR / "D3_cluster_profiles.xlsx"
    with pd.ExcelWriter(profile_path, engine="openpyxl") as writer:
        sizes = pd.Series(cluster_id).value_counts().rename("count").to_frame()
        sizes.to_excel(writer, sheet_name="cluster_sizes")

        numeric_means = X.iloc[:, :].copy()
        numeric_means["cluster_id"] = cluster_id
        numeric_means = numeric_means.groupby("cluster_id").mean()
        numeric_means.to_excel(writer, sheet_name="feature_means")

    return cluster_id, metrics_df


def feature_importance_shap(X: pd.DataFrame, y: pd.Series, out_prefix: str):
    y_cat = pd.Categorical(y)
    y_codes = y_cat.codes
    n_classes = len(y_cat.categories)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        random_state=RANDOM_SEED,
        eval_metric="mlogloss",
    )
    model.fit(X, y_codes)

    sample_idx = np.random.choice(len(X), size=min(1000, len(X)), replace=False)
    X_sample = X.iloc[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        stacked = np.stack(shap_values, axis=0)
        shap_values_agg = np.mean(stacked, axis=0)
    else:
        shap_values_agg = shap_values

    importance = np.abs(shap_values_agg).mean(axis=0).ravel()
    n_features = X.shape[1]
    if importance.shape[0] != n_features:
        importance = importance[:n_features]
        shap_values_agg = shap_values_agg[:, :n_features]
    imp_df = pd.DataFrame({"feature": X.columns, "mean_abs_shap": importance}).sort_values("mean_abs_shap", ascending=False)

    imp_df.to_csv(OUTPUT_DIR / f"D4_feature_importance_{out_prefix}.csv", index=False)

    # SHAP summary plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values_agg, X_sample, show=False)
    ax = plt.gca()
    apply_font_to_axes(ax, get_font_properties())
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"D4_shap_summary_{out_prefix}.png", dpi=200)
    plt.close()

    return imp_df


def create_sample_cards(df: pd.DataFrame, ic_df: pd.DataFrame, cluster_id: np.ndarray, mappings: dict[str, dict[str, str]]):
    out = []
    sex_map = mappings.get("性别", {"0": "Female", "1": "Male"})

    for idx in df.index:
        rid = df.loc[idx, "编号"] if "编号" in df.columns else idx
        age = df.loc[idx, "年龄"] if "年龄" in df.columns else "NA"
        sex_val = df.loc[idx, "性别"] if "性别" in df.columns else "NA"
        sex = sex_map.get(str(sex_val).strip(), sex_val)
        ic_total = ic_df.loc[idx, "IC_total"]
        ic_level = ic_df.loc[idx, "IC_level"]
        domains = []
        for dcol, dname in [
            ("IC_sensory", "Sensory"),
            ("IC_vitality", "Vitality"),
            ("IC_locomotion", "Locomotion"),
            ("IC_cognition", "Cognition"),
            ("IC_psychological", "Psychological"),
        ]:
            if ic_df.loc[idx, dcol] == 1:
                domains.append(dname)

        out.append(f"ID: {rid}")
        out.append(f"Age: {age}")
        out.append(f"Sex: {sex}")
        out.append(f"IC_total: {ic_total}")
        out.append(f"IC_level: {ic_level}")
        out.append(f"Impaired domains: {', '.join(domains) if domains else 'None'}")
        out.append(f"Cluster: {cluster_id[idx]}")
        out.append("")

    (OUTPUT_DIR / "sample_card.txt").write_text("\n".join(out), encoding="utf-8")


def name_clusters(ic_df: pd.DataFrame, cluster_id: np.ndarray) -> None:
    radar_cols = ["IC_sensory", "IC_vitality", "IC_locomotion", "IC_cognition", "IC_psychological"]
    profile = ic_df[radar_cols].groupby(cluster_id).mean()

    names = []
    for cid, row in profile.iterrows():
        top = row.sort_values(ascending=False).head(2)
        top_domains = [c.replace("IC_", "").capitalize() for c in top.index]
        if top.mean() < 0.2:
            label = "Low-impairment"
        else:
            label = " + ".join(top_domains) + "-impaired"
        names.append((cid, label, top_domains))

    lines = ["# Cluster Naming (Rule-Based)", ""]
    for cid, label, doms in names:
        lines.append(f"- Cluster {cid}: {label} (top domains: {', '.join(doms)})")

    (REPORT_DIR / "D3_cluster_names.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_dirs()
    configure_plot_style()

    xlsx_path, docx_path, _ = find_data_files()
    df = pd.read_excel(xlsx_path)

    df = normalize_missing(df)
    mappings = parse_docx_mappings(docx_path, df.columns.tolist())

    # Save config
    config = {
        "random_seed": RANDOM_SEED,
        "missing_tokens": sorted(list(MISSING_TOKENS)),
        "ic_config": IC_CONFIG,
    }
    (CONFIG_DIR / "pipeline_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # D1
    data_audit_report(df, mappings, REPORT_DIR / "D1_data_audit_report.md")

    # D2
    ic_df = build_ic_scores(df)
    write_ic_rules(REPORT_DIR / "D2_IC_rules.md")
    ic_df.to_csv(OUTPUT_DIR / "D2_with_IC.csv", index=False)

    # D3
    X, num_cols, cat_cols, scaler = prepare_features(ic_df, mappings)
    cluster_id, metrics_df = cluster_and_visualize(ic_df, X, ic_df)
    metrics_df.to_csv(OUTPUT_DIR / "D3_cluster_metrics.csv", index=False)

    # D4
    _ = feature_importance_shap(X, ic_df["IC_level"], "IC")
    _ = feature_importance_shap(X, pd.Series(cluster_id, index=ic_df.index), "cluster")

    # Additional D4 summary bar for top features (IC)
    imp_ic = pd.read_csv(OUTPUT_DIR / "D4_feature_importance_IC.csv")
    top = imp_ic.head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=top, y="feature", x="mean_abs_shap", color="#E15759", ax=ax)
    ax.set_title("Top Features by Mean |SHAP| (IC_level)")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_ylabel("Feature")
    apply_font_to_axes(ax, get_font_properties())
    plt.tight_layout()
    fig.savefig(FIG_DIR / "D4_top_features_bar.png", dpi=200)
    plt.close(fig)

    # LLM-related outputs (rule-based)
    create_sample_cards(ic_df, ic_df, cluster_id, mappings)
    name_clusters(ic_df, cluster_id)


if __name__ == "__main__":
    main()
