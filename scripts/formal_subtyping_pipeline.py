# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

from scipy.stats import kruskal, spearmanr

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from umap import UMAP

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoModel, AutoTokenizer

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from run_pipeline import normalize_missing, parse_docx_mappings, apply_mapping  # noqa: E402


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "formal_subtyping"
FIG_DIR = ROOT / "figures" / "formal_subtyping"
REPORT_DIR = ROOT / "reports" / "formal_subtyping"
CONFIG_DIR = ROOT / "configs" / "formal_subtyping"

RANDOM_SEED = 42
EMBED_DIM = 128
K_RANGE = [3, 4, 5, 6]
BERT_MODEL = "bert-base-chinese"

CORE_FEATURES = [
    "年龄", "性别", "婚姻状况", "家庭类型", "文化程度", "经济状况", "受试来源", "健康状态", "健康自评分数",
    "患有慢性病数量", "服用药物数量", "过去一年住院次数", "过去一年急诊次数", "查尔森合并症得分",
    "是否高血压", "是否糖尿病", "是否骨关节病", "是否骨质疏松", "是否脑血栓", "是否焦虑抑郁症", "是否白内障", "是否青光眼",
    "听力障碍", "听力障碍是否影响日常", "视力障碍", "视力障碍是否影响日常", "自我感觉嗅觉",
    "感知-视力", "感知-听力",
    "认知-总分", "心理-总分",
    "活力-过去三个月内有没有因为食欲不振、消化问题、咀嚼或吞咽困难而摄食减少",
    "活力-过去三个月内体重下降情况", "活力-活动能力", "活力-既往3个月内有无重大心理变化或急性疾病",
    "活力-神经心理问题", "活力-BMI得分", "活力-BMI值", "活力-如不能取得BMI，请以下面小腿围代替问题6",
    "活力-总分", "活力-营养描述结果",
    "体重", "身高", "小腿围",
    "运动-双脚并拢站立10秒-保持时间为:", "运动-双脚半前后位站立10秒-保持时间为:", "运动-双脚前后成一直线站立10秒-保持时间为:",
    "运动-从椅子上站起，让其尽快地起立5次-完成时间为", "运动-从椅子上站起，让其尽快地起立5次-完成次数为",
    "运动-4米行走时间：日常行走速度走完4米-平均完成时间", "运动是否完成编码", "运动-总分", "步态异常-编码",
    "Fried衰弱表型评估-4米行走第一次", "Fried衰弱表型评估-4米行走第二次",
    "Fried衰弱表型评估-握力左手最大值", "Fried衰弱表型评估-握力右手最大值",
    "生活行为与社会功能评估-7您是否有吸烟习惯？", "生活行为与社会功能评估-8您是否喝酒？",
]

EXTERNAL_FEATURES = [
    "ADL量表-总分", "IADL量表-总分", "肌少症评估-总分", "Fried衰弱表型评估-总分",
    "衰弱快速筛查量表-总分", "跌倒评估-总分", "弹性评估-总分",
    "生活行为与社会功能评估-总分",
    "生命质量评估-1）总体来讲，您的健康状况是",
    "生命质量评估-10）在过去4个星期里，您有多少时间感到精力充沛？",
    "生命质量评估-11）在过去4个星期里，您有多少时间感到心情不好、闷闷不乐或沮丧？",
    "生命质量评估-12）在过去4个星期里，有多少时间由于您身体健康或情绪问题而妨碍您的社交活动（比如探亲、访友等）？",
]

TEXT_SECTIONS = {
    "Demographics": ["年龄", "性别", "婚姻状况", "家庭类型", "文化程度", "经济状况", "健康状态", "健康自评分数"],
    "Burden": ["患有慢性病数量", "服用药物数量", "过去一年住院次数", "过去一年急诊次数", "查尔森合并症得分"],
    "Key diseases": ["是否高血压", "是否糖尿病", "是否骨关节病", "是否骨质疏松", "是否脑血栓", "是否焦虑抑郁症", "是否白内障", "是否青光眼"],
    "Sensory": ["听力障碍", "听力障碍是否影响日常", "视力障碍", "视力障碍是否影响日常", "感知-视力", "感知-听力", "自我感觉嗅觉"],
    "Cognition": ["认知-总分"],
    "Psychological": ["心理-总分"],
    "Vitality": ["活力-总分", "活力-营养描述结果", "活力-BMI值", "小腿围"],
    "Locomotion": ["运动-总分", "运动是否完成编码", "步态异常-编码", "运动-4米行走时间：日常行走速度走完4米-平均完成时间", "Fried衰弱表型评估-握力左手最大值", "Fried衰弱表型评估-握力右手最大值"],
}


@dataclass
class ClusterEval:
    method: str
    k: int
    labels: np.ndarray
    silhouette: float
    calinski: float
    db: float
    stability_ari: float
    external_separation: float
    ic_cont_corr: float


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def configure_plot_style() -> None:
    font_files = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
    ]
    for f in font_files:
        try:
            if Path(f).exists():
                font_manager.fontManager.addfont(f)
        except Exception:
            pass
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans SC", "SimSun"]
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


def find_raw_files() -> tuple[Path, Path | None]:
    xlsx = next(p for p in DATA_DIR.glob("*.xlsx") if not p.name.startswith("~$"))
    docx = next(DATA_DIR.glob("*.docx"), None)
    return xlsx, docx


def load_raw() -> tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    xlsx, docx = find_raw_files()
    df = pd.read_excel(xlsx)
    df = normalize_missing(df)
    mappings = parse_docx_mappings(docx, df.columns.tolist())
    return df, mappings


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clip01(s: pd.Series) -> pd.Series:
    return s.clip(0.0, 1.0)


def weighted_mean(values: List[pd.Series], weights: List[float]) -> pd.Series:
    frame = pd.concat(values, axis=1)
    weight_arr = np.array(weights, dtype=float)
    valid = frame.notna().values.astype(float)
    weighted = frame.fillna(0.0).values * weight_arr.reshape(1, -1)
    denom = (valid * weight_arr.reshape(1, -1)).sum(axis=1)
    out = weighted.sum(axis=1) / np.where(denom == 0, np.nan, denom)
    return pd.Series(out, index=frame.index)


def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    if s.notna().sum() == 0:
        return s
    q1, q2 = s.quantile(lower), s.quantile(upper)
    return s.clip(q1, q2)


def compute_continuous_ic(df: pd.DataFrame) -> pd.DataFrame:
    # Cognition: MMSE-like continuous score
    cog_total = safe_numeric(df["认知-总分"])
    cog_score = clip01(cog_total / 30.0)

    # Psychological: GDS-15-like total, lower is better; diagnosis adds penalty
    psych_total = safe_numeric(df["心理-总分"])
    psych_score = clip01(1.0 - psych_total / 15.0)
    if "是否焦虑抑郁症" in df.columns:
        psych_dx = safe_numeric(df["是否焦虑抑郁症"])
        psych_score = clip01(weighted_mean([psych_score, 1.0 - psych_dx], [0.85, 0.15]))

    # Vitality: MNA-SF-like total + nutrition state + BMI + calf circumference
    mna_total = safe_numeric(df["活力-总分"])
    mna_score = clip01(mna_total / 14.0)
    nutrition_desc = safe_numeric(df["活力-营养描述结果"])
    nutrition_score = clip01(nutrition_desc / 2.0)
    bmi = safe_numeric(df["活力-BMI值"])
    bmi_score = clip01(1.0 - (bmi - 22.0).abs() / 10.0)
    calf = safe_numeric(df["小腿围"])
    calf_score = clip01((calf - 31.0) / 8.0)
    vitality_score = clip01(weighted_mean([mna_score, nutrition_score, bmi_score, calf_score], [0.55, 0.10, 0.20, 0.15]))

    # Locomotion: SPPB-like total + completion + gait abnormality + gait speed + grip
    sppb = safe_numeric(df["运动-总分"])
    sppb_score = clip01(sppb / 12.0)
    motion_completion = safe_numeric(df["运动是否完成编码"])
    completion_score = clip01(motion_completion / 2.0)
    gait_abn = safe_numeric(df["步态异常-编码"])
    gait_score = clip01(1.0 - gait_abn)

    gait_time = safe_numeric(df["运动-4米行走时间：日常行走速度走完4米-平均完成时间"])
    gait_speed = 4.0 / gait_time.replace(0, np.nan)
    gait_speed_score = clip01((gait_speed - 0.4) / 0.8)

    left_grip = safe_numeric(df["Fried衰弱表型评估-握力左手最大值"])
    right_grip = safe_numeric(df["Fried衰弱表型评估-握力右手最大值"])
    grip = pd.concat([left_grip, right_grip], axis=1).max(axis=1, skipna=True)
    sex = safe_numeric(df["性别"])
    male = sex == 1
    grip_threshold = pd.Series(np.where(male, 28.0, 18.0), index=df.index)
    grip_upper = pd.Series(np.where(male, 40.0, 28.0), index=df.index)
    grip_score = clip01((grip - grip_threshold) / (grip_upper - grip_threshold))

    locomotion_score = clip01(weighted_mean([sppb_score, completion_score, gait_score, gait_speed_score, grip_score], [0.45, 0.10, 0.15, 0.15, 0.15]))

    # Sensory: impairment + screening + daily life impact
    hearing_imp = safe_numeric(df["听力障碍"])
    hearing_screen = safe_numeric(df["感知-听力"])
    hearing_impact = safe_numeric(df["听力障碍是否影响日常"])
    hearing_ok = 1.0 - hearing_imp
    hearing_impact_score = pd.Series(np.where(hearing_imp == 1, 1.0 - hearing_impact, 1.0), index=df.index)
    hearing_score = clip01(weighted_mean([hearing_ok, hearing_screen, hearing_impact_score], [0.45, 0.35, 0.20]))

    vision_imp = safe_numeric(df["视力障碍"])
    vision_screen = safe_numeric(df["感知-视力"])
    vision_impact = safe_numeric(df["视力障碍是否影响日常"])
    vision_ok = 1.0 - vision_imp
    vision_impact_score = pd.Series(np.where(vision_imp == 1, 1.0 - vision_impact, 1.0), index=df.index)
    vision_score = clip01(weighted_mean([vision_ok, vision_screen, vision_impact_score], [0.45, 0.35, 0.20]))
    sensory_score = clip01(weighted_mean([hearing_score, vision_score], [0.5, 0.5]))

    out = pd.DataFrame(
        {
            "IC_cognition_cont": cog_score,
            "IC_psychological_cont": psych_score,
            "IC_vitality_cont": vitality_score,
            "IC_locomotion_cont": locomotion_score,
            "IC_sensory_cont": sensory_score,
        }
    )
    for col in out.columns:
        out[col] = out[col].fillna(out[col].median())
        out[f"{col}_100"] = out[col] * 100.0
    out["IC_total_cont"] = out[[c for c in out.columns if c.endswith("_cont")]].mean(axis=1)
    out["IC_total_cont_100"] = out["IC_total_cont"] * 100.0
    return out


def save_metadata(df: pd.DataFrame, ic_df: pd.DataFrame) -> None:
    present_core = [c for c in CORE_FEATURES if c in df.columns]
    present_external = [c for c in EXTERNAL_FEATURES if c in df.columns]
    meta = {
        "n_samples": int(len(df)),
        "n_raw_features": int(df.shape[1]),
        "core_features_present": present_core,
        "external_features_present": present_external,
        "n_core_features": len(present_core),
        "n_external_features": len(present_external),
        "continuous_ic_columns": ic_df.columns.tolist(),
    }
    (CONFIG_DIR / "feature_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def readable_value(col: str, value, mappings: Dict[str, Dict[str, str]]) -> str:
    if pd.isna(value):
        return "missing"
    mapping = mappings.get(col)
    if mapping:
        key = str(value).strip()
        if key in mapping:
            return str(mapping[key])
    try:
        fv = float(value)
        if fv.is_integer():
            return str(int(fv))
        return f"{fv:.2f}"
    except Exception:
        return str(value)


def build_medical_text(df: pd.DataFrame, mappings: Dict[str, Dict[str, str]], ic_df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for idx, row in df.iterrows():
        sid = row["编号"] if "编号" in df.columns else idx
        parts = [f"Sample {sid}."]
        for section, cols in TEXT_SECTIONS.items():
            items = []
            for col in cols:
                if col not in df.columns:
                    continue
                val = readable_value(col, row[col], mappings)
                if val == "missing":
                    continue
                items.append(f"{col}={val}")
            if items:
                parts.append(f"{section}: {'; '.join(items)}.")
        s = ic_df.loc[idx]
        parts.append(
            "Continuous IC: "
            f"cognition={s['IC_cognition_cont_100']:.1f}; "
            f"psychological={s['IC_psychological_cont_100']:.1f}; "
            f"vitality={s['IC_vitality_cont_100']:.1f}; "
            f"locomotion={s['IC_locomotion_cont_100']:.1f}; "
            f"sensory={s['IC_sensory_cont_100']:.1f}; "
            f"overall={s['IC_total_cont_100']:.1f}."
        )
        out_rows.append({"编号": sid, "medical_text": " ".join(parts)})
    out = pd.DataFrame(out_rows)
    out.to_csv(OUTPUT_DIR / "medical_text_cards.csv", index=False, encoding="utf-8-sig")
    return out


def encode_text_with_bert(texts: Iterable[str]) -> Tuple[np.ndarray, str]:
    cache_path = OUTPUT_DIR / "bert_raw_embedding.npy"
    meta_path = OUTPUT_DIR / "bert_encoder_name.txt"
    if cache_path.exists() and meta_path.exists():
        return np.load(cache_path), meta_path.read_text(encoding="utf-8").strip()

    text_list = list(texts)
    try:
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, local_files_only=False)
        model = AutoModel.from_pretrained(BERT_MODEL, local_files_only=False)
        model.eval()
        device = torch.device("cpu")
        model.to(device)
        outputs = []
        with torch.inference_mode():
            for i in range(0, len(text_list), 48):
                batch = text_list[i:i + 48]
                encoded = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
                encoded = {k: v.to(device) for k, v in encoded.items()}
                hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                outputs.append(pooled.cpu().numpy())
        emb = np.vstack(outputs)
        np.save(cache_path, emb)
        meta_path.write_text(BERT_MODEL, encoding="utf-8")
        return emb, BERT_MODEL
    except Exception:
        vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2), min_df=5)
        tfidf = vectorizer.fit_transform(text_list)
        emb = PCA(n_components=min(512, tfidf.shape[1] - 1), random_state=RANDOM_SEED).fit_transform(tfidf.toarray())
        np.save(cache_path, emb)
        meta_path.write_text("TFIDF+PCA_fallback", encoding="utf-8")
        return emb, "TFIDF+PCA_fallback"


def preprocess_core_features(df: pd.DataFrame, mappings: Dict[str, Dict[str, str]], ic_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    core_cols = [c for c in CORE_FEATURES if c in df.columns]
    core_df = df[core_cols].copy()

    for col in ["患有慢性病数量", "服用药物数量", "过去一年住院次数", "过去一年急诊次数", "查尔森合并症得分"]:
        if col in core_df.columns:
            core_df[col] = winsorize_series(safe_numeric(core_df[col]))

    for col in ["IC_cognition_cont_100", "IC_psychological_cont_100", "IC_vitality_cont_100", "IC_locomotion_cont_100", "IC_sensory_cont_100"]:
        core_df[col] = ic_df[col]

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in core_df.columns:
        if col.startswith("IC_"):
            numeric_cols.append(col)
            continue
        num = pd.to_numeric(core_df[col], errors="coerce")
        num_ratio = num.notna().mean()
        if num_ratio > 0.9 and num.nunique(dropna=True) > 10:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    for col in list(numeric_cols):
        if core_df[col].isna().mean() >= 0.05:
            core_df[f"{col}__missing"] = core_df[col].isna().astype(int)
            categorical_cols.append(f"{col}__missing")

    for col in categorical_cols:
        if col in mappings:
            core_df[col] = apply_mapping(core_df[col], mappings[col])
        core_df[col] = core_df[col].fillna("Missing").astype(str)
        freq = core_df[col].value_counts(normalize=True)
        rare = freq[freq < 0.01].index
        core_df.loc[core_df[col].isin(rare), col] = "Other"

    for col in numeric_cols:
        core_df[col] = pd.to_numeric(core_df[col], errors="coerce")
        core_df[col] = core_df[col].fillna(core_df[col].median())

    core_df.to_csv(OUTPUT_DIR / "core_feature_table.csv", index=False, encoding="utf-8-sig")
    return core_df, numeric_cols, categorical_cols


def build_pca_input(core_df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> np.ndarray:
    cache_path = OUTPUT_DIR / "pca_embedding_128.npy"
    if cache_path.exists():
        return np.load(cache_path)

    num_df = core_df[numeric_cols].copy()
    scaler = StandardScaler()
    num_scaled = pd.DataFrame(scaler.fit_transform(num_df), columns=numeric_cols, index=core_df.index)
    cat_df = pd.get_dummies(core_df[categorical_cols].astype(str), prefix=categorical_cols, prefix_sep="=")
    X = pd.concat([num_scaled, cat_df], axis=1).astype(np.float32)
    X.to_parquet(OUTPUT_DIR / "core_design_matrix.parquet", index=False)
    emb = PCA(n_components=EMBED_DIM, random_state=RANDOM_SEED).fit_transform(X.values)
    np.save(cache_path, emb)
    return emb


class FTTransformerEncoder(nn.Module):
    def __init__(self, n_num: int, cat_cards: List[int], d_token: int = EMBED_DIM, n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.num_weight = nn.Parameter(torch.randn(n_num, d_token) * 0.02)
        self.num_bias = nn.Parameter(torch.zeros(n_num, d_token))
        self.cat_embeddings = nn.ModuleList([nn.Embedding(card, d_token) for card in cat_cards])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads, batch_first=True, dim_feedforward=d_token * 2, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.num_head = nn.Linear(d_token, n_num)
        self.cat_heads = nn.ModuleList([nn.Linear(d_token, card) for card in cat_cards])

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        num_tokens = x_num.unsqueeze(-1) * self.num_weight + self.num_bias
        pieces = [num_tokens]
        if len(self.cat_embeddings) > 0:
            cat_tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
            pieces.append(torch.stack(cat_tokens, dim=1))
        tokens = torch.cat(pieces, dim=1)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        z = self.transformer(tokens)
        cls_out = z[:, 0, :]
        num_pred = self.num_head(cls_out)
        cat_pred = [head(cls_out) for head in self.cat_heads]
        return cls_out, num_pred, cat_pred


def build_ft_input(core_df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    num_path = OUTPUT_DIR / "ft_num_scaled.npy"
    cat_path = OUTPUT_DIR / "ft_cat_codes.npy"
    cards_path = CONFIG_DIR / "ft_cat_cards.json"
    if num_path.exists() and cat_path.exists() and cards_path.exists():
        return np.load(num_path), np.load(cat_path), json.loads(cards_path.read_text(encoding="utf-8"))

    num_df = core_df[numeric_cols].copy()
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num_df).astype(np.float32)

    cat_codes = []
    cat_cards = []
    for col in categorical_cols:
        cat = pd.Categorical(core_df[col].astype(str))
        cat_codes.append(cat.codes)
        cat_cards.append(len(cat.categories))
    cat_array = np.vstack(cat_codes).T.astype(np.int64) if cat_codes else np.empty((len(core_df), 0), dtype=np.int64)

    np.save(num_path, num_scaled)
    np.save(cat_path, cat_array)
    cards_path.write_text(json.dumps(cat_cards, ensure_ascii=False), encoding="utf-8")
    return num_scaled, cat_array, cat_cards


def train_ft_transformer(num_scaled: np.ndarray, cat_array: np.ndarray, cat_cards: List[int]) -> np.ndarray:
    cache_path = OUTPUT_DIR / "ft_embedding_128.npy"
    if cache_path.exists():
        return np.load(cache_path)

    device = torch.device("cpu")
    model = FTTransformerEncoder(num_scaled.shape[1], cat_cards, d_token=EMBED_DIM, n_heads=8, n_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    dataset = TensorDataset(torch.tensor(num_scaled, dtype=torch.float32), torch.tensor(cat_array, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model.train()
    for _ in range(12):
        for xb_num, xb_cat in loader:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            optimizer.zero_grad()
            _, num_pred, cat_pred = model(xb_num, xb_cat)
            loss_num = mse(num_pred, xb_num)
            loss_cat = torch.tensor(0.0, device=device)
            if len(cat_pred) > 0:
                loss_cat = sum(F.cross_entropy(logits, xb_cat[:, i]) for i, logits in enumerate(cat_pred)) / len(cat_pred)
            loss = loss_num + loss_cat
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.inference_mode():
        emb, _, _ = model(torch.tensor(num_scaled, dtype=torch.float32).to(device), torch.tensor(cat_array, dtype=torch.long).to(device))
    emb = emb.cpu().numpy()
    np.save(cache_path, emb)
    return emb


def external_effect_score(labels: np.ndarray, df: pd.DataFrame, cols: List[str]) -> float:
    effect_scores = []
    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        mask = vals.notna()
        if mask.sum() < 30:
            continue
        y = vals[mask].values
        lab = labels[mask.values]
        grand = y.mean()
        ss_total = ((y - grand) ** 2).sum()
        if ss_total == 0:
            continue
        ss_between = sum(len(y[lab == c]) * (y[lab == c].mean() - grand) ** 2 for c in np.unique(lab))
        effect_scores.append(ss_between / ss_total)
    return float(np.mean(effect_scores)) if effect_scores else 0.0


def evaluate_method(method: str, emb: np.ndarray, external_df: pd.DataFrame, ic_df: pd.DataFrame) -> List[ClusterEval]:
    results = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
        labels = km.fit_predict(emb)
        sil = silhouette_score(emb, labels)
        ch = calinski_harabasz_score(emb, labels)
        db = davies_bouldin_score(emb, labels)
        aris = []
        for seed in [0, 1, 2, 3, 4]:
            km2 = KMeans(n_clusters=k, random_state=seed, n_init=20)
            labels2 = km2.fit_predict(emb)
            aris.append(adjusted_rand_score(labels, labels2))
        stability = float(np.mean(aris))
        ext_score = external_effect_score(labels, external_df, [c for c in EXTERNAL_FEATURES if c in external_df.columns])
        corr = float(spearmanr(pd.Series(labels), ic_df["IC_total_cont"], nan_policy="omit").correlation)
        results.append(ClusterEval(method, k, labels, float(sil), float(ch), float(db), stability, ext_score, corr))
    return results


def choose_best(results: List[ClusterEval]) -> ClusterEval:
    df = pd.DataFrame(
        [
            {
                "method": r.method,
                "k": r.k,
                "silhouette": r.silhouette,
                "stability_ari": r.stability_ari,
                "external_separation": r.external_separation,
                "db_score": 1.0 / (1.0 + r.db),
            }
            for r in results
        ]
    )
    for col in ["silhouette", "stability_ari", "external_separation", "db_score"]:
        if df[col].max() > df[col].min():
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        else:
            df[col] = 0.0
    df["combined"] = df[["silhouette", "stability_ari", "external_separation", "db_score"]].mean(axis=1)
    best = df.sort_values("combined", ascending=False).iloc[0]
    return next(r for r in results if r.method == best["method"] and r.k == best["k"])


def build_cluster_profiles(best: ClusterEval, emb: np.ndarray, raw_df: pd.DataFrame, ic_df: pd.DataFrame) -> None:
    method = best.method
    labels = best.labels
    pd.DataFrame({"编号": raw_df["编号"], "cluster_id": labels}).to_csv(OUTPUT_DIR / f"{method}_cluster_assignments.csv", index=False, encoding="utf-8-sig")

    ic_profile = ic_df.copy()
    ic_profile["cluster_id"] = labels
    ic_mean = ic_profile.groupby("cluster_id")[["IC_cognition_cont_100", "IC_psychological_cont_100", "IC_vitality_cont_100", "IC_locomotion_cont_100", "IC_sensory_cont_100", "IC_total_cont_100"]].mean()
    ic_mean.to_csv(OUTPUT_DIR / f"{method}_ic_profile.csv", encoding="utf-8-sig")

    ext_cols = [c for c in EXTERNAL_FEATURES if c in raw_df.columns]
    ext = raw_df[ext_cols].apply(pd.to_numeric, errors="coerce")
    ext["cluster_id"] = labels
    ext_mean = ext.groupby("cluster_id").mean()
    ext_mean.to_csv(OUTPUT_DIR / f"{method}_external_profile.csv", encoding="utf-8-sig")

    stats_rows = []
    for col in ext_cols:
        vals = pd.to_numeric(raw_df[col], errors="coerce")
        groups = [vals[labels == c].dropna().values for c in np.unique(labels)]
        if sum(len(g) > 0 for g in groups) < 2:
            continue
        try:
            _, p = kruskal(*groups)
        except Exception:
            p = 1.0
        means = [g.mean() if len(g) else np.nan for g in groups]
        stats_rows.append({"feature": col, "p_value": float(p), "max_mean_diff": float(np.nanmax(means) - np.nanmin(means))})
    pd.DataFrame(stats_rows).sort_values(["p_value", "max_mean_diff"]).to_csv(OUTPUT_DIR / f"{method}_external_stats.csv", index=False, encoding="utf-8-sig")

    umap = UMAP(n_components=2, random_state=RANDOM_SEED)
    coords = umap.fit_transform(emb)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, palette="tab10", s=12, ax=ax, legend="full")
    ax.set_title(f"UMAP of {method} Embedding")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"{method}_umap.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    sns.heatmap(ic_mean, annot=True, cmap="YlGnBu", fmt=".1f", ax=ax)
    ax.set_title(f"Continuous IC Profile by Cluster ({method})")
    ax.set_xlabel("IC Domain")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"{method}_ic_heatmap.png", dpi=220)
    plt.close(fig)

    radar_cols = ["IC_cognition_cont_100", "IC_psychological_cont_100", "IC_vitality_cont_100", "IC_locomotion_cont_100", "IC_sensory_cont_100"]
    angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    for cluster_id, row in ic_mean[radar_cols].iterrows():
        values = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, values, linewidth=2, label=f"Cluster {cluster_id}")
        ax.fill(angles, values, alpha=0.10)
    ax.set_thetagrids(np.degrees(angles[:-1]), ["Cognition", "Psychological", "Vitality", "Locomotion", "Sensory"])
    ax.set_title(f"Radar Plot of IC Domains ({method})")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15))
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"{method}_radar.png", dpi=220)
    plt.close(fig)

    top_ext = [c for c in ext_cols if c in ["ADL量表-总分", "IADL量表-总分", "肌少症评估-总分", "Fried衰弱表型评估-总分", "衰弱快速筛查量表-总分", "跌倒评估-总分", "弹性评估-总分", "生活行为与社会功能评估-总分"]]
    if top_ext:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        sns.heatmap(ext_mean[top_ext], annot=True, cmap="OrRd", fmt=".1f", ax=ax)
        ax.set_title(f"External Clinical Profile by Cluster ({method})")
        ax.set_xlabel("External Variable")
        ax.set_ylabel("Cluster")
        plt.tight_layout()
        fig.savefig(FIG_DIR / f"{method}_external_heatmap.png", dpi=220)
        plt.close(fig)


def plot_method_comparison(best_results: List[ClusterEval]) -> None:
    df = pd.DataFrame(
        [
            {
                "method": r.method,
                "k": r.k,
                "silhouette": r.silhouette,
                "stability_ari": r.stability_ari,
                "external_separation": r.external_separation,
                "davies_bouldin": r.db,
            }
            for r in best_results
        ]
    )
    df.to_csv(OUTPUT_DIR / "best_method_metrics.csv", index=False, encoding="utf-8-sig")
    plot_df = df.melt(id_vars=["method", "k"], value_vars=["silhouette", "stability_ari", "external_separation"])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    sns.barplot(data=plot_df, x="variable", y="value", hue="method", ax=ax)
    ax.set_title("Method Comparison of Subtyping Quality")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "method_comparison.png", dpi=220)
    plt.close(fig)


def write_report(raw_df: pd.DataFrame, core_df: pd.DataFrame, ic_df: pd.DataFrame, text_encoder_name: str, all_results: List[ClusterEval], best_results: List[ClusterEval]) -> None:
    all_df = pd.DataFrame(
        [
            {
                "method": r.method,
                "k": r.k,
                "silhouette": r.silhouette,
                "calinski_harabasz": r.calinski,
                "davies_bouldin": r.db,
                "stability_ari": r.stability_ari,
                "external_separation": r.external_separation,
                "cluster_label_ic_spearman": r.ic_cont_corr,
            }
            for r in all_results
        ]
    )
    all_df.to_csv(OUTPUT_DIR / "all_cluster_metrics.csv", index=False, encoding="utf-8-sig")

    best_df = pd.DataFrame(
        [
            {
                "method": r.method,
                "k": r.k,
                "silhouette": r.silhouette,
                "stability_ari": r.stability_ari,
                "external_separation": r.external_separation,
                "davies_bouldin": r.db,
            }
            for r in best_results
        ]
    )
    winner = best_df.sort_values(["external_separation", "silhouette", "stability_ari"], ascending=False).iloc[0]

    lines = []
    lines.append("# 正式分型实验技术报告")
    lines.append("")
    lines.append("## 1. 研究目标")
    lines.append("- 在现有横断面数据上建立正式版分型流程，不再依赖IC二值化。")
    lines.append("- 核心比较三条路线：BERT文本语义表征、FTTransformer表格表征、PCA同维度基线。")
    lines.append("- 所有方法统一embedding维度为128，并在k=3~6内比较聚类质量。")
    lines.append("")
    lines.append("## 2. 原始数据与特征工程")
    lines.append(f"- 样本数：{len(raw_df)}")
    lines.append(f"- 原始字段数：{raw_df.shape[1]}")
    lines.append(f"- 用于正式分型的核心字段数：{core_df.shape[1]}")
    lines.append("- 缺失值规则：ND/UK/NA等显式缺失统一置空；0不视为缺失。")
    lines.append("- 数值特征：中位数插补；对慢病数量、药物数量、住院次数、急诊次数、Charlson得分做1%-99% winsorize。")
    lines.append("- 类别特征：编码.docx语义映射；缺失为Missing；低频类别(<1%)并入Other。")
    lines.append("- 对缺失率>=5%的数值列新增missing indicator。")
    lines.append("")
    lines.append("## 3. 连续化IC评分设计")
    lines.append("- Cognition：认知-总分/30，近似MMSE连续得分。")
    lines.append("- Psychological：1-心理-总分/15，再与焦虑抑郁诊断做15%惩罚整合。")
    lines.append("- Vitality：MNA-SF总分、营养描述、BMI、腓肠围加权平均。")
    lines.append("- Locomotion：SPPB总分、完成编码、步态异常、4m gait speed、握力加权平均。")
    lines.append("- Sensory：听力/视力障碍、筛查结果、是否影响日常做双通道加权平均。")
    lines.append("- 所有域分数归一到0~100，总分为5域均值。")
    lines.append("")
    lines.append("## 4. 表征学习路线")
    lines.append(f"- 文本路线：将核心字段通过编码.docx做语义转化，拼接成医学化Tab2Text，再用 `{text_encoder_name}` 编码，随后PCA投影到128维。")
    lines.append("- 表格路线：FTTransformer风格编码器，自监督重建数值与类别，输出128维CLS embedding。")
    lines.append("- 基线路线：对同一批核心字段做统一预处理后，用PCA直接降到128维。")
    lines.append("")
    lines.append("## 5. 聚类评价指标")
    lines.append("- Internal：silhouette, Calinski-Harabasz, Davies-Bouldin。")
    lines.append("- Stability：不同随机种子的KMeans ARI。")
    lines.append("- External separation：在未参与聚类的ADL/IADL/肌少症/Fried/衰弱/跌倒/弹性/社会功能等变量上的平均方差解释度。")
    lines.append("")
    lines.append("## 6. 最优结果总表")
    lines.append(best_df.to_markdown(index=False))
    lines.append("")
    lines.append("## 7. 当前综合最优结果")
    lines.append(f"- 当前综合最优方法：`{winner['method']}`")
    lines.append(f"- 最优聚类数：`k={int(winner['k'])}`")
    lines.append(f"- silhouette：`{winner['silhouette']:.4f}`")
    lines.append(f"- stability_ari：`{winner['stability_ari']:.4f}`")
    lines.append(f"- external_separation：`{winner['external_separation']:.4f}`")
    lines.append("")
    lines.append("## 8. 输出文件")
    lines.append("- `outputs/formal_subtyping/IC_continuous_scores.csv`")
    lines.append("- `outputs/formal_subtyping/best_method_metrics.csv`")
    lines.append("- `outputs/formal_subtyping/*_cluster_assignments.csv`")
    lines.append("- `outputs/formal_subtyping/*_ic_profile.csv`")
    lines.append("- `outputs/formal_subtyping/*_external_profile.csv`")
    lines.append("- `figures/formal_subtyping/*.png`")
    (REPORT_DIR / "TECHNICAL_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline() -> None:
    ensure_dirs()
    configure_plot_style()
    seed_everything()

    raw_df, mappings = load_raw()
    ic_df = compute_continuous_ic(raw_df)
    ic_df.to_csv(OUTPUT_DIR / "IC_continuous_scores.csv", index=False, encoding="utf-8-sig")
    save_metadata(raw_df, ic_df)

    text_df = build_medical_text(raw_df, mappings, ic_df)
    bert_raw, text_encoder_name = encode_text_with_bert(text_df["medical_text"].tolist())
    bert_emb = PCA(n_components=EMBED_DIM, random_state=RANDOM_SEED).fit_transform(bert_raw) if bert_raw.shape[1] != EMBED_DIM else bert_raw
    np.save(OUTPUT_DIR / "bert_embedding_128.npy", bert_emb)

    core_df, numeric_cols, categorical_cols = preprocess_core_features(raw_df, mappings, ic_df)
    pca_emb = build_pca_input(core_df, numeric_cols, categorical_cols)
    num_scaled, cat_array, cat_cards = build_ft_input(core_df, numeric_cols, categorical_cols)
    ft_emb = train_ft_transformer(num_scaled, cat_array, cat_cards)

    external_df = raw_df[[c for c in EXTERNAL_FEATURES if c in raw_df.columns]].copy()

    all_results: List[ClusterEval] = []
    best_results: List[ClusterEval] = []
    for method, emb in [("BERTText", bert_emb), ("FTTransformer", ft_emb), ("PCA", pca_emb)]:
        res = evaluate_method(method, emb, external_df, ic_df)
        all_results.extend(res)
        best = choose_best(res)
        best_results.append(best)
        build_cluster_profiles(best, emb, raw_df, ic_df)

    plot_method_comparison(best_results)
    write_report(raw_df, core_df, ic_df, text_encoder_name, all_results, best_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run the full formal subtyping pipeline")
    args = parser.parse_args()
    if args.run:
        run_pipeline()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
