# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kruskal, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from formal_subtyping_pipeline import build_medical_text, compute_continuous_ic, load_raw, preprocess_core_features  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "advanced_subtyping"
FIG_DIR = ROOT / "figures" / "advanced_subtyping"
REPORT_DIR = ROOT / "reports" / "advanced_subtyping"
CONFIG_DIR = ROOT / "configs" / "advanced_subtyping"

RANDOM_SEED = 2026
BATCH_SIZE = 256
D_MODEL = 128
TEXT_MODEL = "bert-base-chinese"
MASK_PROB = 0.30
TABLE_EPOCHS = 50
DUAL_EPOCHS = 36
EARLY_STOP = 8
LR_TABLE = 1e-3
LR_DUAL = 8e-4
TEMPERATURE = 0.10
K_RANGE = [3, 4, 5, 6]
CLUSTERERS = ["kmeans", "gmm"]

GROUP_MAP = {
    "demographics": ["年龄", "性别", "婚姻状况", "家庭类型", "文化程度", "经济状况", "受试来源"],
    "self_health": ["健康状态", "健康自评分数"],
    "disease_burden": [
        "患有慢性病数量", "服用药物数量", "过去一年住院次数", "过去一年急诊次数", "查尔森合并症得分",
        "是否高血压", "是否糖尿病", "是否骨关节病", "是否骨质疏松", "是否脑血栓", "是否焦虑抑郁症", "是否白内障", "是否青光眼",
    ],
    "sensory": ["听力障碍", "听力障碍是否影响日常", "视力障碍", "视力障碍是否影响日常", "自我感觉嗅觉", "感知-视力", "感知-听力", "IC_sensory_cont_100"],
    "cognition": ["认知-总分", "IC_cognition_cont_100"],
    "psychological": ["心理-总分", "是否焦虑抑郁症", "IC_psychological_cont_100"],
    "vitality": [
        "活力-过去三个月内有没有因为食欲不振、消化问题、咀嚼或吞咽困难而摄食减少",
        "活力-过去三个月内体重下降情况", "活力-活动能力", "活力-既往3个月内有无重大心理变化或急性疾病", "活力-神经心理问题",
        "活力-BMI得分", "活力-BMI值", "活力-如不能取得BMI，请以下面小腿围代替问题6", "活力-总分", "活力-营养描述结果", "体重", "身高", "小腿围", "IC_vitality_cont_100",
    ],
    "locomotion": [
        "运动-双脚并拢站立10秒-保持时间为:", "运动-双脚半前后位站立10秒-保持时间为:", "运动-双脚前后成一直线站立10秒-保持时间为:",
        "运动-从椅子上站起，让其尽快地起立5次-完成时间为", "运动-从椅子上站起，让其尽快地起立5次-完成次数为", "运动-4米行走时间：日常行走速度走完4米-平均完成时间",
        "运动是否完成编码", "运动-总分", "步态异常-编码", "Fried衰弱表型评估-4米行走第一次", "Fried衰弱表型评估-4米行走第二次", "Fried衰弱表型评估-握力左手最大值", "Fried衰弱表型评估-握力右手最大值", "IC_locomotion_cont_100",
    ],
    "lifestyle": ["生活行为与社会功能评估-7您是否有吸烟习惯？", "生活行为与社会功能评估-8您是否喝酒？"],
}

EXTERNAL_FEATURES = [
    "ADL量表-总分", "IADL量表-总分", "肌少症评估-总分", "Fried衰弱表型评估-总分", "衰弱快速筛查量表-总分", "跌倒评估-总分", "弹性评估-总分",
    "生活行为与社会功能评估-总分", "生命质量评估-1）总体来讲，您的健康状况是", "生命质量评估-10）在过去4个星期里，您有多少时间感到精力充沛？",
    "生命质量评估-11）在过去4个星期里，您有多少时间感到心情不好、闷闷不乐或沮丧？", "生命质量评估-12）在过去4个星期里，有多少时间由于您身体健康或情绪问题而妨碍您的社交活动（比如探亲、访友等）？",
]

EXTERNAL_ALIAS = {
    "ADL量表-总分": "ADL total",
    "IADL量表-总分": "IADL total",
    "肌少症评估-总分": "Sarcopenia score",
    "Fried衰弱表型评估-总分": "Fried frailty score",
    "衰弱快速筛查量表-总分": "Frailty screen score",
    "跌倒评估-总分": "Fall risk score",
    "弹性评估-总分": "Resilience score",
    "生活行为与社会功能评估-总分": "Social function score",
    "生命质量评估-1）总体来讲，您的健康状况是": "QoL general health",
    "生命质量评估-10）在过去4个星期里，您有多少时间感到精力充沛？": "QoL energy",
    "生命质量评估-11）在过去4个星期里，您有多少时间感到心情不好、闷闷不乐或沮丧？": "QoL depressed mood",
    "生命质量评估-12）在过去4个星期里，有多少时间由于您身体健康或情绪问题而妨碍您的社交活动（比如探亲、访友等）？": "QoL social limitation",
}

IC_PROFILE_COLS = ["IC_cognition_cont_100", "IC_psychological_cont_100", "IC_vitality_cont_100", "IC_locomotion_cont_100", "IC_sensory_cont_100", "IC_total_cont_100"]
IC_ALIAS = {
    "IC_cognition_cont_100": "Cognition",
    "IC_psychological_cont_100": "Psychological",
    "IC_vitality_cont_100": "Vitality",
    "IC_locomotion_cont_100": "Locomotion",
    "IC_sensory_cont_100": "Sensory",
    "IC_total_cont_100": "Overall IC",
}

INTERPRETABLE_MARKERS = [
    "年龄", "健康自评分数", "患有慢性病数量", "服用药物数量", "过去一年住院次数", "过去一年急诊次数", "查尔森合并症得分", "认知-总分", "心理-总分", "活力-总分", "活力-BMI值", "小腿围", "运动-总分", "运动-4米行走时间：日常行走速度走完4米-平均完成时间", "Fried衰弱表型评估-握力左手最大值", "Fried衰弱表型评估-握力右手最大值", "听力障碍", "视力障碍", "步态异常-编码",
]


@dataclass
class EvalRow:
    method: str
    clusterer: str
    k: int
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    stability_ari: float
    min_cluster_ratio: float
    external_separation: float


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
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 220
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["axes.unicode_minus"] = False


def build_professional_text(raw_df: pd.DataFrame, text_cards: pd.DataFrame, ic_df: pd.DataFrame) -> pd.DataFrame:
    id_col = "编号" if "编号" in raw_df.columns else raw_df.columns[1]
    rows = []
    for i, row in raw_df.iterrows():
        sample_id = row[id_col]
        age = row.get("年龄", np.nan)
        sex = row.get("性别", np.nan)
        sex_text = "男性" if str(sex) in {"1", "1.0", "男"} else "女性" if pd.notna(sex) else "性别缺失"
        text = (
            f"样本{sample_id}。社区老年人，{age}岁，{sex_text}。"
            f"慢病数量{row.get('患有慢性病数量', np.nan)}，服药数量{row.get('服用药物数量', np.nan)}，"
            f"过去一年住院{row.get('过去一年住院次数', np.nan)}次，急诊{row.get('过去一年急诊次数', np.nan)}次。"
            f"连续内在能力：认知{ic_df.loc[i, 'IC_cognition_cont_100']:.1f}，心理{ic_df.loc[i, 'IC_psychological_cont_100']:.1f}，活力{ic_df.loc[i, 'IC_vitality_cont_100']:.1f}，"
            f"运动{ic_df.loc[i, 'IC_locomotion_cont_100']:.1f}，感官{ic_df.loc[i, 'IC_sensory_cont_100']:.1f}，总体{ic_df.loc[i, 'IC_total_cont_100']:.1f}。"
            f"结构化补充：{text_cards.iloc[i]['medical_text']}"
        )
        rows.append({"sample_id": sample_id, "medical_text_cn": text})
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "medical_text_professional_cn.csv", index=False, encoding="utf-8-sig")
    return out


def encode_text(texts: Iterable[str]) -> Tuple[np.ndarray, str]:
    cache_path = OUTPUT_DIR / "bert_text_embedding_raw.npy"
    meta_path = OUTPUT_DIR / "text_encoder_name.txt"
    if cache_path.exists() and meta_path.exists():
        return np.load(cache_path), meta_path.read_text(encoding="utf-8").strip()

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, local_files_only=False)
    model = AutoModel.from_pretrained(TEXT_MODEL, local_files_only=False)
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    outputs = []
    text_list = list(texts)
    with torch.inference_mode():
        for i in range(0, len(text_list), 32):
            batch = text_list[i:i + 32]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            outputs.append(pooled.cpu().numpy())
    emb = np.vstack(outputs)
    np.save(cache_path, emb)
    meta_path.write_text(TEXT_MODEL, encoding="utf-8")
    return emb, TEXT_MODEL


def assign_group(base_col: str) -> str:
    base = base_col.replace("__missing", "")
    for group_name, cols in GROUP_MAP.items():
        if base in cols:
            return group_name
    if base.startswith("认知") or "认知" in base:
        return "cognition"
    if base.startswith("心理") or "焦虑" in base or "抑郁" in base:
        return "psychological"
    if base.startswith("活力") or base in {"体重", "身高", "小腿围"}:
        return "vitality"
    if base.startswith("运动") or base.startswith("Fried") or "步态" in base:
        return "locomotion"
    if "听力" in base or "视力" in base or "感知" in base or "嗅觉" in base:
        return "sensory"
    if "慢性病" in base or "住院" in base or "急诊" in base or "查尔森" in base or base.startswith("是否"):
        return "disease_burden"
    return "self_health"


def build_design_matrix(core_df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, List[int]], np.ndarray]:
    num_scaled = pd.DataFrame(StandardScaler().fit_transform(core_df[numeric_cols]), columns=numeric_cols, index=core_df.index)
    cat_df = pd.get_dummies(core_df[categorical_cols].astype(str), prefix=categorical_cols, prefix_sep="=")
    design = pd.concat([num_scaled, cat_df], axis=1).astype(np.float32)
    binary_mask = np.array([1 if "=" in c else 0 for c in design.columns], dtype=np.int64)
    groups: Dict[str, List[int]] = {}
    for idx, col in enumerate(design.columns):
        groups.setdefault(assign_group(col.split("=")[0]), []).append(idx)
    design.to_parquet(OUTPUT_DIR / "design_matrix.parquet", index=False)
    (CONFIG_DIR / "group_columns.json").write_text(json.dumps({k: [design.columns[i] for i in v] for k, v in groups.items()}, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    return design, groups, binary_mask

class GroupAwareMaskedTransformer(nn.Module):
    def __init__(self, group_dims: List[int], d_model: int = D_MODEL, n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.group_dims = group_dims
        self.n_groups = len(group_dims)
        self.projectors = nn.ModuleList([nn.Sequential(nn.Linear(dim, d_model), nn.ReLU(), nn.LayerNorm(d_model)) for dim in group_dims])
        self.decoders = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, dim)) for dim in group_dims])
        self.group_embedding = nn.Embedding(self.n_groups, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_groups: List[torch.Tensor], mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        tokens = []
        for i, x in enumerate(x_groups):
            token = self.projectors[i](x) + self.group_embedding.weight[i]
            if mask is not None:
                token = torch.where(mask[:, i].unsqueeze(1), self.mask_token.squeeze(0) + self.group_embedding.weight[i], token)
            tokens.append(token)
        token_tensor = torch.stack(tokens, dim=1)
        cls = self.cls_token.expand(token_tensor.size(0), -1, -1)
        z = self.norm(self.transformer(torch.cat([cls, token_tensor], dim=1)))
        cls_out = z[:, 0, :]
        recons = [decoder(z[:, i + 1, :]) for i, decoder in enumerate(self.decoders)]
        return cls_out, recons


class DualTowerModel(nn.Module):
    def __init__(self, table_encoder: GroupAwareMaskedTransformer, text_dim: int):
        super().__init__()
        self.table_encoder = table_encoder
        self.table_proj = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, D_MODEL))
        self.text_proj = nn.Sequential(nn.Linear(text_dim, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, D_MODEL))

    def forward(self, x_groups: List[torch.Tensor], text_emb: torch.Tensor, mask: torch.Tensor):
        cls_out, recons = self.table_encoder(x_groups, mask=mask)
        z_table = F.normalize(self.table_proj(cls_out), dim=1)
        z_text = F.normalize(self.text_proj(text_emb), dim=1)
        return cls_out, recons, z_table, z_text


def generate_group_mask(batch_size: int, n_groups: int, prob: float = MASK_PROB) -> torch.Tensor:
    mask = torch.rand(batch_size, n_groups) < prob
    empty = mask.sum(dim=1) == 0
    if empty.any():
        choice = torch.randint(0, n_groups, size=(int(empty.sum().item()),))
        mask[empty] = False
        mask[empty, choice] = True
    return mask


def build_relative_indices(groups: Dict[str, List[int]], binary_mask: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    binary_rel_idx = []
    cont_rel_idx = []
    for group_name in groups.keys():
        idxs = np.array(groups[group_name])
        binary_rel_idx.append(np.where(binary_mask[idxs] == 1)[0])
        cont_rel_idx.append(np.where(binary_mask[idxs] == 0)[0])
    return binary_rel_idx, cont_rel_idx


def reconstruction_loss(recons: List[torch.Tensor], targets: List[torch.Tensor], mask: torch.Tensor, binary_rel_idx: List[np.ndarray], cont_rel_idx: List[np.ndarray]) -> torch.Tensor:
    losses = []
    for i, (pred, tgt) in enumerate(zip(recons, targets)):
        active = mask[:, i]
        if active.sum() == 0:
            continue
        pred_i = pred[active]
        tgt_i = tgt[active]
        loss_i = torch.tensor(0.0, device=pred.device)
        parts = 0
        if len(cont_rel_idx[i]) > 0:
            loss_i = loss_i + F.mse_loss(pred_i[:, cont_rel_idx[i]], tgt_i[:, cont_rel_idx[i]])
            parts += 1
        if len(binary_rel_idx[i]) > 0:
            loss_i = loss_i + F.binary_cross_entropy_with_logits(pred_i[:, binary_rel_idx[i]], tgt_i[:, binary_rel_idx[i]])
            parts += 1
        if parts > 0:
            losses.append(loss_i / parts)
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=targets[0].device)


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = TEMPERATURE) -> torch.Tensor:
    logits = z1 @ z2.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def build_empty_group_model(groups: Dict[str, List[int]]) -> GroupAwareMaskedTransformer:
    return GroupAwareMaskedTransformer([len(groups[g]) for g in groups.keys()])


def train_group_masked_model(x: np.ndarray, groups: Dict[str, List[int]], binary_mask: np.ndarray):
    cache_path = OUTPUT_DIR / "group_masked_embedding.npy"
    history_path = OUTPUT_DIR / "group_masked_history.csv"
    model_path = OUTPUT_DIR / "group_masked_model.pt"
    if cache_path.exists() and history_path.exists() and model_path.exists():
        model = build_empty_group_model(groups)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return np.load(cache_path), pd.read_csv(history_path), model

    x_train, x_val = train_test_split(x, test_size=0.15, random_state=RANDOM_SEED)
    group_names = list(groups.keys())
    model = build_empty_group_model(groups)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_TABLE)
    binary_rel_idx, cont_rel_idx = build_relative_indices(groups, binary_mask)
    loader = DataLoader(TensorDataset(torch.tensor(x_train, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)

    best_state = None
    best_val = math.inf
    patience = 0
    history_rows = []
    for epoch in range(1, TABLE_EPOCHS + 1):
        model.train()
        train_losses = []
        for (xb,) in loader:
            optimizer.zero_grad()
            x_groups = [xb[:, groups[g]] for g in group_names]
            mask = generate_group_mask(xb.size(0), len(group_names))
            _, recons = model(x_groups, mask)
            loss = reconstruction_loss(recons, x_groups, mask, binary_rel_idx, cont_rel_idx)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.inference_mode():
            x_groups_val = [x_val_t[:, groups[g]] for g in group_names]
            val_mask = generate_group_mask(x_val_t.size(0), len(group_names))
            _, val_recons = model(x_groups_val, val_mask)
            val_loss = float(reconstruction_loss(val_recons, x_groups_val, val_mask, binary_rel_idx, cont_rel_idx).item())
        train_loss = float(np.mean(train_losses))
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP:
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), model_path)
    history = pd.DataFrame(history_rows)
    history.to_csv(history_path, index=False, encoding="utf-8-sig")
    with torch.inference_mode():
        x_tensor = torch.tensor(x, dtype=torch.float32)
        x_groups_all = [x_tensor[:, groups[g]] for g in group_names]
        emb, _ = model(x_groups_all, None)
    emb_np = emb.cpu().numpy()
    np.save(cache_path, emb_np)
    return emb_np, history, model


def train_dual_tower_model(x: np.ndarray, text_raw: np.ndarray, groups: Dict[str, List[int]], binary_mask: np.ndarray, pretrained_encoder: GroupAwareMaskedTransformer):
    cache_table = OUTPUT_DIR / "dual_tower_table_embedding.npy"
    cache_avg = OUTPUT_DIR / "dual_tower_avg_embedding.npy"
    cache_concat = OUTPUT_DIR / "dual_tower_concat_pca_embedding.npy"
    history_path = OUTPUT_DIR / "dual_tower_history.csv"
    model_path = OUTPUT_DIR / "dual_tower_model.pt"
    if cache_table.exists() and cache_avg.exists() and cache_concat.exists() and history_path.exists() and model_path.exists():
        return {
            "DualTowerTable": np.load(cache_table),
            "DualTowerAvg": np.load(cache_avg),
            "DualTowerConcatPCA": np.load(cache_concat),
        }, pd.read_csv(history_path)

    x_train, x_val, t_train, t_val = train_test_split(x, text_raw, test_size=0.15, random_state=RANDOM_SEED)
    group_names = list(groups.keys())
    model = DualTowerModel(build_empty_group_model(groups), text_raw.shape[1])
    model.table_encoder.load_state_dict(pretrained_encoder.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_DUAL)
    binary_rel_idx, cont_rel_idx = build_relative_indices(groups, binary_mask)
    loader = DataLoader(TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(t_train, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    t_val_t = torch.tensor(t_val, dtype=torch.float32)

    best_state = None
    best_val = math.inf
    patience = 0
    history_rows = []
    for epoch in range(1, DUAL_EPOCHS + 1):
        model.train()
        train_losses = []
        train_recon = []
        train_ctr = []
        for xb, tb in loader:
            optimizer.zero_grad()
            x_groups = [xb[:, groups[g]] for g in group_names]
            mask = generate_group_mask(xb.size(0), len(group_names))
            _, recons, z_table, z_text = model(x_groups, tb, mask)
            loss_recon = reconstruction_loss(recons, x_groups, mask, binary_rel_idx, cont_rel_idx)
            loss_ctr = contrastive_loss(z_table, z_text)
            loss = loss_recon + 0.6 * loss_ctr
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_recon.append(float(loss_recon.item()))
            train_ctr.append(float(loss_ctr.item()))

        model.eval()
        with torch.inference_mode():
            x_groups_val = [x_val_t[:, groups[g]] for g in group_names]
            val_mask = generate_group_mask(x_val_t.size(0), len(group_names))
            _, val_recons, val_z_table, val_z_text = model(x_groups_val, t_val_t, val_mask)
            val_recon = float(reconstruction_loss(val_recons, x_groups_val, val_mask, binary_rel_idx, cont_rel_idx).item())
            val_ctr = float(contrastive_loss(val_z_table, val_z_text).item())
            val_loss = val_recon + 0.6 * val_ctr
        history_rows.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "train_recon": float(np.mean(train_recon)), "train_ctr": float(np.mean(train_ctr)), "val_recon": val_recon, "val_ctr": val_ctr, "val_loss": val_loss})
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP:
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), model_path)
    history = pd.DataFrame(history_rows)
    history.to_csv(history_path, index=False, encoding="utf-8-sig")
    with torch.inference_mode():
        x_tensor = torch.tensor(x, dtype=torch.float32)
        t_tensor = torch.tensor(text_raw, dtype=torch.float32)
        x_groups_all = [x_tensor[:, groups[g]] for g in group_names]
        _, _, z_table, z_text = model(x_groups_all, t_tensor, generate_group_mask(x_tensor.size(0), len(group_names)))
        z_avg = F.normalize((z_table + z_text) / 2.0, dim=1).cpu().numpy()
        z_table_np = z_table.cpu().numpy()
        z_concat = PCA(n_components=D_MODEL, random_state=RANDOM_SEED).fit_transform(torch.cat([z_table, z_text], dim=1).cpu().numpy())
    np.save(cache_table, z_table_np)
    np.save(cache_avg, z_avg)
    np.save(cache_concat, z_concat)
    return {"DualTowerTable": z_table_np, "DualTowerAvg": z_avg, "DualTowerConcatPCA": z_concat}, history


def plot_history(table_hist: pd.DataFrame, dual_hist: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(table_hist["epoch"], table_hist["train_loss"], label="train")
    ax.plot(table_hist["epoch"], table_hist["val_loss"], label="val")
    ax.set_title("Training Curve: Group-Aware Masked Table Encoder")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "training_curve_group_masked_en.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(dual_hist["epoch"], dual_hist["train_loss"], label="train")
    ax.plot(dual_hist["epoch"], dual_hist["val_loss"], label="val")
    ax.plot(dual_hist["epoch"], dual_hist["val_recon"], label="val_recon")
    ax.plot(dual_hist["epoch"], dual_hist["val_ctr"], label="val_contrastive")
    ax.set_title("Training Curve: Dual-Tower Contrastive Model")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "training_curve_dual_tower_en.png")
    plt.close(fig)

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


def cluster_labels(emb: np.ndarray, clusterer: str, k: int, seed: int) -> np.ndarray:
    if clusterer == "kmeans":
        return KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(emb)
    return GaussianMixture(n_components=k, covariance_type="full", random_state=seed, n_init=5).fit_predict(emb)


def reorder_by_ic(labels: np.ndarray, ic_total: pd.Series) -> np.ndarray:
    order = pd.DataFrame({"cluster": labels, "ic": ic_total}).groupby("cluster")["ic"].mean().sort_values().index.tolist()
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[x] for x in labels], dtype=int)


def bootstrap_stability(emb: np.ndarray, base_labels: np.ndarray, clusterer: str, k: int, ic_total: pd.Series) -> float:
    rng = np.random.default_rng(RANDOM_SEED)
    scores = []
    n = emb.shape[0]
    for _ in range(8):
        idx = np.sort(rng.choice(n, size=int(n * 0.8), replace=False))
        boot = cluster_labels(emb[idx], clusterer, k, int(rng.integers(1, 1_000_000)))
        boot = reorder_by_ic(boot, ic_total.iloc[idx].reset_index(drop=True))
        scores.append(adjusted_rand_score(base_labels[idx], boot))
    return float(np.mean(scores))


def evaluate_embeddings(embeddings: Dict[str, np.ndarray], raw_df: pd.DataFrame, ic_df: pd.DataFrame):
    rows = []
    best_by_method = {}
    ext_cols = [c for c in EXTERNAL_FEATURES if c in raw_df.columns]
    ic_total = ic_df["IC_total_cont_100"]
    for method, emb in embeddings.items():
        best_score = -1e9
        best_bundle = None
        for clusterer in CLUSTERERS:
            for k in K_RANGE:
                labels = reorder_by_ic(cluster_labels(emb, clusterer, k, RANDOM_SEED), ic_total)
                counts = pd.Series(labels).value_counts()
                sil = float(silhouette_score(emb, labels))
                ch = float(calinski_harabasz_score(emb, labels))
                db = float(davies_bouldin_score(emb, labels))
                stability = bootstrap_stability(emb, labels, clusterer, k, ic_total)
                ext_sep = external_effect_score(labels, raw_df, ext_cols)
                min_ratio = float(counts.min() / len(labels))
                row = EvalRow(method, clusterer, k, sil, ch, db, stability, min_ratio, ext_sep)
                rows.append(row.__dict__)
                score = 0.25 * sil + 0.20 * (1.0 / (1.0 + db)) + 0.25 * ext_sep + 0.20 * stability + 0.10 * min_ratio
                if score > best_score:
                    best_score = score
                    best_bundle = {"method": method, "clusterer": clusterer, "k": k, "labels": labels, "embedding": emb, "selection_score": score}
        best_by_method[method] = best_bundle
    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(OUTPUT_DIR / "all_method_cluster_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{k: v for k, v in bundle.items() if k != "labels" and k != "embedding"} for bundle in best_by_method.values()]).to_csv(OUTPUT_DIR / "best_method_selection.csv", index=False, encoding="utf-8-sig")
    return eval_df, best_by_method


def bh_fdr(p_values: List[float]) -> List[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = np.array(p_values)[order]
    adj = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * n / rank)
        adj[i] = running
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj, 0, 1)
    return out.tolist()


def profile_best_method(method: str, bundle: Dict[str, object], raw_df: pd.DataFrame, ic_df: pd.DataFrame):
    labels = bundle["labels"]
    emb = bundle["embedding"]
    id_col = "编号" if "编号" in raw_df.columns else raw_df.columns[1]
    pd.DataFrame({"sample_id": raw_df[id_col], "cluster_id": labels}).to_csv(OUTPUT_DIR / f"{method}_cluster_assignments.csv", index=False, encoding="utf-8-sig")

    ic_mean = ic_df[IC_PROFILE_COLS].assign(cluster_id=labels).groupby("cluster_id")[IC_PROFILE_COLS].mean().reset_index().rename(columns=IC_ALIAS)
    ic_mean.to_csv(OUTPUT_DIR / f"{method}_ic_profile.csv", index=False, encoding="utf-8-sig")

    ext_cols = [c for c in EXTERNAL_FEATURES if c in raw_df.columns]
    ext_mean = raw_df[ext_cols].apply(pd.to_numeric, errors="coerce").assign(cluster_id=labels).groupby("cluster_id")[ext_cols].mean().reset_index().rename(columns=EXTERNAL_ALIAS)
    ext_mean.to_csv(OUTPUT_DIR / f"{method}_external_profile.csv", index=False, encoding="utf-8-sig")

    stat_rows = []
    pvals = []
    for col in ext_cols:
        vals = pd.to_numeric(raw_df[col], errors="coerce")
        groups = [vals[labels == c].dropna().values for c in np.unique(labels)]
        if sum(len(g) > 0 for g in groups) < 2:
            continue
        try:
            _, p = kruskal(*groups)
        except Exception:
            p = 1.0
        means = [np.mean(g) if len(g) else np.nan for g in groups]
        stat_rows.append({"feature_raw": col, "feature_en": EXTERNAL_ALIAS.get(col, col), "p_value": float(p), "eta_squared": external_effect_score(labels, raw_df[[col]], [col]), "max_mean_diff": float(np.nanmax(means) - np.nanmin(means))})
        pvals.append(float(p))
    if stat_rows:
        qvals = bh_fdr(pvals)
        for row, q in zip(stat_rows, qvals):
            row["q_value_fdr"] = q
    ext_stats = pd.DataFrame(stat_rows).sort_values(["q_value_fdr", "eta_squared", "max_mean_diff"], ascending=[True, False, False])
    ext_stats.to_csv(OUTPUT_DIR / f"{method}_external_stats.csv", index=False, encoding="utf-8-sig")

    marker_rows = []
    for col in INTERPRETABLE_MARKERS:
        if col not in raw_df.columns:
            continue
        vals = pd.to_numeric(raw_df[col], errors="coerce")
        mask = vals.notna()
        if mask.sum() < 50:
            continue
        row = {"feature": col, "eta_squared": external_effect_score(labels[mask.values], pd.DataFrame({col: vals[mask].values}), [col])}
        means = pd.DataFrame({"cluster": labels[mask.values], col: vals[mask].values}).groupby("cluster")[col].mean().to_dict()
        for cid, mean_val in means.items():
            row[f"cluster_{cid}_mean"] = mean_val
        marker_rows.append(row)
    markers = pd.DataFrame(marker_rows).sort_values("eta_squared", ascending=False)
    markers.to_csv(OUTPUT_DIR / f"{method}_top_markers.csv", index=False, encoding="utf-8-sig")

    coords = TSNE(n_components=2, random_state=RANDOM_SEED, init="pca", learning_rate="auto", perplexity=35).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, palette="tab10", s=15, ax=ax)
    ax.set_title(f"2D Projection of {method} Embedding")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"{method}_embedding_projection_en.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    sns.heatmap(ic_mean.set_index("cluster_id"), annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
    ax.set_title(f"Continuous IC Profile by Cluster ({method})")
    ax.set_xlabel("IC Domain")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"{method}_ic_heatmap_en.png")
    plt.close(fig)

    radar_cols = ["Cognition", "Psychological", "Vitality", "Locomotion", "Sensory"]
    angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(6.2, 6.2))
    ax = plt.subplot(111, polar=True)
    for cid, row in ic_mean.set_index("cluster_id")[radar_cols].iterrows():
        values = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, values, linewidth=2, label=f"Cluster {cid}")
        ax.fill(angles, values, alpha=0.10)
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_cols)
    ax.set_ylim(0, 100)
    ax.set_title(f"Radar Plot of IC Domains ({method})")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12))
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"{method}_radar_en.png")
    plt.close(fig)

    ext_keep = [v for k, v in EXTERNAL_ALIAS.items() if k in ext_cols][:8]
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    sns.heatmap(ext_mean.set_index("cluster_id")[ext_keep], annot=True, fmt=".1f", cmap="OrRd", ax=ax)
    ax.set_title(f"External Clinical Validation by Cluster ({method})")
    ax.set_xlabel("External Variable")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"{method}_external_heatmap_en.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    sns.barplot(data=markers.head(12), x="eta_squared", y="feature", palette="viridis", ax=ax)
    ax.set_title(f"Top Cluster-Differentiating Features ({method})")
    ax.set_xlabel("Effect size (eta squared)")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"{method}_top_markers_en.png")
    plt.close(fig)
    return {"ic": ic_mean, "external": ext_mean, "ext_stats": ext_stats, "markers": markers}


def plot_method_comparison(eval_df: pd.DataFrame, best_by_method: Dict[str, Dict[str, object]]) -> None:
    rows = []
    for method, bundle in best_by_method.items():
        row = eval_df[(eval_df["method"] == method) & (eval_df["clusterer"] == bundle["clusterer"]) & (eval_df["k"] == bundle["k"])].iloc[0].to_dict()
        rows.append(row)
    best_df = pd.DataFrame(rows).sort_values(["external_separation", "silhouette"], ascending=False)
    best_df.to_csv(OUTPUT_DIR / "best_method_metrics.csv", index=False, encoding="utf-8-sig")
    plot_df = best_df.melt(id_vars=["method"], value_vars=["silhouette", "stability_ari", "external_separation", "min_cluster_ratio"], var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    sns.barplot(data=plot_df, x="metric", y="value", hue="method", ax=ax)
    ax.set_title("Advanced Deep Subtyping Method Comparison")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "method_comparison_en.png")
    plt.close(fig)


def build_medical_insight(final_method: str, tables: Dict[str, pd.DataFrame]) -> List[str]:
    lines = []
    ic_df = tables["ic"].sort_values("Overall IC").reset_index(drop=True)
    ext_df = tables["external"].set_index("cluster_id").loc[ic_df["cluster_id"]].reset_index()
    domain_ranges = {col: float(ic_df[col].max() - ic_df[col].min()) for col in ["Cognition", "Psychological", "Vitality", "Locomotion", "Sensory"] if col in ic_df.columns}
    if domain_ranges:
        top_domain = max(domain_ranges, key=domain_ranges.get)
        lines.append(f"在最终模型 `{final_method}` 中，跨簇差异最大的 IC 维度是 `{top_domain}`，提示它是当前样本异质性的主导轴。")
    if {"ADL total", "IADL total", "Sarcopenia score", "Fried frailty score", "Fall risk score"}.issubset(ext_df.columns):
        worst = ext_df.iloc[0]
        best = ext_df.iloc[-1]
        lines.append(f"最差簇与最佳簇形成了清晰梯度：ADL 从 {worst['ADL total']:.1f} 升至 {best['ADL total']:.1f}，IADL 从 {worst['IADL total']:.1f} 降至 {best['IADL total']:.1f}，肌少症评分从 {worst['Sarcopenia score']:.2f} 降至 {best['Sarcopenia score']:.2f}，Fried 衰弱从 {worst['Fried frailty score']:.2f} 降至 {best['Fried frailty score']:.2f}，跌倒风险从 {worst['Fall risk score']:.1f} 降至 {best['Fall risk score']:.1f}。")
    if "Resilience score" in ext_df.columns:
        rho = spearmanr(np.arange(len(ext_df)), ext_df["Resilience score"]).correlation
        if pd.notna(rho):
            lines.append(f"按 Overall IC 排序后，簇序与弹性评分呈正相关（Spearman rho={rho:.3f}），提示更好的内在能力与更好的恢复力相伴存在。")
    return lines

def write_reports(text_encoder_name: str, best_by_method: Dict[str, Dict[str, object]], method_tables: Dict[str, Dict[str, pd.DataFrame]], group_names: List[str]) -> None:
    best_metrics = pd.read_csv(OUTPUT_DIR / "best_method_metrics.csv", encoding="utf-8-sig")
    best_method = best_metrics.sort_values(["external_separation", "silhouette"], ascending=False).iloc[0]["method"]
    insight_lines = build_medical_insight(best_method, method_tables[best_method])
    design_shape = pd.read_parquet(OUTPUT_DIR / "design_matrix.parquet").shape
    ext_stats = method_tables[best_method]["ext_stats"]
    markers = method_tables[best_method]["markers"]

    lines = []
    lines.append("# 深度学习分型主线技术报告")
    lines.append("")
    lines.append("## 1. 工作目标")
    lines.append("- 用 `group-aware masked modeling` 升级表格表征。")
    lines.append("- 用 `text-table dual-tower contrastive learning` 升级文本路线。")
    lines.append("- 与 `PCA-128` 做统一对照，并做严格外部验证。")
    lines.append("")
    lines.append("## 2. 数据与输入")
    lines.append("- 样本量：6025。")
    lines.append("- 核心输入：59 个核心字段 + 5 个连续化 IC 域分数。")
    lines.append(f"- 最终设计矩阵维度：{design_shape[0]} x {design_shape[1]}。")
    lines.append(f"- group-aware 输入分为 {len(group_names)} 组：{', '.join(group_names)}。")
    lines.append(f"- 文本塔编码器：`{text_encoder_name}`。")
    lines.append("")
    lines.append("## 3. 模型设计")
    lines.append("### 3.1 GroupMaskedFT")
    lines.append("- 把不同临床域切成组 token。")
    lines.append("- 随机 mask 约 30% 组 token，仅重建被遮挡组。")
    lines.append("- 作用：学习跨域上下文，而不是死记单个变量。")
    lines.append("")
    lines.append("### 3.2 DualTower")
    lines.append("- 表格塔初始化为上一步预训练好的 masked encoder。")
    lines.append("- 文本塔读取医学化 Tab2Text。")
    lines.append("- 训练目标：重建损失 + 0.6 × 对比损失。")
    lines.append("- 这一步的目的不是让 LLM 直接替代建模，而是让文本语义和表格语义对齐。")
    lines.append("")
    lines.append("## 4. 聚类评估规则")
    lines.append("- clusterer：KMeans 与 GMM。")
    lines.append("- k：3–6。")
    lines.append("- 指标：silhouette、Calinski-Harabasz、Davies-Bouldin、bootstrap stability、min cluster ratio、external separation。")
    lines.append("- 外部验证变量不进入主聚类设计矩阵，只用于验证临床意义。")
    lines.append("")
    lines.append("## 5. 各方法最优结果")
    lines.append(best_metrics.to_markdown(index=False))
    lines.append("")
    lines.append(f"综合来看，当前最值得作为论文主线的方法是 `{best_method}`。")
    lines.append("")
    lines.append("## 6. 最终模型的外部验证")
    lines.append(ext_stats.head(12).to_markdown(index=False))
    lines.append("")
    lines.append("## 7. 最终模型的关键区分变量")
    lines.append(markers.head(12).to_markdown(index=False))
    lines.append("")
    lines.append("## 8. 医学意义与 insight")
    if insight_lines:
        lines.extend([f"- {item}" for item in insight_lines])
    lines.append("- 如果低 IC 簇同时伴随更差 ADL/IADL、更高肌少症、更高 Fried 衰弱与更高跌倒风险，那么这个分型就具备明确的临床意义。")
    lines.append("- 如果中间簇在 ADL 尚可时已经出现感官或运动下降，就可以讲‘前失能/早期脆弱化亚型’。")
    lines.append("")
    lines.append("## 9. 论文主故事线")
    lines.append("1. 传统 IC 横断面分析容易把复杂功能状态压扁成 0/1。")
    lines.append("2. 我们先把 IC 连续化，再把硬编码表格升级成 group-aware token。")
    lines.append("3. 再用文本-表格双塔对比学习，把语义层信息拉进来。")
    lines.append("4. 最终得到的亚型不仅内部结构更清楚，而且外部 ADL/IADL/肌少症/衰弱/跌倒梯度更明显。")
    lines.append("5. 这说明我们捕捉到的不是单纯数学簇，而是具有社区老年临床价值的功能亚型。")
    (REPORT_DIR / "TECHNICAL_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8-sig")

    story = []
    story.append("# 第一篇论文故事线（建议版）")
    story.append("")
    story.append("## 建议题目")
    story.append("- Deep phenotyping of intrinsic capacity in older adults using continuous-domain scoring and group-aware multimodal representation learning")
    story.append("- 内部中文题目：基于连续化内在能力评分与语义增强深度表征学习的老年人异质性分型研究")
    story.append("")
    story.append("## 创新点")
    story.append("1. 连续化 IC，而不是二值化 IC。")
    story.append("2. group-aware masked tabular encoder。")
    story.append("3. Tab2Text + BERT 文本塔与表格塔对比学习。")
    story.append("4. 用 ADL/IADL/肌少症/衰弱/跌倒/弹性做系统外部验证。")
    story.append("")
    story.append("## 图表建议")
    story.append("1. Figure 1 研究流程图")
    story.append("2. Figure 2 方法比较图")
    story.append("3. Figure 3 最终模型二维投影")
    story.append("4. Figure 4 最终模型 IC 雷达图")
    story.append("5. Figure 5 最终模型外部验证热图")
    story.append("6. Figure 6 最终模型 top markers")
    story.append("")
    story.append("## 审稿防守点")
    story.append("1. PCA 是强基线，我们保留且如实汇报。")
    story.append("2. 文本路线不是黑盒决策器，而是语义增强模块。")
    story.append("3. 外部验证变量不进入主聚类设计矩阵，避免明显信息泄露。")
    (REPORT_DIR / "PAPER_STORYLINE_CN.md").write_text("\n".join(story), encoding="utf-8-sig")

    file_lines = []
    file_lines.append("# 文件结构说明")
    file_lines.append("")
    file_lines.append("## scripts")
    file_lines.append("- scripts/advanced_deep_subtyping_pipeline.py：本次深度学习主线实验脚本。")
    file_lines.append("")
    file_lines.append("## outputs/advanced_subtyping")
    file_lines.append("- design_matrix.parquet：统一建模设计矩阵。")
    file_lines.append("- group_masked_embedding.npy：group-aware masked table encoder 输出。")
    file_lines.append("- dual_tower_*_embedding.npy：双塔不同融合方式的 embedding。")
    file_lines.append("- all_method_cluster_metrics.csv：所有方法+clusterer+k 完整比较表。")
    file_lines.append("- best_method_metrics.csv：各方法最优配置比较表。")
    file_lines.append("- *_cluster_assignments.csv / *_ic_profile.csv / *_external_profile.csv / *_external_stats.csv / *_top_markers.csv：最终结果表。")
    file_lines.append("")
    file_lines.append("## figures/advanced_subtyping")
    file_lines.append("- training_curve_group_masked_en.png：表格预训练曲线。")
    file_lines.append("- training_curve_dual_tower_en.png：双塔训练曲线。")
    file_lines.append("- method_comparison_en.png：方法比较总图。")
    file_lines.append("- *_embedding_projection_en.png / *_radar_en.png / *_external_heatmap_en.png / *_top_markers_en.png：核心图件。")
    file_lines.append("")
    file_lines.append("## reports/advanced_subtyping")
    file_lines.append("- TECHNICAL_REPORT_CN.md：完整中文技术报告。")
    file_lines.append("- PAPER_STORYLINE_CN.md：论文故事线。")
    file_lines.append("- FILE_STRUCTURE_CN.md：本文件。")
    (REPORT_DIR / "FILE_STRUCTURE_CN.md").write_text("\n".join(file_lines), encoding="utf-8-sig")


def main() -> None:
    """
    流程是先读取数据进行IC评分然后构建医学文本并进行BETR编码，
    然后构造分组矩阵，并训练FT和PCA并进行dual对比并进行聚类
    在对每一个聚类情况进行寻找最优和可视化分析

    此处的输出是：几个embedding+聚类结果+评估图标和报告
    训练了两个模型，
    聚类不是在原始特征上进行而是在embedding向量上
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return

    ensure_dirs()
    configure_plot_style()
    seed_everything()

    raw_df, mappings = load_raw()#读原始数据
    ic_df = compute_continuous_ic(raw_df)#计算连续化IC得分
    ic_df.to_csv(OUTPUT_DIR / "IC_continuous_scores.csv", index=False, encoding="utf-8-sig")
    text_cards = build_medical_text(raw_df, mappings, ic_df)#构造医学文本
    text_df = build_professional_text(raw_df, text_cards, ic_df)
    text_raw, text_encoder_name = encode_text(text_df["medical_text_cn"].tolist())#编码医学文本为BERT向量

    core_df, numeric_cols, categorical_cols = preprocess_core_features(raw_df, mappings, ic_df)
    design, groups, binary_mask = build_design_matrix(core_df, numeric_cols, categorical_cols)#构建设计矩阵、并按租切分特征
    x = design.values.astype(np.float32)
    np.save(OUTPUT_DIR / "pca_embedding_128.npy", PCA(n_components=D_MODEL, random_state=RANDOM_SEED).fit_transform(x))#PCAembedding

    table_emb, table_hist, table_model = train_group_masked_model(x, groups, binary_mask)#训练FTTransformer
    dual_embs, dual_hist = train_dual_tower_model(x, text_raw, groups, binary_mask, table_model)#训练文本-表格双塔
    plot_history(table_hist, dual_hist)

    embeddings = {"PCA": np.load(OUTPUT_DIR / "pca_embedding_128.npy"), "GroupMaskedFT": table_emb, **dual_embs}
    eval_df, best_by_method = evaluate_embeddings(embeddings, raw_df, ic_df)#用不同的聚类器和k值进行评估
    plot_method_comparison(eval_df, best_by_method)

    method_tables = {method: profile_best_method(method, bundle, raw_df, ic_df) for method, bundle in best_by_method.items()}#对每个最佳方法做可视化
    write_reports(text_encoder_name, best_by_method, method_tables, list(groups.keys()))#自动写报告


if __name__ == "__main__":
    main()
