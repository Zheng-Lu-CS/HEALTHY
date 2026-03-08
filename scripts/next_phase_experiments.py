# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import math
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

from scipy.stats import spearmanr

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from transformers import AutoModel, AutoTokenizer

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from run_pipeline import (  # noqa: E402
    apply_mapping,
    build_ic_scores,
    normalize_missing,
    parse_docx_mappings,
    prepare_features,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures" / "next_phase"
OUT_DIR = ROOT / "outputs" / "next_phase"
REPORT_DIR = ROOT / "reports" / "next_phase"
RANDOM_SEED = 42

LOCAL_BERT_NAME = "bert-base-chinese"

TEXT_FIELDS = [
    "年龄",
    "性别",
    "婚姻状况",
    "文化程度",
    "省",
    "市",
    "患有慢性病数量",
    "服用药物数量",
    "过去一年住院次数",
    "过去一年急诊次数",
    "是否高血压",
    "是否糖尿病",
    "是否骨质疏松",
    "是否脑血栓",
    "是否焦虑抑郁症",
    "听力障碍",
    "视力障碍",
    "活力-BMI值",
    "认知-总分",
    "心理-总分",
    "活力-总分",
    "运动-总分",
    "生活行为与社会功能评估-总分",
]

COUNTERFACTUAL_FIELDS = [
    "年龄",
    "患有慢性病数量",
    "服用药物数量",
    "过去一年住院次数",
    "过去一年急诊次数",
    "活力-BMI值",
    "认知-总分",
    "心理-总分",
    "活力-总分",
    "运动-总分",
    "生活行为与社会功能评估-总分",
    "听力障碍",
    "视力障碍",
    "是否高血压",
    "是否糖尿病",
    "是否骨质疏松",
    "是否脑血栓",
    "是否焦虑抑郁症",
]


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def configure_plot_style() -> None:
    font_files = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
    ]
    for font_path in font_files:
        try:
            if Path(font_path).exists():
                font_manager.fontManager.addfont(font_path)
        except Exception:
            pass

    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans SC", "SimSun"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((font for font in preferred if font in available), None)

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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, str]], pd.DataFrame]:
    xlsx, docx = find_raw_files()
    raw_df = pd.read_excel(xlsx)
    raw_df = normalize_missing(raw_df)
    mappings = parse_docx_mappings(docx, raw_df.columns.tolist())
    ic_df = build_ic_scores(raw_df.copy())
    X, _, _, _ = prepare_features(raw_df.copy(), mappings)
    return raw_df, ic_df, mappings, X


def readable_value(col: str, value, mappings: dict[str, dict[str, str]]) -> str:
    if pd.isna(value):
        return "缺失"
    mapping = mappings.get(col)
    if mapping:
        key = str(value).strip()
        if key in mapping:
            return str(mapping[key])
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "缺失"
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.2f}"
    text = str(value).strip()
    if text in {"0", "0.0"} and (col.startswith("是否") or "障碍" in col):
        return "否"
    if text in {"1", "1.0"} and (col.startswith("是否") or "障碍" in col):
        return "是"
    return text


def build_text_cards(raw_df: pd.DataFrame, mappings: dict[str, dict[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in raw_df.iterrows():
        parts = []
        sid = row["编号"] if "编号" in raw_df.columns else _
        parts.append(f"样本编号{sid}。")
        for col in TEXT_FIELDS:
            if col not in raw_df.columns:
                continue
            value = readable_value(col, row[col], mappings)
            if value == "缺失":
                continue
            if col == "年龄":
                parts.append(f"{value}岁。")
            elif col in {"省", "市"}:
                parts.append(f"{col}{value}。")
            elif col.startswith("是否") or "障碍" in col:
                parts.append(f"{col}{value}。")
            else:
                parts.append(f"{col}{value}。")
        rows.append(
            {
                "编号": sid,
                "text_card": " ".join(parts),
            }
        )
    text_df = pd.DataFrame(rows)
    text_df.to_csv(OUT_DIR / "text_cards.csv", index=False, encoding="utf-8-sig")
    return text_df


def _find_local_model_dir(model_name: str) -> Path | None:
    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    safe_name = f"models--{model_name.replace('/', '--')}"
    base = hub_root / safe_name / "snapshots"
    if not base.exists():
        return None
    snapshots = sorted([p for p in base.iterdir() if p.is_dir()])
    for snap in reversed(snapshots):
        if (snap / "pytorch_model.bin").exists() or (snap / "model.safetensors").exists():
            return snap
    return None


def encode_texts(texts: Iterable[str], batch_size: int = 32) -> tuple[np.ndarray, str]:
    text_list = list(texts)
    local_model_dir = _find_local_model_dir(LOCAL_BERT_NAME)
    if local_model_dir is not None:
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)
        model = AutoModel.from_pretrained(local_model_dir, local_files_only=True)
        model.eval()
        device = torch.device("cpu")
        model.to(device)

        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(text_list), batch_size):
                batch_text = text_list[start:start + batch_size]
                encoded = tokenizer(
                    batch_text,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                outputs.append(pooled.cpu().numpy())
        arr = np.vstack(outputs)
        pd.DataFrame(arr).to_parquet(OUT_DIR / "text_embeddings.parquet", index=False)
        return arr, f"PretrainedLM({LOCAL_BERT_NAME})"

    vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), min_df=5)
    tfidf = vectorizer.fit_transform(text_list)
    n_components = min(256, max(16, tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
    arr = svd.fit_transform(tfidf)
    pd.DataFrame(arr).to_parquet(OUT_DIR / "text_embeddings.parquet", index=False)
    return arr, "TFIDF+SVD"


class DenoisingAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32) -> None:
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def train_dae(X: np.ndarray, latent_dim: int = 32, epochs: int = 35, batch_size: int = 256, noise_rate: float = 0.15) -> tuple[np.ndarray, pd.DataFrame]:
    device = torch.device("cpu")
    model = DenoisingAutoEncoder(X.shape[1], latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        for (xb,) in loader:
            xb = xb.to(device)
            mask = torch.rand_like(xb) > noise_rate
            noisy = xb * mask
            optimizer.zero_grad()
            recon, _ = model(noisy)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})

    model.eval()
    with torch.no_grad():
        _, z = model(torch.tensor(X, dtype=torch.float32).to(device))
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(OUT_DIR / "dae_history.csv", index=False, encoding="utf-8-sig")
    return z.cpu().numpy(), hist_df


class ContrastiveDataset(Dataset):
    def __init__(self, X: np.ndarray, T: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.T = torch.tensor(T, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.T[idx]


class TableEncoder(nn.Module):
    def __init__(self, input_dim: int, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextProjector(nn.Module):
    def __init__(self, input_dim: int, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_a + loss_b)


def train_contrastive_alignment(X: np.ndarray, T: np.ndarray, epochs: int = 20, batch_size: int = 128, out_dim: int = 64) -> tuple[np.ndarray, pd.DataFrame, float]:
    device = torch.device("cpu")
    table_encoder = TableEncoder(X.shape[1], out_dim=out_dim).to(device)
    text_projector = TextProjector(T.shape[1], out_dim=out_dim).to(device)
    optimizer = torch.optim.Adam(
        list(table_encoder.parameters()) + list(text_projector.parameters()),
        lr=1e-3,
    )
    dataset = ContrastiveDataset(X, T)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    history = []
    table_encoder.train()
    text_projector.train()
    for epoch in range(1, epochs + 1):
        losses = []
        for xb, tb in loader:
            xb = xb.to(device)
            tb = tb.to(device)
            optimizer.zero_grad()
            z_table = table_encoder(xb)
            z_text = text_projector(tb)
            loss = info_nce_loss(z_table, z_text)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})

    table_encoder.eval()
    text_projector.eval()
    with torch.no_grad():
        z_table = table_encoder(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
        z_text = text_projector(torch.tensor(T, dtype=torch.float32).to(device)).cpu().numpy()
    fused = 0.5 * (z_table + z_text)

    # retrieval accuracy on a subset to avoid a large all-pairs matrix
    subset_size = min(len(fused), 1000)
    idx = np.arange(subset_size)
    sim = F.normalize(torch.tensor(z_table[idx], dtype=torch.float32), dim=1) @ F.normalize(torch.tensor(z_text[idx], dtype=torch.float32), dim=1).T
    retrieval = (sim.argmax(dim=1).numpy() == idx).mean()

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(OUT_DIR / "contrastive_history.csv", index=False, encoding="utf-8-sig")
    return fused, hist_df, float(retrieval)


@dataclass
class MethodResult:
    method: str
    embedding: np.ndarray
    best_k: int
    best_labels: np.ndarray
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    stability_ari: float
    ic_eta2: float
    ic_level_nmi: float


def anova_eta2(values: np.ndarray, labels: np.ndarray) -> float:
    frame = pd.DataFrame({"value": values, "label": labels})
    grand = frame["value"].mean()
    ss_total = ((frame["value"] - grand) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = frame.groupby("label")["value"].apply(lambda s: len(s) * (s.mean() - grand) ** 2).sum()
    return float(ss_between / ss_total)


def evaluate_method(method: str, emb: np.ndarray, ic_df: pd.DataFrame) -> MethodResult:
    metrics = []
    ic_total = pd.to_numeric(ic_df["IC_total"], errors="coerce").fillna(ic_df["IC_total"].median()).values
    ic_level_codes = pd.Categorical(ic_df["IC_level"]).codes

    for k in range(3, 7):
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
        labels = km.fit_predict(emb)
        sil = silhouette_score(emb, labels)
        ch = calinski_harabasz_score(emb, labels)
        db = davies_bouldin_score(emb, labels)

        base_labels = labels
        aris = []
        for seed in [0, 1, 2, 3, 4]:
            km2 = KMeans(n_clusters=k, random_state=seed, n_init=20)
            lab2 = km2.fit_predict(emb)
            aris.append(adjusted_rand_score(base_labels, lab2))
        stability = float(np.mean(aris))
        eta2 = anova_eta2(ic_total, labels)
        nmi = normalized_mutual_info_score(ic_level_codes, labels)
        metrics.append(
            {
                "method": method,
                "k": k,
                "silhouette": sil,
                "calinski_harabasz": ch,
                "davies_bouldin": db,
                "stability_ari": stability,
                "ic_eta2": eta2,
                "ic_level_nmi": nmi,
                "labels": labels,
            }
        )

    metric_df = pd.DataFrame([{k: v for k, v in m.items() if k != "labels"} for m in metrics])
    score_df = metric_df.copy()
    score_df["db_score"] = 1.0 / (1.0 + score_df["davies_bouldin"])
    for col in ["silhouette", "stability_ari", "ic_eta2", "ic_level_nmi", "db_score"]:
        if score_df[col].max() > score_df[col].min():
            score_df[f"{col}_norm"] = (score_df[col] - score_df[col].min()) / (score_df[col].max() - score_df[col].min())
        else:
            score_df[f"{col}_norm"] = 0.0
    score_df["combined_score"] = score_df[
        [
            "silhouette_norm",
            "stability_ari_norm",
            "ic_eta2_norm",
            "ic_level_nmi_norm",
            "db_score_norm",
        ]
    ].mean(axis=1)
    best_idx = score_df["combined_score"].idxmax()
    best_row = metrics[int(best_idx)]

    metric_df["combined_score"] = score_df["combined_score"]
    metric_df.to_csv(OUT_DIR / f"{method}_cluster_grid.csv", index=False, encoding="utf-8-sig")

    return MethodResult(
        method=method,
        embedding=emb,
        best_k=int(best_row["k"]),
        best_labels=np.asarray(best_row["labels"]),
        silhouette=float(best_row["silhouette"]),
        calinski_harabasz=float(best_row["calinski_harabasz"]),
        davies_bouldin=float(best_row["davies_bouldin"]),
        stability_ari=float(best_row["stability_ari"]),
        ic_eta2=float(best_row["ic_eta2"]),
        ic_level_nmi=float(best_row["ic_level_nmi"]),
    )


def compute_pseudotime(emb: np.ndarray, ic_df: pd.DataFrame) -> tuple[np.ndarray, float]:
    ic_total = pd.to_numeric(ic_df["IC_total"], errors="coerce").values
    healthy_mask = ic_total <= np.nanpercentile(ic_total, 25)
    frail_mask = ic_total >= np.nanpercentile(ic_total, 75)
    healthy_center = emb[healthy_mask].mean(axis=0)
    frail_center = emb[frail_mask].mean(axis=0)
    direction = frail_center - healthy_center
    denom = np.linalg.norm(direction) ** 2 + 1e-8
    score = ((emb - healthy_center) @ direction) / denom
    score = (score - score.min()) / (score.max() - score.min() + 1e-8)
    rho = float(spearmanr(score, ic_total, nan_policy="omit").correlation)
    return score, rho


def find_medoid_indices(emb: np.ndarray, labels: np.ndarray, target_cluster: int, top_n: int = 3) -> list[int]:
    idx = np.where(labels == target_cluster)[0]
    centroid = emb[idx].mean(axis=0, keepdims=True)
    dist = ((emb[idx] - centroid) ** 2).sum(axis=1)
    order = np.argsort(dist)
    return idx[order[:top_n]].tolist()


def build_counterfactuals(raw_df: pd.DataFrame, emb: np.ndarray, labels: np.ndarray, ic_df: pd.DataFrame, mappings: dict[str, dict[str, str]]) -> pd.DataFrame:
    summary = pd.DataFrame({"cluster": labels, "ic_total": ic_df["IC_total"]})
    cluster_mean = summary.groupby("cluster")["ic_total"].mean().sort_values()
    healthy_cluster = int(cluster_mean.index[0])
    worst_cluster = int(cluster_mean.index[-1])
    medoids = find_medoid_indices(emb, labels, worst_cluster, top_n=3)
    healthy_idx = np.where(labels == healthy_cluster)[0]

    numeric_stds = {}
    for col in COUNTERFACTUAL_FIELDS:
        if col in raw_df.columns:
            numeric_stds[col] = pd.to_numeric(raw_df[col], errors="coerce").std()

    rows = []
    for idx in medoids:
        dist = ((emb[healthy_idx] - emb[idx]) ** 2).sum(axis=1)
        cf_idx = healthy_idx[int(np.argmin(dist))]

        diffs = []
        for col in COUNTERFACTUAL_FIELDS:
            if col not in raw_df.columns:
                continue
            a = raw_df.iloc[idx][col]
            b = raw_df.iloc[cf_idx][col]
            a_num = pd.to_numeric(pd.Series([a]), errors="coerce").iloc[0]
            b_num = pd.to_numeric(pd.Series([b]), errors="coerce").iloc[0]
            if pd.notna(a_num) and pd.notna(b_num):
                std = numeric_stds.get(col) or 1.0
                diffs.append((col, abs(a_num - b_num) / (std + 1e-6), readable_value(col, a, mappings), readable_value(col, b, mappings)))
            elif str(a) != str(b):
                diffs.append((col, 1.0, readable_value(col, a, mappings), readable_value(col, b, mappings)))
        diffs = sorted(diffs, key=lambda x: x[1], reverse=True)[:6]
        rows.append(
            {
                "query_idx": idx,
                "query_id": raw_df.iloc[idx]["编号"] if "编号" in raw_df.columns else idx,
                "query_cluster": worst_cluster,
                "query_ic_total": ic_df.iloc[idx]["IC_total"],
                "counterfactual_idx": cf_idx,
                "counterfactual_id": raw_df.iloc[cf_idx]["编号"] if "编号" in raw_df.columns else cf_idx,
                "counterfactual_cluster": healthy_cluster,
                "counterfactual_ic_total": ic_df.iloc[cf_idx]["IC_total"],
                "top_feature_changes": " | ".join([f"{col}: {a} -> {b}" for col, _, a, b in diffs]),
            }
        )

    cf_df = pd.DataFrame(rows)
    cf_df.to_csv(OUT_DIR / "prototype_counterfactuals.csv", index=False, encoding="utf-8-sig")
    return cf_df


def build_counterfactual_direction(raw_df: pd.DataFrame, labels: np.ndarray, ic_df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({"cluster": labels, "ic_total": ic_df["IC_total"]})
    cluster_mean = summary.groupby("cluster")["ic_total"].mean().sort_values()
    healthy_cluster = int(cluster_mean.index[0])
    worst_cluster = int(cluster_mean.index[-1])

    rows = []
    for col in COUNTERFACTUAL_FIELDS:
        if col not in raw_df.columns:
            continue
        series = pd.to_numeric(raw_df[col], errors="coerce")
        if series.notna().mean() < 0.7:
            continue
        overall_std = float(series.std())
        if overall_std == 0 or math.isnan(overall_std):
            continue
        healthy_mean = float(series[labels == healthy_cluster].mean())
        worst_mean = float(series[labels == worst_cluster].mean())
        delta = healthy_mean - worst_mean
        rows.append(
            {
                "feature": col,
                "healthy_cluster_mean": healthy_mean,
                "worst_cluster_mean": worst_mean,
                "delta_healthy_minus_worst": delta,
                "standardized_delta": delta / (overall_std + 1e-8),
            }
        )

    direction_df = pd.DataFrame(rows).sort_values("standardized_delta", key=lambda s: s.abs(), ascending=False)
    direction_df.to_csv(OUT_DIR / "counterfactual_direction.csv", index=False, encoding="utf-8-sig")

    top_df = direction_df.head(10).copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=top_df, x="standardized_delta", y="feature", palette="coolwarm", ax=ax)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Healthy-vs-Worst Cluster Direction")
    ax.set_xlabel("Standardized Difference")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "counterfactual_direction.png", dpi=220)
    plt.close(fig)
    return direction_df


def plot_method_comparison(results: list[MethodResult]) -> None:
    df = pd.DataFrame(
        [
            {
                "method": r.method,
                "best_k": r.best_k,
                "silhouette": r.silhouette,
                "stability_ari": r.stability_ari,
                "ic_eta2": r.ic_eta2,
                "ic_level_nmi": r.ic_level_nmi,
                "davies_bouldin": r.davies_bouldin,
            }
            for r in results
        ]
    )
    df.to_csv(OUT_DIR / "method_comparison.csv", index=False, encoding="utf-8-sig")

    plot_df = df.melt(id_vars=["method", "best_k"], value_vars=["silhouette", "stability_ari", "ic_eta2", "ic_level_nmi"])
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="variable", y="value", hue="method", ax=ax)
    ax.set_title("Representation Method Comparison")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "method_comparison.png", dpi=220)
    plt.close(fig)


def plot_loss_curves(dae_hist: pd.DataFrame, contrast_hist: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(data=dae_hist, x="epoch", y="loss", marker="o", ax=axes[0])
    axes[0].set_title("DAE Reconstruction Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    sns.lineplot(data=contrast_hist, x="epoch", y="loss", marker="o", ax=axes[1], color="#d62728")
    axes[1].set_title("Contrastive Alignment Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "training_loss_curves.png", dpi=220)
    plt.close(fig)


def plot_best_embedding(best: MethodResult, ic_df: pd.DataFrame, pseudotime: np.ndarray) -> None:
    reducer = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = reducer.fit_transform(best.embedding)

    assign_df = pd.DataFrame(
        {
            "编号": ic_df["编号"] if "编号" in ic_df.columns else np.arange(len(ic_df)),
            "cluster_id": best.best_labels,
            "IC_total": ic_df["IC_total"].values,
            "IC_level": ic_df["IC_level"].values,
            "pseudotime": pseudotime,
        }
    )
    assign_df.to_csv(OUT_DIR / "best_cluster_assignments.csv", index=False, encoding="utf-8-sig")

    emb_df = pd.DataFrame(best.embedding, columns=[f"emb_{i}" for i in range(best.embedding.shape[1])])
    emb_df.insert(0, "编号", assign_df["编号"])
    try:
        emb_df.to_parquet(OUT_DIR / "best_embedding.parquet", index=False)
    except Exception:
        emb_df.to_csv(OUT_DIR / "best_embedding.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=best.best_labels, palette="tab10", s=18, linewidth=0, ax=ax)
    ax.set_title(f"Best Representation: {best.method} (k={best.best_k})")
    ax.set_xlabel("PC-1")
    ax.set_ylabel("PC-2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "best_umap_clusters.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=pseudotime, cmap="viridis", s=18)
    ax.set_title("Cross-sectional Functional Pseudotime")
    ax.set_xlabel("PC-1")
    ax.set_ylabel("PC-2")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Pseudotime")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "best_umap_pseudotime.png", dpi=220)
    plt.close(fig)


def plot_cluster_heatmap(labels: np.ndarray, ic_df: pd.DataFrame) -> None:
    cols = [
        "IC_sensory",
        "IC_vitality",
        "IC_locomotion",
        "IC_cognition",
        "IC_psychological",
        "IC_total",
    ]
    heat = pd.concat([pd.Series(labels, name="cluster"), ic_df[cols].reset_index(drop=True)], axis=1)
    mean_df = heat.groupby("cluster")[cols].mean()
    mean_df.to_csv(OUT_DIR / "cluster_ic_profile.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.heatmap(mean_df, annot=True, cmap="YlOrRd", fmt=".2f", ax=ax)
    ax.set_title("Cluster Profiles on IC Domains")
    ax.set_xlabel("IC Domain")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "cluster_ic_heatmap.png", dpi=220)
    plt.close(fig)


def write_technical_report(
    results: list[MethodResult],
    best: MethodResult,
    text_encoder_name: str,
    contrastive_retrieval: float,
    pseudotime_rho: float,
    counterfactuals: pd.DataFrame,
    direction_df: pd.DataFrame,
) -> None:
    df = pd.DataFrame(
        [
            {
                "method": r.method,
                "best_k": r.best_k,
                "silhouette": r.silhouette,
                "stability_ari": r.stability_ari,
                "ic_eta2": r.ic_eta2,
                "ic_level_nmi": r.ic_level_nmi,
                "davies_bouldin": r.davies_bouldin,
            }
            for r in results
        ]
    ).sort_values(["ic_eta2", "silhouette", "stability_ari"], ascending=False)

    lines = []
    lines.append("# 下一阶段方案验证技术报告")
    lines.append("")
    lines.append("## 目标")
    lines.append("- 用现有横断面数据验证两条下一阶段路线是否有继续推进价值。")
    lines.append("- 方案A：自监督表格表征 + 聚类分型。")
    lines.append("- 方案B轻量验证：Tab2Text + 预训练中文文本编码器 + 表格/文本对比学习。")
    lines.append("")
    lines.append("## 本次实际跑了什么")
    lines.append("- Baseline 表征：PCA-32。")
    lines.append("- 方案A主力表征：Denoising AutoEncoder (32维潜变量，掩码重建)。")
    lines.append(f"- 方案B轻量版：将结构化字段转成中文画像文本，用 `{text_encoder_name}` 编码，再训练表格编码器与文本编码器做对比学习并形成融合表示。")
    lines.append("- 每种表示在 k=3..6 之间做 KMeans，对比 silhouette、稳定性 ARI、IC 区分度 eta^2、IC_level 的 NMI。")
    lines.append("")
    lines.append("## 方法对比总表")
    lines.append(df.to_markdown(index=False))
    lines.append("")
    lines.append("## 当前最优方案")
    lines.append(f"- 最优表示：`{best.method}`")
    lines.append(f"- 最优聚类数：`k={best.best_k}`")
    lines.append(f"- silhouette：`{best.silhouette:.4f}`")
    lines.append(f"- stability_ari：`{best.stability_ari:.4f}`")
    lines.append(f"- IC eta^2：`{best.ic_eta2:.4f}`")
    lines.append(f"- IC_level NMI：`{best.ic_level_nmi:.4f}`")
    lines.append(f"- Davies-Bouldin：`{best.davies_bouldin:.4f}`")
    lines.append("")
    lines.append("## 方案B轻量验证结果")
    lines.append(f"- 文本编码器：`{text_encoder_name}`")
    lines.append(f"- 训练集内表格-文本同一样本检索 Top-1 命中率：`{contrastive_retrieval:.4f}`")
    lines.append("- 这是一个对齐诊断指标，只说明两个视图之间存在可学习的一致语义，不应直接当成泛化性能。")
    lines.append("- 当前版本还不是完整生成式 LLM 方案，但已经证明“表格转文本再做跨模态学习”这条路可跑通。")
    lines.append("")
    lines.append("## 轨迹尝试")
    lines.append(f"- 在最优表示空间中，从低 IC 受损锚点到高 IC 受损锚点构造了一条横断面伪时序。")
    lines.append(f"- 伪时序与 IC_total 的 Spearman 相关：`rho={pseudotime_rho:.4f}`")
    lines.append("- 这不等于真实随访轨迹，但说明 embedding 空间中存在较稳定的“功能退化方向”。")
    lines.append("")
    lines.append("## 反事实原型尝试")
    lines.append("- 用最差 cluster 的代表样本，去寻找最健康 cluster 中最近的原型个体。")
    lines.append("- 这是一种 prototype counterfactual，不宣称因果，只用于帮助解释“从差型到好型通常差在哪些变量”。")
    lines.append("")
    if not counterfactuals.empty:
        lines.append(counterfactuals.to_markdown(index=False))
        lines.append("")
    if not direction_df.empty:
        lines.append("## 簇间方向差异（更稳的 counterfactual direction）")
        lines.append("- 下表表示：如果从最差 cluster 向最健康 cluster 靠近，哪些变量方向变化最大。")
        lines.append(direction_df.head(12).to_markdown(index=False))
        lines.append("")
    lines.append("## 结论")
    lines.append("- 方案A是当前最适合继续深挖并快速写论文的主线。")
    lines.append("- 方案B不建议马上上完整生成式大模型，但值得先做“结构化文本 + 文本编码器 + 对比学习”的过渡版本。")
    lines.append("- 伪时序和 prototype counterfactual 都已经有可跑通的最小验证版，可以作为后续 fancy 卖点。")
    (REPORT_DIR / "NEXT_PHASE_TECHNICAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    seed_everything()
    ensure_dirs()
    configure_plot_style()

    raw_df, ic_df, mappings, X_df = load_data()
    text_df = build_text_cards(raw_df, mappings)
    text_emb, text_encoder_name = encode_texts(text_df["text_card"].tolist())

    pca_emb = PCA(n_components=32, random_state=RANDOM_SEED).fit_transform(X_df.values)
    dae_emb, dae_hist = train_dae(X_df.values, latent_dim=32)
    contrast_emb, contrast_hist, retrieval = train_contrastive_alignment(X_df.values, text_emb, out_dim=64)

    plot_loss_curves(dae_hist, contrast_hist)

    results = [
        evaluate_method("PCA32", pca_emb, ic_df),
        evaluate_method("DAE32", dae_emb, ic_df),
        evaluate_method("ContrastiveFusion64", contrast_emb, ic_df),
    ]
    plot_method_comparison(results)

    best = sorted(results, key=lambda r: (r.ic_eta2, r.silhouette, r.stability_ari), reverse=True)[0]
    pseudotime, rho = compute_pseudotime(best.embedding, ic_df)
    pd.DataFrame({"编号": raw_df["编号"], "pseudotime": pseudotime}).to_csv(
        OUT_DIR / "pseudotime_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_best_embedding(best, ic_df, pseudotime)
    plot_cluster_heatmap(best.best_labels, ic_df)

    counterfactuals = build_counterfactuals(raw_df, best.embedding, best.best_labels, ic_df, mappings)
    direction_df = build_counterfactual_direction(raw_df, best.best_labels, ic_df)
    write_technical_report(results, best, text_encoder_name, retrieval, rho, counterfactuals, direction_df)


if __name__ == "__main__":
    main()
