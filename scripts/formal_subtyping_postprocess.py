from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "formal_subtyping"
FIG_DIR = ROOT / "figures" / "formal_subtyping"
REPORT_DIR = ROOT / "reports" / "formal_subtyping"
RANDOM_SEED = 2026

METHOD_FILES = {
    "BERTText": "bert_embedding_128.npy",
    "FTTransformer": "ft_embedding_128.npy",
    "PCA": "pca_embedding_128.npy",
}

IC_ALIAS = {
    "IC_cognition_cont_100": "Cognition",
    "IC_psychological_cont_100": "Psychological",
    "IC_vitality_cont_100": "Vitality",
    "IC_locomotion_cont_100": "Locomotion",
    "IC_sensory_cont_100": "Sensory",
    "IC_total_cont_100": "Overall IC",
}

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

CORE_SUMMARY_ALIAS = {
    "年龄": "Age",
    "患有慢性病数量": "Chronic disease count",
    "服用药物数量": "Medication count",
    "过去一年住院次数": "Hospitalizations (1y)",
    "过去一年急诊次数": "ED visits (1y)",
    "查尔森合并症得分": "Charlson score",
    "健康自评分数": "Self-rated health score",
    "认知-总分": "Cognition raw score",
    "心理-总分": "Psychological raw score",
    "活力-总分": "Vitality raw score",
    "运动-总分": "Locomotion raw score",
}

RECOMMENDED_MAIN_METHOD = "FTTransformer"


@dataclass
class MethodSummary:
    method: str
    k: int
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    stability_ari: float
    external_separation: float
    min_cluster_ratio: float
    max_cluster_ratio: float
    cluster_sizes: List[int]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 220
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["axes.unicode_minus"] = False


def load_raw() -> pd.DataFrame:
    xlsx = next(DATA_DIR.glob("*.xlsx"))
    return pd.read_excel(xlsx)


def load_ic() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / "IC_continuous_scores.csv", encoding="utf-8-sig")


def get_best_k_map() -> Dict[str, int]:
    df = pd.read_csv(OUTPUT_DIR / "best_method_metrics.csv", encoding="utf-8-sig")
    return {row["method"]: int(row["k"]) for _, row in df.iterrows()}


def load_embeddings() -> Dict[str, np.ndarray]:
    return {method: np.load(OUTPUT_DIR / file_name) for method, file_name in METHOD_FILES.items()}


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


def reorder_labels_by_ic(labels: np.ndarray, ic_total: pd.Series) -> np.ndarray:
    tmp = pd.DataFrame({"cluster": labels, "ic_total": ic_total})
    order = (
        tmp.groupby("cluster")["ic_total"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[x] for x in labels], dtype=int)


def compute_method_summary(method: str, emb: np.ndarray, k: int, raw_df: pd.DataFrame, ic_df: pd.DataFrame) -> tuple[MethodSummary, np.ndarray]:
    labels = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20).fit_predict(emb)
    labels = reorder_labels_by_ic(labels, ic_df["IC_total_cont_100"])

    silhouette = float(silhouette_score(emb, labels))
    calinski = float(calinski_harabasz_score(emb, labels))
    db = float(davies_bouldin_score(emb, labels))

    aris = []
    for seed in [0, 1, 2, 3, 4]:
        seed_labels = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(emb)
        seed_labels = reorder_labels_by_ic(seed_labels, ic_df["IC_total_cont_100"])
        aris.append(adjusted_rand_score(labels, seed_labels))

    ext_cols = [c for c in EXTERNAL_ALIAS if c in raw_df.columns]
    external_separation = external_effect_score(labels, raw_df, ext_cols)
    counts = pd.Series(labels).value_counts().sort_index()

    summary = MethodSummary(
        method=method,
        k=k,
        silhouette=silhouette,
        calinski_harabasz=calinski,
        davies_bouldin=db,
        stability_ari=float(np.mean(aris)),
        external_separation=external_separation,
        min_cluster_ratio=float(counts.min() / len(labels)),
        max_cluster_ratio=float(counts.max() / len(labels)),
        cluster_sizes=counts.tolist(),
    )
    return summary, labels


def english_external_profile(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns=EXTERNAL_ALIAS).copy()
    cols = ["cluster_id"] + [EXTERNAL_ALIAS[c] for c in EXTERNAL_ALIAS if c in df.columns]
    return out[[c for c in cols if c in out.columns]]


def english_ic_profile(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns=IC_ALIAS).copy()
    cols = ["cluster_id", "Cognition", "Psychological", "Vitality", "Locomotion", "Sensory", "Overall IC"]
    return out[[c for c in cols if c in out.columns]]


def save_profiles(method: str, labels: np.ndarray, emb: np.ndarray, raw_df: pd.DataFrame, ic_df: pd.DataFrame) -> None:
    id_col = "编号" if "编号" in raw_df.columns else raw_df.columns[1]
    assign = pd.DataFrame({"sample_id": raw_df[id_col], "cluster_id": labels})
    assign.to_csv(OUTPUT_DIR / f"{method}_cluster_assignments_reportable.csv", index=False, encoding="utf-8-sig")

    ic_cols = list(IC_ALIAS.keys())
    ic_profile = ic_df[ic_cols].copy()
    ic_profile["cluster_id"] = labels
    ic_mean = ic_profile.groupby("cluster_id")[ic_cols].mean().reset_index()
    ic_mean_en = english_ic_profile(ic_mean)
    ic_mean_en.to_csv(OUTPUT_DIR / f"{method}_ic_profile_en.csv", index=False, encoding="utf-8-sig")

    ext_cols = [c for c in EXTERNAL_ALIAS if c in raw_df.columns]
    ext = raw_df[ext_cols].apply(pd.to_numeric, errors="coerce")
    ext["cluster_id"] = labels
    ext_mean = ext.groupby("cluster_id").mean().reset_index()
    ext_mean_en = english_external_profile(ext_mean)
    ext_mean_en.to_csv(OUTPUT_DIR / f"{method}_external_profile_en.csv", index=False, encoding="utf-8-sig")

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
        stats_rows.append(
            {
                "feature_raw": col,
                "feature_en": EXTERNAL_ALIAS.get(col, col),
                "p_value": float(p),
                "max_mean_diff": float(np.nanmax(means) - np.nanmin(means)),
            }
        )
    pd.DataFrame(stats_rows).sort_values(["p_value", "max_mean_diff"]).to_csv(
        OUTPUT_DIR / f"{method}_external_stats_en.csv", index=False, encoding="utf-8-sig"
    )

    if emb.shape[1] >= 2:
        coords = TSNE(n_components=2, random_state=RANDOM_SEED, init="pca", learning_rate="auto", perplexity=35).fit_transform(emb)
    else:
        coords = emb

    fig, ax = plt.subplots(figsize=(6.2, 5.1))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, palette="tab10", s=14, ax=ax, legend="full")
    ax.set_title(f"2D Projection of {method} Embedding")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"{method}_embedding_projection_en.png")
    plt.close(fig)

    ic_plot = ic_mean_en.set_index("cluster_id")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.heatmap(ic_plot, annot=True, cmap="YlGnBu", fmt=".1f", ax=ax)
    ax.set_title(f"Continuous IC Profile by Cluster ({method})")
    ax.set_xlabel("IC Domain")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"{method}_ic_heatmap_en.png")
    plt.close(fig)

    radar_cols = ["Cognition", "Psychological", "Vitality", "Locomotion", "Sensory"]
    angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(6.1, 6.1))
    ax = plt.subplot(111, polar=True)
    for cluster_id, row in ic_plot[radar_cols].iterrows():
        values = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, values, linewidth=2, label=f"Cluster {cluster_id}")
        ax.fill(angles, values, alpha=0.10)
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_cols)
    ax.set_ylim(0, 100)
    ax.set_title(f"Radar Plot of IC Domains ({method})")
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.15))
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"{method}_radar_en.png")
    plt.close(fig)

    ext_keep = [
        "ADL total",
        "IADL total",
        "Sarcopenia score",
        "Fried frailty score",
        "Frailty screen score",
        "Fall risk score",
        "Resilience score",
        "Social function score",
    ]
    ext_plot = ext_mean_en.set_index("cluster_id")
    ext_plot = ext_plot[[c for c in ext_keep if c in ext_plot.columns]]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    sns.heatmap(ext_plot, annot=True, cmap="OrRd", fmt=".1f", ax=ax)
    ax.set_title(f"External Clinical Profile by Cluster ({method})")
    ax.set_xlabel("External Variable")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"{method}_external_heatmap_en.png")
    plt.close(fig)


def find_column(df: pd.DataFrame, col_name: str) -> str | None:
    return col_name if col_name in df.columns else None


def cluster_keywords(raw_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    cols = [c for c in CORE_SUMMARY_ALIAS if c in raw_df.columns]
    if not cols:
        return pd.DataFrame()
    out = raw_df[cols].copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    out["cluster_id"] = labels
    grouped = out.groupby("cluster_id").mean().reset_index()
    grouped = grouped.rename(columns=CORE_SUMMARY_ALIAS)
    return grouped


def write_method_comparison_figure(summary_df: pd.DataFrame) -> None:
    plot_df = summary_df.melt(
        id_vars=["method"],
        value_vars=["silhouette", "stability_ari", "external_separation", "min_cluster_ratio"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.barplot(data=plot_df, x="metric", y="value", hue="method", ax=ax)
    ax.set_title("Formal Subtyping Method Comparison")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "method_comparison_reportable_en.png")
    plt.close(fig)


def build_cluster_interpretation(method: str, ic_profile: pd.DataFrame, ext_profile: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    for _, row in ic_profile.sort_values("cluster_id").iterrows():
        cid = int(row["cluster_id"])
        overall = row["Overall IC"]
        loco = row["Locomotion"]
        sensory = row["Sensory"]
        cog = row["Cognition"]
        psych = row["Psychological"]
        ext = ext_profile.loc[ext_profile["cluster_id"] == cid].iloc[0]

        note_parts = [f"Cluster {cid}：Overall IC={overall:.1f}。"]
        if overall < 65:
            note_parts.append("这是最受损的整体低功能簇。")
        elif overall > 80:
            note_parts.append("这是整体功能最高的一组。")
        else:
            note_parts.append("这是中间型/过渡型簇。")

        domain_flags = []
        if loco < 45:
            domain_flags.append("运动能力明显偏弱")
        elif loco > 70:
            domain_flags.append("运动能力保持较好")
        if sensory < 50:
            domain_flags.append("感官维度较差")
        elif sensory > 85:
            domain_flags.append("感官维度保持较好")
        if cog < 75:
            domain_flags.append("认知维度偏弱")
        if psych < 75:
            domain_flags.append("心理维度偏弱")
        if domain_flags:
            note_parts.append("主要特征：" + "，".join(domain_flags) + "。")

        extra_parts = []
        if "Sarcopenia score" in ext.index:
            extra_parts.append(f"Sarcopenia={ext['Sarcopenia score']:.2f}")
        if "Fried frailty score" in ext.index:
            extra_parts.append(f"Fried={ext['Fried frailty score']:.2f}")
        if "ADL total" in ext.index:
            extra_parts.append(f"ADL={ext['ADL total']:.1f}")
        if "IADL total" in ext.index:
            extra_parts.append(f"IADL={ext['IADL total']:.1f}")
        if "Fall risk score" in ext.index:
            extra_parts.append(f"Fall risk={ext['Fall risk score']:.1f}")
        if extra_parts:
            note_parts.append("外部临床画像：" + "，".join(extra_parts) + "。")

        lines.append(" ".join(note_parts))

    if method == "FTTransformer":
        lines.append("FTTransformer 的分型更像一条‘整体功能下降 + 感官分化’的谱系，因此更适合做主线深度学习分型。")
    elif method == "BERTText":
        lines.append("BERTText 的簇几何边界最清楚，但外部临床量表分离度偏低，说明纯文本语义目前更像‘语义整理器’，还不是最强的临床分型器。")
    else:
        lines.append("PCA 是非常强的基线，说明这批数据本身已经带有较强的低秩结构。")
    return lines


def write_reports(summary_df: pd.DataFrame, method_tables: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    comparison = summary_df.copy()
    comparison["cluster_sizes"] = comparison["cluster_sizes"].apply(lambda x: ", ".join(str(v) for v in x))
    comparison.to_csv(OUTPUT_DIR / "reportable_method_summary.csv", index=False, encoding="utf-8-sig")

    recommended_row = summary_df.loc[summary_df["method"] == RECOMMENDED_MAIN_METHOD].iloc[0]
    best_external_row = summary_df.sort_values(["external_separation", "silhouette"], ascending=False).iloc[0]
    best_internal_row = summary_df.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).iloc[0]

    lines: List[str] = []
    lines.append("# 正式分型后处理技术报告")
    lines.append("")
    lines.append("## 1. 这一步在做什么")
    lines.append("- 这份报告不是重新训练模型，而是在已有正式分型结果上做“可汇报版本”的后处理。")
    lines.append("- 核心目标有三个：")
    lines.append("  1. 重新用固定随机种子生成可复现的聚类标签，避免旧文件里出现标签不一致的问题。")
    lines.append("  2. 把所有正式汇报图中的文字统一改成英文，彻底绕开中文字体显示问题。")
    lines.append("  3. 生成一份中文技术说明，明确三条路线各自擅长什么、短板是什么、下一步该怎么推进。")
    lines.append("")
    lines.append("## 2. 本次后处理使用的输入")
    lines.append("- 连续化 IC 结果：`outputs/formal_subtyping/IC_continuous_scores.csv`")
    lines.append("- 三条 128 维表征：`bert_embedding_128.npy`、`ft_embedding_128.npy`、`pca_embedding_128.npy`")
    lines.append("- 最优 k 值来自已有正式实验：BERTText=4，FTTransformer=5，PCA=4")
    lines.append("- 后处理统一用 `random_state=2026, n_init=20` 重新做 KMeans，并按 Overall IC 从低到高重排 cluster_id。")
    lines.append("")
    lines.append("## 3. 连续化 IC 的含义")
    lines.append("- 认知：使用 `认知-总分 / 30`，近似 MMSE 连续化。")
    lines.append("- 心理：使用 `1 - 心理-总分 / 15`，再把焦虑抑郁诊断作为 15% 惩罚项。")
    lines.append("- 活力：综合 `活力-总分`、营养描述、BMI 偏离 22 的程度、小腿围。")
    lines.append("- 运动：综合 `运动-总分`、运动完成情况、步态异常、4 米步速、握力。")
    lines.append("- 感官：综合听力/视力障碍、筛查结果、是否影响日常。")
    lines.append("- 最终每个维度都是 0–100 分，总分是五域平均。这样做的价值是：不再把老人强行压成“受损/未受损”两类，而是保留程度信息。")
    lines.append("")
    lines.append("## 4. 三条分型路线怎么理解")
    lines.append("- `BERTText`：先把样本字段用 `编码.docx` 映射成更可读的医学文本，再用 `bert-base-chinese` 编码，最后压到 128 维。")
    lines.append("- `FTTransformer`：直接对表格做 Transformer 风格的自监督重建，输出 128 维表示。")
    lines.append("- `PCA`：对同一套核心特征做统一预处理后直接降到 128 维，作为强基线。")
    lines.append("")
    lines.append("## 5. 统一比较结果")
    lines.append(comparison[[
        "method",
        "k",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "stability_ari",
        "external_separation",
        "min_cluster_ratio",
        "max_cluster_ratio",
        "cluster_sizes",
    ]].to_markdown(index=False))
    lines.append("")
    lines.append("解释：")
    lines.append(f"- 纯几何聚类质量最强的是 `{best_internal_row['method']}`，它的 silhouette={best_internal_row['silhouette']:.4f}。")
    lines.append(f"- 外部临床区分度最强的是 `{best_external_row['method']}`，external_separation={best_external_row['external_separation']:.4f}。")
    lines.append(f"- 作为下一阶段主推深度学习路线，我推荐 `{RECOMMENDED_MAIN_METHOD}`。理由不是“它绝对第一”，而是它在外部临床分离度上已经接近最强基线，同时保留了明确的深度模型增量空间。")
    lines.append("")
    lines.append("## 6. 结果怎么解读")
    lines.append("- `BERTText`：簇边界最清楚，说明‘把硬编码字段翻译成自然语言再编码’这条路是有效的；但它对 ADL/IADL/肌少症/衰弱等外部量表的分离度偏低，说明当前文本路线更像语义整理，而不是最强临床分型主线。")
    lines.append("- `FTTransformer`：silhouette 不高，但 external_separation 很强，而且 5 个簇都不是小簇，比较适合讲“临床异质性分层”。这条路线最适合作为你们后面冲 A 类论文的深度学习主线。")
    lines.append("- `PCA`：基线依然很强，说明这批数据本身结构性很强。这个结果很重要，因为它逼着我们后面的 fancy 方法必须拿出真正的增益，而不是只换一个更复杂的模型名字。")
    lines.append("")

    for method in ["BERTText", "FTTransformer", "PCA"]:
        lines.append(f"## 7. {method} 的分型画像")
        ic_profile = method_tables[method]["ic"]
        ext_profile = method_tables[method]["ext"]
        lines.extend([f"- {item}" for item in build_cluster_interpretation(method, ic_profile, ext_profile)])
        lines.append("")

    lines.append("## 8. 为什么主推 FTTransformer，而不是直接拿 PCA 发")
    lines.append("- 因为现在项目的目标不是只做一个 baseline 分层，而是要走一条“深度学习/大模型辅助”的论文路线。")
    lines.append("- PCA 必须保留，并且要如实承认它很强；这会让文章更可信。")
    lines.append("- 但真正能继续往前扩展成 fancy 方法的，是 FTTransformer 这条线：")
    lines.append("  1. 它可以自然接上缺失掩码预训练。")
    lines.append("  2. 它可以接入 group-aware token、region token、disease token 等语义结构。")
    lines.append("  3. 它可以和文本/BERT 表征做跨模态对比学习，而不是停留在单一表格模型。")
    lines.append("")
    lines.append("## 9. 下一步最值得做的事")
    lines.append("- 先把 FTTransformer 作为主线模型，做 group-aware 输入和 masked modeling。")
    lines.append("- 再把 BERTText 从‘单独聚类’升级成‘文本-表格双塔对比学习’，而不是让文本路线单打独斗。")
    lines.append("- 聚类评价不能只看 silhouette，必须同时汇报 external_separation、cluster size balance、以及关键外部量表的组间检验。")
    lines.append("- 论文写法上要坚持：深度模型不是为了炫，而是为了更好地恢复被硬编码压扁的语义结构。")
    lines.append("")
    (REPORT_DIR / "TECHNICAL_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8-sig")

    fig_lines: List[str] = []
    fig_lines.append("# 图像说明（如果图片软件再次出现字体问题，可以直接看这里）")
    fig_lines.append("")
    fig_lines.append("## method_comparison_reportable_en.png")
    fig_lines.append("- 比较三条路线在四个维度上的表现：silhouette、stability_ari、external_separation、min_cluster_ratio。")
    fig_lines.append("- `silhouette` 越大越说明簇内更紧、簇间更分。")
    fig_lines.append("- `stability_ari` 越大越说明换随机种子后结果更稳定。")
    fig_lines.append("- `external_separation` 越大越说明聚类能把 ADL/IADL/肌少症/衰弱/跌倒等外部临床变量分开。")
    fig_lines.append("- `min_cluster_ratio` 越大越说明没有特别小、特别难解释的边缘簇。")
    fig_lines.append("")
    for method in ["BERTText", "FTTransformer", "PCA"]:
        fig_lines.append(f"## {method}_embedding_projection_en.png")
        fig_lines.append(f"- 看 {method} 在二维空间里的分布。这里只看几何形状，不代表临床解释一定最强。")
        fig_lines.append(f"## {method}_ic_heatmap_en.png")
        fig_lines.append("- 看每个 cluster 在五个连续 IC 域和总分上的均值。")
        fig_lines.append(f"## {method}_radar_en.png")
        fig_lines.append("- 和热图是同一批信息，只是换成雷达图，更适合口头汇报。")
        fig_lines.append(f"## {method}_external_heatmap_en.png")
        fig_lines.append("- 用外部临床量表验证聚类：ADL、IADL、肌少症、Fried 衰弱、跌倒风险、弹性、社会功能。")
        fig_lines.append("- 如果某个方法的外部热图层次感更强，说明它更接近“医学上有意义的分型”。")
        fig_lines.append("")
    (FIG_DIR / "FIGURE_NOTES_CN.md").write_text("\n".join(fig_lines), encoding="utf-8-sig")

    file_lines = []
    file_lines.append("# 文件结构说明")
    file_lines.append("")
    file_lines.append("## outputs/formal_subtyping")
    file_lines.append("- `reportable_method_summary.csv`：后处理后正式汇报使用的方法比较表。")
    file_lines.append("- `*_cluster_assignments_reportable.csv`：重新生成且可复现的 cluster 结果。")
    file_lines.append("- `*_ic_profile_en.csv`：英文列名的 IC 连续画像表。")
    file_lines.append("- `*_external_profile_en.csv`：英文列名的外部临床画像表。")
    file_lines.append("- `*_external_stats_en.csv`：各外部量表的 Kruskal-Wallis 检验结果。")
    file_lines.append("")
    file_lines.append("## figures/formal_subtyping")
    file_lines.append("- `*_embedding_projection_en.png`：二维投影图。")
    file_lines.append("- `*_ic_heatmap_en.png`：IC 五域及总分热图。")
    file_lines.append("- `*_radar_en.png`：IC 雷达图。")
    file_lines.append("- `*_external_heatmap_en.png`：外部临床验证热图。")
    file_lines.append("- `method_comparison_reportable_en.png`：方法总体比较图。")
    file_lines.append("- `FIGURE_NOTES_CN.md`：图片读图说明，作为字体异常时的文字兜底。")
    file_lines.append("")
    file_lines.append("## reports/formal_subtyping")
    file_lines.append("- `TECHNICAL_REPORT_CN.md`：正式中文技术说明。")
    file_lines.append("- `FILE_STRUCTURE_CN.md`：本文件。")
    (REPORT_DIR / "FILE_STRUCTURE_CN.md").write_text("\n".join(file_lines), encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    configure_plot_style()

    raw_df = load_raw().reset_index(drop=True)
    ic_df = load_ic().reset_index(drop=True)
    best_k_map = get_best_k_map()
    embeddings = load_embeddings()

    summaries: List[MethodSummary] = []
    method_tables: Dict[str, Dict[str, pd.DataFrame]] = {}

    for method in ["BERTText", "FTTransformer", "PCA"]:
        summary, labels = compute_method_summary(method, embeddings[method], best_k_map[method], raw_df, ic_df)
        summaries.append(summary)
        save_profiles(method, labels, embeddings[method], raw_df, ic_df)

        ic_profile = pd.read_csv(OUTPUT_DIR / f"{method}_ic_profile_en.csv", encoding="utf-8-sig")
        ext_profile = pd.read_csv(OUTPUT_DIR / f"{method}_external_profile_en.csv", encoding="utf-8-sig")
        method_tables[method] = {
            "ic": ic_profile,
            "ext": ext_profile,
        }

    summary_df = pd.DataFrame([s.__dict__ for s in summaries])
    summary_df = summary_df.sort_values(["external_separation", "silhouette"], ascending=False).reset_index(drop=True)
    write_method_comparison_figure(summary_df)
    write_reports(summary_df, method_tables)


if __name__ == "__main__":
    main()
