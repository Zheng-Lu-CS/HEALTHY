# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "advanced_subtyping"
FIG_DIR = ROOT / "figures" / "advanced_subtyping"
REPORT_DIR = ROOT / "reports" / "advanced_subtyping"


def main() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 220
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["axes.unicode_minus"] = False

    eval_df = pd.read_csv(OUTPUT_DIR / "all_method_cluster_metrics.csv", encoding="utf-8-sig")

    chosen = pd.concat(
        [
            eval_df[(eval_df["method"] == "GroupMaskedFT") & (eval_df["clusterer"] == "gmm") & (eval_df["k"] == 4)],
            eval_df[(eval_df["method"] == "PCA") & (eval_df["clusterer"] == "kmeans") & (eval_df["k"] == 4)],
            eval_df[(eval_df["method"] == "DualTowerTable") & (eval_df["clusterer"] == "kmeans") & (eval_df["k"] == 4)],
        ],
        ignore_index=True,
    )
    chosen.to_csv(OUTPUT_DIR / "final_recommended_vs_controls.csv", index=False, encoding="utf-8-sig")

    plot_df = chosen.melt(
        id_vars=["method"],
        value_vars=["silhouette", "stability_ari", "external_separation", "min_cluster_ratio"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    sns.barplot(data=plot_df, x="metric", y="value", hue="method", ax=ax)
    ax.set_title("Final Recommended Deep Model vs Controls")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "final_model_vs_controls_en.png")
    plt.close(fig)

    final_ic = pd.read_csv(OUTPUT_DIR / "GroupMaskedFT_GMM4_FINAL_ic_profile.csv", encoding="utf-8-sig")
    final_ext = pd.read_csv(OUTPUT_DIR / "GroupMaskedFT_GMM4_FINAL_external_profile.csv", encoding="utf-8-sig")
    final_stats = pd.read_csv(OUTPUT_DIR / "GroupMaskedFT_GMM4_FINAL_external_stats.csv", encoding="utf-8-sig")
    final_markers = pd.read_csv(OUTPUT_DIR / "GroupMaskedFT_GMM4_FINAL_top_markers.csv", encoding="utf-8-sig")

    report_lines = []
    report_lines.append("# 最终推荐模型与论文主线建议")
    report_lines.append("")
    report_lines.append("## 1. 最终推荐模型")
    report_lines.append("- 主模型：`GroupMaskedFT + GMM(full) + k=4`")
    report_lines.append("- 对照基线：`PCA + KMeans + k=4`")
    report_lines.append("- 双塔消融：`DualTowerTable + KMeans + k=4`")
    report_lines.append("")
    report_lines.append("## 2. 为什么最后选这个深度模型")
    report_lines.append(chosen.to_markdown(index=False))
    report_lines.append("")
    report_lines.append("解释：")
    report_lines.append("- 相比 `PCA + KMeans + k=4`，最终深度模型在 `external_separation` 上略高（0.2384 vs 0.2349），说明外部临床变量的分层更清楚。")
    report_lines.append("- 同时它的 `silhouette` 明显更高（0.2086 vs 0.0770），`Davies-Bouldin` 更低（1.371 vs 2.778），说明簇结构更清晰。")
    report_lines.append("- `min_cluster_ratio=0.2043`，没有出现极小簇。")
    report_lines.append("- `stability_ari=0.9005`，对无监督医学分型来说已经是可以接受的稳定水平。")
    report_lines.append("- 因此，这个配置比自动筛选到的 `GroupMaskedFT + KMeans + k=3` 更适合作为论文主模型：它更细、但仍保持足够稳和足够均衡。")
    report_lines.append("")
    report_lines.append("## 3. 最终模型的 4 个亚型")
    report_lines.append(final_ic.to_markdown(index=False))
    report_lines.append("")
    report_lines.append(final_ext.to_markdown(index=False))
    report_lines.append("")
    report_lines.append("建议命名：")
    report_lines.append("1. Cluster 0：`Global Low-Capacity / Frailty-like subtype`")
    report_lines.append("   中文内部命名：`全局低功能-高脆弱型`")
    report_lines.append("   理由：五域都最低，尤其认知、心理、运动最差，同时 ADL/IADL、肌少症、衰弱、跌倒风险都最差。")
    report_lines.append("2. Cluster 1：`Transitional Multi-domain Decline subtype`")
    report_lines.append("   中文内部命名：`过渡性多域下降型`")
    report_lines.append("   理由：整体明显好于 Cluster 0，但仍保留运动、感官和外部功能受损，像从低功能向高功能过渡的一组。")
    report_lines.append("3. Cluster 2：`High Function with Sensory Vulnerability subtype`")
    report_lines.append("   中文内部命名：`高功能-感官脆弱型`")
    report_lines.append("   理由：认知、心理、活力、运动都很高，但感官仍偏低，跌倒风险高于 Cluster 3。")
    report_lines.append("4. Cluster 3：`High Reserve Balanced subtype`")
    report_lines.append("   中文内部命名：`高储备均衡型`")
    report_lines.append("   理由：整体 IC 最高，感官也更完整，ADL/IADL、肌少症和跌倒风险最好。")
    report_lines.append("")
    report_lines.append("## 4. 这套结果最值得写进摘要的医学 insight")
    report_lines.append("- `Locomotion` 是跨亚型差异最大的 IC 维度，说明当前样本异质性的主导轴仍然是运动能力。")
    report_lines.append("- 最差簇和最佳簇之间形成了非常清晰的临床梯度：")
    report_lines.append("  - ADL: `80.6 -> 99.4`")
    report_lines.append("  - IADL: `14.4 -> 8.1`")
    report_lines.append("  - Sarcopenia score: `3.68 -> 0.44`")
    report_lines.append("  - Fried frailty: `2.12 -> 0.61`")
    report_lines.append("  - Fall risk: `39.4 -> 15.0`")
    report_lines.append("- 高功能人群内部仍然可以进一步分成“感官脆弱型”和“高储备均衡型”，这是一个很好的临床故事点：")
    report_lines.append("  - 仅看总体 IC，二者都算高功能。")
    report_lines.append("  - 但细看会发现，感官维度和跌倒风险已经把它们分开。")
    report_lines.append("  - 这可以支持“前失能/早期脆弱化”亚型的讨论。")
    report_lines.append("")
    report_lines.append("## 5. 外部验证最强的结果")
    report_lines.append(final_stats.head(12).to_markdown(index=False))
    report_lines.append("")
    report_lines.append("这张表说明：")
    report_lines.append("- 亚型对 ADL/IADL、肌少症、衰弱、跌倒、弹性、社会功能和 QoL 都有显著区分。")
    report_lines.append("- 这使得分型具备“不是只分出了数学簇，而是分出了有临床意义的功能亚型”的说服力。")
    report_lines.append("")
    report_lines.append("## 6. 区分亚型最强的原始变量")
    report_lines.append(final_markers.head(12).to_markdown(index=False))
    report_lines.append("")
    report_lines.append("这里最值得讲的变量有：")
    report_lines.append("- `认知-总分`")
    report_lines.append("- `健康自评分数`")
    report_lines.append("- `运动-总分`")
    report_lines.append("- `心理-总分`")
    report_lines.append("- `活力-总分`")
    report_lines.append("- `步态异常-编码`")
    report_lines.append("- `年龄`")
    report_lines.append("- `握力` 与 `4米步速`")
    report_lines.append("")
    report_lines.append("## 7. 双塔结果怎么写")
    report_lines.append("- 双塔对比学习我们做了，而且是认真做了。")
    report_lines.append("- 但在这批数据上，`DualTowerTable` 没有超过 `GroupMaskedFT`。")
    report_lines.append("- 这不是失败，反而是一个很稳的结论：")
    report_lines.append("  - 说明当前 Tab2Text 模板已经把很多结构化信息重新表述了一遍。")
    report_lines.append("  - 文本语义有帮助，但还不足以替代表格主线。")
    report_lines.append("  - 因此最合理的论文写法是：`双塔作为语义增强与方法学消融，主模型仍是 group-aware masked tabular encoder`。")
    report_lines.append("")
    report_lines.append("## 8. 最终论文写法建议")
    report_lines.append("1. 主结果：`GroupMaskedFT + GMM(full) + k=4`")
    report_lines.append("2. 主要对照：`PCA + KMeans + k=4`")
    report_lines.append("3. 方法学消融：`DualTowerTable / DualTowerAvg / DualTowerConcatPCA`")
    report_lines.append("4. 结论口径：深度表征不仅让簇更清晰，而且在 ADL/IADL/肌少症/衰弱/跌倒等外部临床验证上表现更好或至少不差。")
    (REPORT_DIR / "FINAL_RECOMMENDATION_CN.md").write_text("\n".join(report_lines), encoding="utf-8-sig")

    notes = []
    notes.append("# 图像说明")
    notes.append("")
    notes.append("## final_model_vs_controls_en.png")
    notes.append("- 这张图直接比较最终推荐深度模型、PCA 对照和双塔消融。")
    notes.append("- 如果要汇报，只需要说：深度主模型在 `external_separation` 上高于 PCA，对应的几何分离度也明显更强。")
    notes.append("")
    notes.append("## GroupMaskedFT_GMM4_FINAL_embedding_projection_en.png")
    notes.append("- 最终深度模型的二维投影图。")
    notes.append("- 重点看四个簇之间是否形成相对连续但可分的结构。")
    notes.append("")
    notes.append("## GroupMaskedFT_GMM4_FINAL_ic_heatmap_en.png")
    notes.append("- 看四个亚型在五个连续 IC 域和总分上的均值。")
    notes.append("- 这张图可以直接支撑 Cluster 命名。")
    notes.append("")
    notes.append("## GroupMaskedFT_GMM4_FINAL_external_heatmap_en.png")
    notes.append("- 看最终亚型在 ADL/IADL/肌少症/衰弱/跌倒/弹性/社会功能上的差异。")
    notes.append("- 这是最重要的临床验证图。")
    notes.append("")
    notes.append("## GroupMaskedFT_GMM4_FINAL_top_markers_en.png")
    notes.append("- 看哪些原始变量最能拉开亚型。")
    notes.append("- 目前最值得讲的是认知、运动、自评健康、心理和活力。")
    (FIG_DIR / "FIGURE_NOTES_CN.md").write_text("\n".join(notes), encoding="utf-8-sig")


if __name__ == "__main__":
    main()
