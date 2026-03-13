# 文件结构说明

## scripts
- scripts/advanced_deep_subtyping_pipeline.py：本次深度学习主线实验脚本。
- scripts/advanced_subtyping_finalize.py：论文定稿版后处理脚本，负责最终推荐模型、控制组对照和写作包整理。

## outputs/advanced_subtyping
- design_matrix.parquet：统一建模设计矩阵。
- group_masked_embedding.npy：group-aware masked table encoder 输出。
- dual_tower_*_embedding.npy：双塔不同融合方式的 embedding。
- all_method_cluster_metrics.csv：所有方法+clusterer+k 完整比较表。
- best_method_metrics.csv：各方法最优配置比较表。
- final_recommended_vs_controls.csv：最终推荐深度模型、PCA 对照和双塔消融的直接比较表。
- *_cluster_assignments.csv / *_ic_profile.csv / *_external_profile.csv / *_external_stats.csv / *_top_markers.csv：最终结果表。

## figures/advanced_subtyping
- training_curve_group_masked_en.png：表格预训练曲线。
- training_curve_dual_tower_en.png：双塔训练曲线。
- method_comparison_en.png：方法比较总图。
- final_model_vs_controls_en.png：最终推荐深度模型与控制组的直接比较图。
- *_embedding_projection_en.png / *_radar_en.png / *_external_heatmap_en.png / *_top_markers_en.png：核心图件。
- FIGURE_NOTES_CN.md：图件中文读图说明。

## reports/advanced_subtyping
- TECHNICAL_REPORT_CN.md：完整中文技术报告。
- FINAL_RECOMMENDATION_CN.md：论文定稿版最终模型建议与核心结论。
- PAPER_STORYLINE_CN.md：论文故事线。
- FILE_STRUCTURE_CN.md：本文件。
