# 文件结构说明

## configs/formal_subtyping
- `feature_metadata.json`：正式分型时实际进入核心分析的字段清单，以及连续化 IC 列信息。
- `ft_cat_cards.json`：FTTransformer 使用的类别变量基数配置。

## outputs/formal_subtyping
- `reportable_method_summary.csv`：后处理后正式汇报使用的方法比较表。
- `*_cluster_assignments_reportable.csv`：重新生成且可复现的 cluster 结果。
- `*_ic_profile_en.csv`：英文列名的 IC 连续画像表。
- `*_external_profile_en.csv`：英文列名的外部临床画像表。
- `*_external_stats_en.csv`：各外部量表的 Kruskal-Wallis 检验结果。

## figures/formal_subtyping
- `*_embedding_projection_en.png`：二维投影图。
- `*_ic_heatmap_en.png`：IC 五域及总分热图。
- `*_radar_en.png`：IC 雷达图。
- `*_external_heatmap_en.png`：外部临床验证热图。
- `method_comparison_reportable_en.png`：方法总体比较图。
- `FIGURE_NOTES_CN.md`：图片读图说明，作为字体异常时的文字兜底。

## reports/formal_subtyping
- `TECHNICAL_REPORT_CN.md`：正式中文技术说明。
- `LITERATURE_REVIEW_CN.md`：后续论文方向的文献对标与投稿建议。
- `FILE_STRUCTURE_CN.md`：本文件。

## scripts
- `scripts/formal_subtyping_pipeline.py`：正式分型主脚本，负责连续化 IC、三条表征路线、聚类评估与初始输出。
- `scripts/formal_subtyping_postprocess.py`：正式汇报版后处理脚本，负责英文图重绘、聚类结果重建、中文报告与图注生成。
