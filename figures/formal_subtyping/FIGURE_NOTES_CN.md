# 图像说明（如果图片软件再次出现字体问题，可以直接看这里）

## method_comparison_reportable_en.png
- 比较三条路线在四个维度上的表现：silhouette、stability_ari、external_separation、min_cluster_ratio。
- `silhouette` 越大越说明簇内更紧、簇间更分。
- `stability_ari` 越大越说明换随机种子后结果更稳定。
- `external_separation` 越大越说明聚类能把 ADL/IADL/肌少症/衰弱/跌倒等外部临床变量分开。
- `min_cluster_ratio` 越大越说明没有特别小、特别难解释的边缘簇。

## BERTText_embedding_projection_en.png
- 看 BERTText 在二维空间里的分布。这里只看几何形状，不代表临床解释一定最强。
## BERTText_ic_heatmap_en.png
- 看每个 cluster 在五个连续 IC 域和总分上的均值。
## BERTText_radar_en.png
- 和热图是同一批信息，只是换成雷达图，更适合口头汇报。
## BERTText_external_heatmap_en.png
- 用外部临床量表验证聚类：ADL、IADL、肌少症、Fried 衰弱、跌倒风险、弹性、社会功能。
- 如果某个方法的外部热图层次感更强，说明它更接近“医学上有意义的分型”。

## FTTransformer_embedding_projection_en.png
- 看 FTTransformer 在二维空间里的分布。这里只看几何形状，不代表临床解释一定最强。
## FTTransformer_ic_heatmap_en.png
- 看每个 cluster 在五个连续 IC 域和总分上的均值。
## FTTransformer_radar_en.png
- 和热图是同一批信息，只是换成雷达图，更适合口头汇报。
## FTTransformer_external_heatmap_en.png
- 用外部临床量表验证聚类：ADL、IADL、肌少症、Fried 衰弱、跌倒风险、弹性、社会功能。
- 如果某个方法的外部热图层次感更强，说明它更接近“医学上有意义的分型”。

## PCA_embedding_projection_en.png
- 看 PCA 在二维空间里的分布。这里只看几何形状，不代表临床解释一定最强。
## PCA_ic_heatmap_en.png
- 看每个 cluster 在五个连续 IC 域和总分上的均值。
## PCA_radar_en.png
- 和热图是同一批信息，只是换成雷达图，更适合口头汇报。
## PCA_external_heatmap_en.png
- 用外部临床量表验证聚类：ADL、IADL、肌少症、Fried 衰弱、跌倒风险、弹性、社会功能。
- 如果某个方法的外部热图层次感更强，说明它更接近“医学上有意义的分型”。
