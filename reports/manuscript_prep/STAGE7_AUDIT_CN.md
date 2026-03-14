# 阶段 7：终审级自查

## A. 论文潜在问题清单

1. 主研究问题成立，但文章仍然容易被审稿人质疑为“更复杂的聚类方法替代了传统方法”，而不是一个真正有医学增量的问题。  
2. 主模型 `GroupMaskedFT + KMeans + k=3` 的正文证据链是成立的，但 `PCA` 外部分离度略高，若正文写成“深度模型整体更优”，会被抓住。  
3. 外部验证变量虽未进入主聚类矩阵，但与输入变量存在构念邻近，若写成“independent external validation”会构成过度表述。  
4. 3 簇结果更像“功能梯度兼具 phenotype 特征”，若写成完全独立机制型亚群，会过度拔高。  
5. 4 簇 `GMM4_FINAL` 的高功能组存在明显 sex-dominant split，若在主文中占据过重位置，会显著增加拒稿风险。  
6. `GroupMaskedFT_GMM4_FINAL_*` 当前缺少明确生成脚本，补充材料若直接引用而不说明，会留下复现缺口。  
7. 论文目前仍缺正式版 `Table 1`、workflow figure 和图表编号映射，投稿前属于必须补齐的体例短板。  
8. 横断面设计天然限制了因果、进展、分期和预后表述，讨论若不够克制，容易被批评为 overstated。  
9. 地区/来源差异当前只适合 exploratory enrichment，若写入主结论会超过证据边界。  
10. 文本双塔结果属于负结果或中性结果，若硬包装成“多模态带来增益”，会与现有结果冲突。

## B. 严重程度分级

- 高：问题 2、3、5、6、8  
- 中：问题 1、4、7  
- 低：问题 9、10

## C. 每个问题对应的修改建议

1. 把主问题明确写成“continuous IC-guided cross-sectional phenotyping”，而不是“new AI clustering method”。  
2. 统一正文口径为“deep model improved latent structure, while PCA remained a strong clinical baseline”，避免“outperformed across all metrics”。  
3. 全文统一使用 `related functional correlates`、`held-out functional validators` 或 `external clinical correlates`，不用 `independent validation endpoints`。  
4. 在 Results 和 Discussion 中直接承认 3 簇兼具“phenotype”和“ordered functional spectrum”特征。  
5. 把 4 簇结果降级到补充材料，并明确写出其高功能群 sex-dominant split。  
6. 在方法或补充材料中说明 `GMM4_FINAL` 为 post hoc sensitivity solution，当前仓库仅保存结果文件，未形成完整 tracked generation chain。  
7. 投稿前补齐 Table 1、workflow schematic、Figure/Table 编号映射和图注统一体例。  
8. 全文避免使用 `trajectory`、`progression`、`predictive`、`causal` 等强词，除非明确加上 `cross-sectional approximation` 或 `cannot infer`。  
9. 地区/来源差异仅放在 Supplementary，并用 `exploratory` 限定。  
10. 将双塔结果写成认真完成的语义增强消融，而不是主要创新成效。

## D. 哪些问题我可以直接修正

- 主文中关于深度模型与 PCA 的表述边界。  
- 外部验证术语的统一。  
- 3 簇结果的叙事语气控制。  
- 4 簇结果从主发现降级为敏感性分析。  
- 双塔结果的定位。  
- 讨论中对横断面边界、proxy IC 边界和非因果边界的强化。  
- 阶段 5-7 稿件与投稿包文件补齐。

## E. 哪些问题必须由人类补充

- 正式投稿版 `Table 1` 的最终字段选择与排版。  
- workflow figure 的最终绘制。  
- 图表编号、图注和补充材料的最终投稿格式化。  
- 是否需要额外跑 sex-stratified 或 de-sexed sensitivity analysis 来进一步处理 4 簇风险。  
- 如需投稿更高层级期刊，是否补做外部队列验证或额外稳健性实验。

## F. 给作者的最终投稿前核对清单

1. 确认主稿主模型固定为 `GroupMaskedFT + KMeans + k=3`，不要在摘要和正文中混入 `GMM4_FINAL` 主结果口径。  
2. 确认全文所有 “验证” 表述均不暗示完全独立终点。  
3. 确认摘要、结果、讨论、结论对 `PCA` 的定位完全一致。  
4. 确认 Figure 1 workflow、Figure 2 方法比较、主模型 phenotype 图和补充 4 簇敏感性图之间逻辑连贯。  
5. 确认 `Table 1`、`Table 2`、`Table 3` 与正文数字完全一致。  
6. 确认补充材料中单独说明 4 簇 sex-dominant split 和复现链现状。  
7. 确认所有“轨迹”“早期脆弱化”“分期”类措辞都加上横断面限定。  
8. 确认标题、摘要和讨论没有把本文包装成未来风险预测研究。  
9. 确认地区/来源差异只作为 exploratory 内容。  
10. 确认目标期刊若为 JGSA，全文篇幅、结构化摘要、图表数量和补充材料体例均符合期刊要求。
