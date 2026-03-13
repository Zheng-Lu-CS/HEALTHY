# 第一篇论文故事线（建议版）

## 建议题目
- Deep phenotyping of intrinsic capacity in older adults using continuous-domain scoring and group-aware multimodal representation learning
- 内部中文题目：基于连续化内在能力评分与语义增强深度表征学习的老年人异质性分型研究

## 创新点
1. 连续化 IC，而不是二值化 IC。
2. group-aware masked tabular encoder。
3. Tab2Text + BERT 文本塔与表格塔对比学习。
4. 用 ADL/IADL/肌少症/衰弱/跌倒/弹性做系统外部验证。

## 图表建议
1. Figure 1 研究流程图
2. Figure 2 方法比较图
3. Figure 3 最终模型二维投影
4. Figure 4 最终模型 IC 雷达图
5. Figure 5 最终模型外部验证热图
6. Figure 6 最终模型 top markers

## 审稿防守点
1. PCA 是强基线，我们保留且如实汇报。
2. 文本路线不是黑盒决策器，而是语义增强模块。
3. 外部验证变量不进入主聚类设计矩阵，避免明显信息泄露。