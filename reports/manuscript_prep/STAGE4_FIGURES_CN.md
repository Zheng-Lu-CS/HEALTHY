# 阶段 4：图表与可视化规划

## A. 最终推荐图表目录

### 主文 Figures

1. Figure 1. Study workflow and analysis design
2. Figure 2. Comparison of candidate representation-learning strategies
3. Figure 3. Two-dimensional projection of the primary three-phenotype solution
4. Figure 4. Continuous IC profile across the three phenotypes
5. Figure 5. External clinical validation across the three phenotypes
6. Figure 6. Top phenotype-differentiating markers

### 主文 Tables

1. Table 1. Cohort characteristics overall and by phenotype
2. Table 2. Method comparison of candidate clustering solutions
3. Table 3. External validation statistics of the primary model

### Supplementary

1. Figure S1. Four-cluster sensitivity analysis showing sex-dominant split
2. Figure S2. Dual-tower ablation results
3. Figure S3. Exploratory province/source enrichment
4. Table S1. Continuous IC rule specification
5. Table S2. External validator list and construct-overlap note

## B. 每张图/表的功能说明

### Figure 1

目的：

- 一眼讲清楚项目不是“直接聚类”，而是完整的治理、评分、表征、分型、外部验证流程

对应正文：

- Methods opening

关键信息：

- raw data
- cleaning
- continuous IC
- candidate embeddings
- clustering selection
- external validation
- sensitivity analysis

### Figure 2

目的：

- 让审稿人知道我们认真比较了线性基线、表格深度模型、文本增强模型

对应正文：

- Results 2

关键信息：

- `PCA`
- `GroupMaskedFT`
- `DualTowerTable / Avg / Concat`
- 指标：`silhouette`, `stability_ari`, `external_separation`, `min_cluster_ratio`

### Figure 3

目的：

- 展示主模型 3 簇在 latent space 中的可分离性

对应正文：

- Results 3

关键信息：

- 3 个 phenotype
- cluster size
- 避免图上堆太多统计符号

### Figure 4

目的：

- 展示 3 个 phenotype 的 IC 结构，不只是 overall high/low

对应正文：

- Results 3

关键信息：

- Cognition
- Psychological
- Vitality
- Locomotion
- Sensory
- Overall IC

### Figure 5

目的：

- 把 cluster 和 ADL/IADL/肌少症/衰弱/跌倒/QoL 的外部梯度讲清楚

对应正文：

- Results 4

关键信息：

- 主张“functional relevance”而不是“independent validation”

### Figure 6

目的：

- 告诉审稿人哪些变量最能区分 phenotype

对应正文：

- Results 4

关键信息：

- 认知总分
- 自评健康
- 运动总分
- 心理总分
- 活力总分
- 步态异常
- 年龄
- 握力 / 4 m gait speed

### Table 1

目的：

- 让主结果具备临床稿的基本体例

对应正文：

- Results 1 和 Results 3 开头

建议字段：

- n
- age
- sex
- self-rated health
- chronic disease count
- medication count
- source
- education
- marital status
- province missingness summary

### Table 2

目的：

- 用表格一次性说明候选方法比较，避免正文重复

对应正文：

- Results 2

### Table 3

目的：

- 精确汇报外部统计结果与效应大小

对应正文：

- Results 4

## C. 可直接使用 / 需修改 / 需重做 分类

### 可直接使用

- `figures/advanced_subtyping/GroupMaskedFT_embedding_projection_en.png`
- `figures/advanced_subtyping/GroupMaskedFT_ic_heatmap_en.png`
- `figures/advanced_subtyping/GroupMaskedFT_external_heatmap_en.png`
- `figures/advanced_subtyping/GroupMaskedFT_radar_en.png`
- `outputs/advanced_subtyping/GroupMaskedFT_external_stats.csv`
- `outputs/advanced_subtyping/best_method_metrics.csv`

### 需修改

- `figures/advanced_subtyping/method_comparison_en.png`
  - 需在图注中明确：主稿采用 3 簇 GroupMaskedFT，PCA 为强基线
- `figures/advanced_subtyping/GroupMaskedFT_top_markers_en.png`
  - 需在正文或图注中提醒：部分 marker 与 locomotion-related constructs 邻近，不应夸大为独立决定因素
- `figures/advanced_subtyping/final_model_vs_controls_en.png`
  - 保留，但降级为 supplementary 或内部汇报图

### 需重做

- Figure 1 workflow schematic
- Table 1 baseline characteristics
- Figure S1 sex-dominant split sensitivity figure
- Region/source exploratory supplementary figure

## D. 复杂图的详细制作说明

### Figure 1：Workflow schematic

布局建议：

- 左到右 6 个模块
- 每个模块用矩形框，模块间用箭头

模块内容：

1. Raw data
   - `6025 participants`
   - `361 raw variables`
2. Data cleaning
   - explicit missing normalization
   - remove ultra-missing variables
   - reserve external validators
3. Continuous IC construction
   - five domains
   - `0-100` scaling
4. Candidate representations
   - `PCA-128`
   - `GroupMaskedFT`
   - `DualTower` ablations
5. Unsupervised phenotyping
   - clustering across `k=3-6`
   - internal geometry + stability + balance
6. Clinical interpretation
   - external functional correlates
   - top differentiating markers
   - sensitivity analyses

视觉建议：

- 背景纯白
- 模块色块用 muted teal / warm gray / desaturated orange
- 不要用过深颜色
- 每个模块右下角可加简短英文小标题

### Figure 2：Method comparison panel

推荐形式：

- 4 个并列柱状图或 1 个 grouped bar chart
- 指标顺序固定：`silhouette`, `stability`, `external separation`, `min cluster ratio`

需要特别标注：

- `PCA` = linear baseline
- `GroupMaskedFT` = primary deep tabular model
- `DualTower*` = semantic enhancement ablations

### Figure S1：4-cluster sensitivity

推荐形式：

- 左 panel：GMM4 embedding projection
- 中 panel：IC heatmap
- 右 panel：stacked sex proportion bars

标题建议：

- `Higher-granularity sensitivity solution (4 clusters) showed a sex-dominant split within the high-function group`

### Table 1：Baseline characteristics

建议按列组织：

- Overall
- Phenotype 1
- Phenotype 2
- Phenotype 3
- P value / standardized difference

至少包含：

- age
- sex
- education
- marital status
- self-rated health
- chronic disease count
- medication count
- source

## E. 图表与论文章节的对应关系

| 图表 | 章节 |
|---|---|
| Figure 1 | Methods |
| Figure 2 | Results 2 |
| Figure 3 | Results 3 |
| Figure 4 | Results 3 |
| Figure 5 | Results 4 |
| Figure 6 | Results 4 |
| Table 1 | Results 1 / 3 |
| Table 2 | Results 2 |
| Table 3 | Results 4 |

## 复盘与反思

阶段 4 的核心不是“图够不够多”，而是“每张图是否替正文承担了明确的论证任务”。当前仓库里已经有足够多的分析图，但真正能直接服务投稿主线的，其实主要集中在 `advanced_subtyping` 的那几张英文图。最大的补口仍然是 workflow figure 和 Table 1。

## 自动进入下一阶段

已进入阶段 5：基于上述蓝图和图表布局，撰写英文可提交版本草稿。

