# 阶段 3：论文蓝图设计

## A. 论文总标题候选

### 候选 1

**Deep phenotyping of intrinsic capacity in older adults using continuous-domain scoring and group-aware masked representation learning**

优点：

- 最平衡
- 不夸张
- 既点到 continuous IC，也点到 deep representation

### 候选 2

**Continuous intrinsic capacity modeling and data-driven functional phenotypes in older adults from multidimensional geriatric assessment data**

优点：

- 医学味更强
- 对深度学习着墨更少，更稳

### 候选 3

**Group-aware self-supervised phenotyping of intrinsic capacity heterogeneity in older adults**

优点：

- 方法感更强

缺点：

- 对临床期刊来说略显“先方法、后问题”

### 我建议的最终标题

采用候选 1。

## B. 论文完整大纲

### Title

Deep phenotyping of intrinsic capacity in older adults using continuous-domain scoring and group-aware masked representation learning

### Abstract

结构：

1. Background and Objectives
2. Research Design and Methods
3. Results
4. Discussion and Implications

### Introduction

建议 4 段：

1. 老龄化背景下，IC/功能维持的重要性
2. 现有 IC 横断面文章主要问题：二值化、低维输入、LCA/传统方法居多
3. 结构化老年综合评估数据的机会：可做 richer phenotyping，但必须避免过度承诺未来预测
4. 本研究目的：连续 IC + group-aware masked representation + 无监督功能亚型 + 外部功能验证

### Research Design and Methods

建议分 6 小节：

1. Data source and participants
2. Data cleaning and feature selection
3. Continuous IC construction
4. Representation learning and clustering
5. External validation and marker discovery
6. Sensitivity analyses

### Results

建议分 5 小节：

1. Sample characteristics and data quality
2. Candidate embedding comparison
3. Final three-phenotype solution from GroupMaskedFT
4. External clinical validation
5. Sensitivity analyses

### Discussion

建议分 6 小节：

1. Principal findings
2. Relation to existing IC / LCA / functional aging literature
3. Clinical interpretation of the three phenotypes
4. What the deep model adds, and what it does not add
5. Limitations
6. Conclusions

## C. 每节要点与证据映射

### Introduction

核心目的：

- 让审稿人认同：在没有随访的前提下，做 continuous IC phenotyping 仍然是成立的问题

主要证据：

- `reports/formal_subtyping/LITERATURE_REVIEW_CN.md`
- `handover_cn/03_论文借鉴_对标与可吸收.md`
- JGSA / The Gerontologist / BMC / CIA 对标论文

审稿人可能质疑：

- “你们没有 gold standard IC”
- “你们为什么不做 LCA”

规避策略：

- 坚持把 continuous IC 写成 proxy/approximation
- 把 LCA 作为对标文献，而不是被动回避

### Methods - Data and cleaning

核心目的：

- 交代样本、变量、缺失与筛选边界

主要证据：

- `data/数据导出-202602.xlsx`
- `reports/stage1/D1_data_audit_report.md`
- `reports/DATA_CLEANING_SPEC.md`

质疑点：

- 地区字段缺失高
- rare disease duration/treatment 几乎全空

规避策略：

- 明确说明：这些字段不进入主设计矩阵

### Methods - Continuous IC

核心目的：

- 证明你们没有“瞎造分”，而是在 ICOPE/老年量表语境下进行可解释近似

主要证据：

- `scripts/formal_subtyping_pipeline.py`
- `outputs/formal_subtyping/IC_continuous_scores.csv`

质疑点：

- 权重是否任意

规避策略：

- 说清是 clinically informed approximation
- 不宣称“validated gold-standard IC score”

### Methods - Representation and clustering

核心目的：

- 说明为什么选择 `GroupMaskedFT`
- 同时说明为什么保留 `PCA`

主要证据：

- `scripts/advanced_deep_subtyping_pipeline.py`
- `outputs/advanced_subtyping/best_method_metrics.csv`
- `outputs/advanced_subtyping/best_method_selection.csv`

质疑点：

- 为什么 deep model 不是全面优于 PCA

规避策略：

- 主动承认 PCA 强
- 把 deep model 的价值写成“更清晰的 latent structure + 后续可扩展性”，不是“全面碾压”

### Results - Final model

核心目的：

- 把 3 个 phenotype 讲清楚

主要证据：

- `outputs/advanced_subtyping/GroupMaskedFT_cluster_assignments.csv`
- `outputs/advanced_subtyping/GroupMaskedFT_ic_profile.csv`
- `outputs/advanced_subtyping/GroupMaskedFT_external_profile.csv`
- `outputs/advanced_subtyping/GroupMaskedFT_top_markers.csv`

质疑点：

- 这是不是只是 low-high gradient，没有真正 subtype

规避策略：

- 直接承认它兼具“分型”和“分期/梯度”特征
- 不强行夸大成多个完全独立机制型亚群

### Results - Sensitivity analyses

核心目的：

- 提前替审稿人处理最危险的问题

主要证据：

- `outputs/advanced_subtyping/PCA_*`
- `outputs/advanced_subtyping/GroupMaskedFT_GMM4_FINAL_*`
- `outputs/advanced_subtyping/DualTower*`

质疑点：

- 4 簇为什么不用
- dual tower 为什么不用

规避策略：

- 明确写：4 簇版本产生了 sex-dominant split，因此降级为敏感性分析
- dual tower 是认真做过的 ablation，但当前未带来稳定增益

### Discussion

核心目的：

- 让文章以“审慎但有 insight”的方式收束

主要证据：

- 主模型外部验证梯度
- 与 BMC/CIA/JGSA/The Gerontologist 对标文献的异同

质疑点：

- 是否过度拔高

规避策略：

- 明确限定：cross-sectional, proxy IC, related validators, no causal claim, no prognostic claim

## D. 预期图表布局

### Main Figures

1. Figure 1. Study design and analysis workflow
2. Figure 2. Method comparison and model selection
3. Figure 3. Two-dimensional embedding projection of the primary model
4. Figure 4. Continuous IC profile of the three phenotypes
5. Figure 5. External clinical validation heatmap
6. Figure 6. Top phenotype-differentiating markers

### Main Tables

1. Table 1. Baseline characteristics of the cohort and the three phenotypes
2. Table 2. Candidate method comparison
3. Table 3. External validation statistics for the primary model

### Supplementary

- Figure S1. Four-cluster sensitivity solution and sex-dominant split
- Figure S2. Dual-tower ablation results
- Figure S3. Exploratory region/source enrichment
- Table S1. Continuous IC rule specification
- Table S2. External validators and construct-overlap notes

## E. 高风险段落与规避策略

### 风险 1：摘要中把结果写成 deep model superiority claim

规避：

- 不说“outperformed across all metrics”
- 只说“yielded clearer latent structure while maintaining clinically meaningful external gradients”

### 风险 2：把 3 个簇写成强机制型亚群

规避：

- 用 `phenotypes` 或 `functional phenotypes`
- 避免过强的 pathophysiology 语言

### 风险 3：把 GMM4 的两个高功能簇写成独立核心发现

规避：

- 只放 sensitivity analysis
- 明说其受性别驱动，需要后续 sex-stratified 或 de-sexed modeling 再验证

### 风险 4：把 external validation 写成“完全独立验证”

规避：

- 统一使用 `external clinical correlates` 或 `related functional validators`

## 复盘与反思

阶段 3 的关键，是把“项目里有什么结果”转换成“期刊能接受什么叙事”。蓝图确定后，后面写作就不再是把结果堆进去，而是围绕一个主问题去安排证据顺序。这里我主动把 `GMM4` 从主位移到敏感性分析，是为了让整篇文章的逻辑更稳、而不是更保守。

## 自动进入下一阶段

已进入阶段 4：以主模型 `GroupMaskedFT + KMeans + k=3` 为中心，规划最终投稿图表与表格。

