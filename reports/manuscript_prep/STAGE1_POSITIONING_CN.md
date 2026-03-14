# 阶段 1：建立论文叙事基础

## A. 候选论文定位方案对比表

| 方案 | 核心研究问题 | 创新性 | 临床/应用意义 | 机制解释力度 | 实验支撑度 | A 类期刊适配度 | 我的判断 |
|---|---|---|---|---|---|---|---|
| 方案 A | 在横断面老年综合评估数据中，能否通过连续 IC 评分和 group-aware 表格表征学习识别 clinically meaningful 功能亚型？ | 中高 | 高 | 中 | 高 | 高 | 最推荐 |
| 方案 B | 能否构建“IC 功能年龄/功能轴”并据此描述健康老化连续谱？ | 高 | 中高 | 中高 | 中 | 中高 | 有潜力，但当前证据还不够完整 |
| 方案 C | 能否用去泄露机器学习筛查低 ADL/IADL 或高风险功能状态？ | 中 | 中 | 高 | 中高 | 中 | 更适合作为衍生短文，不适合作为主稿 |

### 方案 A：连续 IC + 深度表征 + 无监督功能亚型

- 研究问题：在没有未来标签的前提下，能否从 richer tabular representation 中恢复出老年人内在能力的异质性结构？
- 现有支撑：
  - 连续 IC 已实现，见 `outputs/formal_subtyping/IC_continuous_scores.csv`
  - `GroupMaskedFT` 自动最佳版本已实现且可复现，见 `outputs/advanced_subtyping/best_method_selection.csv`
  - 外部功能梯度清晰，见 `outputs/advanced_subtyping/GroupMaskedFT_external_profile.csv`
- 风险：
  - 如果直接使用 `GMM4_FINAL`，会遭遇 sex-driven split 风险
  - 因此必须把主模型收敛到更稳的 3 簇版本

### 方案 B：连续功能轴 / functional aging axis

- 研究问题：能否把连续 IC 或 embedding 进一步抽象成“功能年龄 / 退化轴”？
- 现有支撑：
  - `next_phase` 有 pseudotime 最小验证版，见 `reports/next_phase/NEXT_PHASE_TECHNICAL_REPORT.md`
  - `The Gerontologist 2025` 的 IC-age 文献提供了很好的对标
- 短板：
  - 当前 advanced 主线没有把功能轴单独做成正式主结果
  - 如果现在强推这一定位，会把不同阶段的结果硬拼在一起

### 方案 C：去泄露筛查模型

- 研究问题：在去除定义重叠变量后，能否筛查低 ADL/IADL 或高风险功能状态？
- 现有支撑：
  - 严格去重叠后仍有可观 AUC，见 `outputs/PRED_ADL_low_metrics_strict.csv` 和 `outputs/PRED_IADL_low_metrics_strict.csv`
- 短板：
  - 本质上仍是横断面 label 的近端映射
  - 问题意识没有方案 A 强
  - 也更容易被审稿人问“临床增量到底在哪里”

## B. 最推荐的最终定位

我推荐把主稿定位为：

**“基于连续化 IC 评分与 group-aware masked tabular representation learning 的老年人功能异质性分型研究”**

英文可表达为：

**Deep phenotyping of intrinsic capacity in older adults using continuous-domain scoring and group-aware masked representation learning**

更具体地说，主稿应采用：

- 主模型：`GroupMaskedFT + KMeans + k=3`
- 强基线：`PCA + KMeans + k=3`
- 语义增强/消融：`DualTower*`
- 高粒度敏感性分析：`GroupMaskedFT + GMM(full) + k=4`

## C. 推荐理由

### 1. 这条线最贴合项目原始目标

项目最初就不是要做单纯预测，而是要做“IC 近似评分 -> embedding -> 分型 -> 解释”。方案 A 和项目本体是同向的；方案 B 和 C 更像在已有主线上的分支。

### 2. 这条线能最大化现有最强证据

- 连续 IC：已经完成
- 深度表征：已经完成
- 对照基线：已经完成
- 外部功能验证：已经完成
- 负结果/边界：也已经完成（PCA 强、dual tower 未胜出、GMM4 sex split）

这意味着我们可以写出一篇“有主线、有对照、有边界”的论文，而不是一篇只挑最好看结果的论文。

### 3. 这条线最容易兼顾新意和可信度

如果直接押注双塔/LLM，会太冒险；如果只写 PCA/LCA，又会太保守。`GroupMaskedFT + 连续 IC + 强基线` 刚好位于“足够新”与“足够稳”之间。

### 4. 这条线最适合老年医学 A 类期刊口味

老年医学期刊往往更重视：

- 问题是否清楚
- 结果是否 clinically meaningful
- 方法是否透明
- 基线和敏感性分析是否完整

而不是单纯追求最花哨的模型名字。

## D. 该定位下最关键的证据链

### 证据链 1：项目需要连续 IC，而不是继续停留在二值 IC

- `scripts/formal_subtyping_pipeline.py` 中 `compute_continuous_ic()`
- `outputs/formal_subtyping/IC_continuous_scores.csv`
- `reports/formal_subtyping/TECHNICAL_REPORT_CN.md`

### 证据链 2：深度表格表征确实值得保留，但必须承认 PCA 很强

- `outputs/advanced_subtyping/best_method_metrics.csv`
- `outputs/advanced_subtyping/all_method_cluster_metrics.csv`
- `reports/advanced_subtyping/TECHNICAL_REPORT_CN.md`

### 证据链 3：主稿应选 `GroupMaskedFT + KMeans + k=3`，而不是无条件照搬 `GMM4_FINAL`

- `outputs/advanced_subtyping/best_method_selection.csv`
- `outputs/advanced_subtyping/GroupMaskedFT_cluster_assignments.csv`
- `outputs/advanced_subtyping/GroupMaskedFT_external_profile.csv`
- `outputs/advanced_subtyping/GroupMaskedFT_external_stats.csv`
- 我对 `GroupMaskedFT_GMM4_FINAL_cluster_assignments.csv` 与 `D2_with_IC.csv` 的再次合并核对：4 簇的高功能簇被性别强烈切开

### 证据链 4：亚型不是“数学簇”，而是有外部功能意义的功能分层

- `outputs/advanced_subtyping/GroupMaskedFT_external_profile.csv`
- `outputs/advanced_subtyping/GroupMaskedFT_external_stats.csv`
- `outputs/advanced_subtyping/GroupMaskedFT_top_markers.csv`

### 证据链 5：文本路线和预测路线都应降级为配角

- `outputs/formal_subtyping/reportable_method_summary.csv`
- `outputs/advanced_subtyping/best_method_metrics.csv`
- `outputs/PRED_ADL_low_metrics_strict.csv`
- `outputs/PRED_IADL_low_metrics_strict.csv`

## E. 还缺哪些支撑材料

虽然主线已经成立，但距离“投稿版本”还差以下几块：

1. 缺一个正式的 `Table 1`
   - 即主模型 3 个簇的基础人口学与核心临床特征汇总
2. 缺一个面向投稿的 workflow figure
   - 现有图件更多是分析图，不是 publication-grade 研究设计图
3. 缺一个对 sex-driven split 的明确敏感性说明
   - 如果保留 4 簇结果，必须说明它只能当 supplementary sensitivity
4. 缺一个对“外部验证并非完全独立”的更克制写法
5. 缺最终版 figure/table 编号与正文映射

## 复盘与反思

阶段 1 的关键，不是找到“最炫”的故事，而是把故事和证据重新对齐。仓库里最新的 `FINAL_RECOMMENDATION_CN.md` 给了一个很诱人的 4 簇版本，但严格核对之后，我认为它更适合作为敏感性分析而不是主稿核心。这一步虽然让故事少了一点“花哨”，但显著提高了可投稿性。

## 自动进入下一阶段

已基于上述定位进入阶段 2：筛选最适合这条叙事线的老年医学 A 类目标期刊，并学习其结构与风格。

