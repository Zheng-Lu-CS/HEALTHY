# 阶段 0：项目接管与材料盘点

## A. 项目全景摘要

这个项目不是单一版本，而是一条持续演化的老年健康科研流水线。按时间线看，项目大致经历了四层：

1. `stage1`：建立“数据治理 -> 二值 IC proxy -> AutoEncoder embedding -> KMeans 聚类 -> SHAP 解释”的第一版可复用流程。
2. `next_phase`：验证两条升级路线是否值得继续推进，即“自监督表格表征”和“Tab2Text + 对比学习”。
3. `formal_subtyping`：把二值 IC 升级为连续 IC，统一比较 `PCA / FTTransformer / BERTText` 三条正式分型路线，并补齐英文图件。
4. `advanced_subtyping`：进一步引入 `GroupMaskedFT` 和 `DualTower`，形成更接近投稿叙事的深度学习主线与控制组比较。

项目当前最接近投稿版本的材料不在 `stage1`，而在 `reports/advanced_subtyping/`、`outputs/advanced_subtyping/`、`figures/advanced_subtyping/`，但这些最新材料内部仍然存在“自动最优选择”和“论文最终推荐”两套不同口径，必须拆开理解，不能直接照抄。

## B. 项目材料清单

### 1. 原始数据与编码说明

- `data/数据导出-202602.xlsx`
  - 单表，`6025 x 361`
  - 无整行重复
  - 主键 `编号` 唯一
- `data/编码.docx`
  - 类别编码说明
  - 已被 `scripts/run_pipeline.py` 的 `parse_docx_mappings()` 用于标签映射
- `data/老年人重要内在功能综合评估体系建立与应用(1).pdf`
  - 背景材料
  - 对当前横断面分析有框架参考价值，但不是现阶段结果判定的主证据

### 2. 第一阶段主流程与基础报告

- `scripts/run_pipeline.py`
- `reports/WORKFLOW_REPORT.md`
- `reports/IC_RULES_SPEC.md`
- `reports/CLUSTER_MODEL_CARD.md`
- `reports/LEAKAGE_CHECK.md`
- `reports/PREDICTION_EXPERIMENTS*.md`
- `outputs/D2_with_IC.csv`
- `outputs/D3_*`
- `outputs/D4_*`
- `figures/D1_*`, `figures/D3_*`, `figures/D4_*`

作用：这是项目的可复现起点，也是后来所有升级实验的共同基座。

### 3. 交接与中文理解材料

- `handover_cn/00_总览.md`
- `handover_cn/01_项目工作流_快速上手.md`
- `handover_cn/02_关键产物_少而精.md`
- `handover_cn/03_论文借鉴_对标与可吸收.md`
- `handover_cn/04_下一步研究方向_A类论文.md`
- `handover_cn/05_当前结果能说明什么.md`
- `handover_cn/07_下一阶段方案验证与汇报包/*`

作用：这些文件不是最终证据本身，但对理解版本演化、作者当时的决策和叙事倾向非常关键。

### 4. 中间升级实验

- `scripts/next_phase_experiments.py`
- `reports/next_phase/NEXT_PHASE_TECHNICAL_REPORT.md`
- `outputs/next_phase/*`
- `figures/next_phase/*`

作用：回答“简单 AE 是否足够”“Tab2Text 这条路是否值得继续”等方法路线问题。

### 5. 正式分型版本

- `scripts/formal_subtyping_pipeline.py`
- `scripts/formal_subtyping_postprocess.py`
- `reports/formal_subtyping/TECHNICAL_REPORT_CN.md`
- `reports/formal_subtyping/LITERATURE_REVIEW_CN.md`
- `outputs/formal_subtyping/*`
- `figures/formal_subtyping/*`

作用：把项目从“二值分层”推进到“连续 IC + 多路线正式分型比较”，是第一次真正具备论文化结构的版本。

### 6. 深度学习主线版本

- `scripts/advanced_deep_subtyping_pipeline.py`
- `scripts/advanced_subtyping_finalize.py`
- `reports/advanced_subtyping/TECHNICAL_REPORT_CN.md`
- `reports/advanced_subtyping/FINAL_RECOMMENDATION_CN.md`
- `reports/advanced_subtyping/PAPER_STORYLINE_CN.md`
- `outputs/advanced_subtyping/*`
- `figures/advanced_subtyping/*`

作用：这是当前“最像投稿稿”的一组材料，但也是冲突最集中的一组材料。

## C. 研究主线理解

结合脚本、报告和输出，我对项目主线的理解如下：

1. 现实约束：只有横断面数据，没有随访、没有 RCT、没有未来结局标签，因此主问题不能写成“预测未来 decline/frailty/sarcopenia”。
2. 医学锚点：仍然必须站在 WHO ICOPE / IC 框架内，因此需要一个可解释的 IC 构造方式作为锚。
3. 方法主线：不是做单纯分类器，而是做“连续 IC 表征 + 无监督异质性发现 + 外部功能意义验证”。
4. 论文卖点：不在于“用了深度学习”本身，而在于：
   - 从二值 IC 升级到连续 IC
   - 从人工低维汇总升级到 richer tabular representation
   - 在不引入未来标签的前提下做 clinically meaningful phenotyping
   - 保留强基线，控制泄露和过度解释风险

## D. 当前最可信结论

下面这些结论，是我认为目前证据最扎实、可以进入论文主叙事的部分。

### 1. 数据质量与横断面属性是清楚的

- 原始数据 `6025 x 361`，无整行重复，唯一 ID 可用，见 `data/数据导出-202602.xlsx` 与 `reports/stage1/D1_data_audit_report.md`
- 年龄范围 `60-100`，均值约 `71.1`；出生日期无缺失；根据 `出生日期 + 年龄` 反推评估时间，中位数约为 `2024-06-21`，见对原始表的再次核对
- 地区字段存在，但 `省/市/区` 缺失率约 `28%-30%`，因此能做探索，难做强结论

### 2. 二值 IC proxy 是可用起点，但不是最终论文最强形态

- 第一版 IC 规则定义清晰，可直接写入方法部分，见 `reports/IC_RULES_SPEC.md` 与 `scripts/run_pipeline.py`
- 二值 IC 分布具有实际分层意义，`IC_total` 主要集中在 `1-3` 域受损，见 `reports/WORKFLOW_REPORT.md` 和 `outputs/D2_with_IC.csv`
- 但作者自己在交接文档中已多次指出二值化信息损失明显，连续化更合理，见 `handover_cn/02_关键产物_少而精.md`

### 3. 连续 IC 是项目真正的升级点之一

- `formal_subtyping` 开始采用五域 `0-100` 连续分数，保留程度信息，见 `scripts/formal_subtyping_pipeline.py` 的 `compute_continuous_ic()` 和 `outputs/formal_subtyping/IC_continuous_scores.csv`
- 连续 IC 的全样本均值显示：`Locomotion` 和 `Sensory` 相对最低，`Cognition / Psychological / Vitality` 相对更高
- 这为后续“功能梯度”而不是简单“是否受损”提供了更合适的叙事基础

### 4. 简单 AE 不是终点，PCA 是必须正视的强基线

- `next_phase` 结果显示 `PCA32` 明显优于 `DAE32` 和轻量对比学习版本，见 `reports/next_phase/NEXT_PHASE_TECHNICAL_REPORT.md`
- 这说明“上一个深度模型名字”并不能自动带来增益，后面所有 fancy 方案都必须和强线性基线正面对比

### 5. 文本路线可跑通，但当前更像语义增强，不是主分型器

- `formal_subtyping` 中 `BERTText` 的几何聚类指标最好，但外部临床区分度最差，见 `outputs/formal_subtyping/reportable_method_summary.csv`
- `advanced_subtyping` 中 `DualTower*` 也没有超过 `GroupMaskedFT`，见 `outputs/advanced_subtyping/best_method_metrics.csv`
- 因此，“文本/LLM 只做语义增强，不做决定性主模型”是可信结论

### 6. 当前最可信的深度主线不是 `GMM4` 最新推荐版，而是 `GroupMaskedFT + KMeans + k=3`

支持理由：

- 自动模型选择结果把 `GroupMaskedFT + KMeans + k=3` 选为该方法最佳方案，见 `outputs/advanced_subtyping/best_method_selection.csv`
- 该方案指标为：
  - `silhouette = 0.3107`
  - `stability_ari = 0.9878`
  - `min_cluster_ratio = 0.2398`
  - `external_separation = 0.2172`
  见 `outputs/advanced_subtyping/best_method_metrics.csv`
- 它生成 3 个临床有序簇，外部梯度清晰，见：
  - `outputs/advanced_subtyping/GroupMaskedFT_ic_profile.csv`
  - `outputs/advanced_subtyping/GroupMaskedFT_external_profile.csv`
  - `outputs/advanced_subtyping/GroupMaskedFT_external_stats.csv`

### 7. `GroupMaskedFT + GMM(full) + k=4` 只能作为高粒度敏感性分析，不能直接无条件当主结果

原因有三条，而且都很实：

1. 它不是当前脚本自动选出来的最佳方案，而是后续“论文推荐口径”的手工再选择，见：
   - `reports/advanced_subtyping/TECHNICAL_REPORT_CN.md`
   - `reports/advanced_subtyping/FINAL_RECOMMENDATION_CN.md`
2. 仓库中没有生成 `GroupMaskedFT_GMM4_FINAL_*` 的显式脚本；`advanced_subtyping_finalize.py` 只是读取这些文件，不负责生成，因此复现链条不完整
3. 我重新把 `outputs/advanced_subtyping/GroupMaskedFT_GMM4_FINAL_cluster_assignments.csv` 与 `outputs/D2_with_IC.csv` 合并后发现：
   - cluster 2 几乎全为男性（`99.8%`）
   - cluster 3 几乎全为女性（女性 `98.3%`）
   这说明 4 簇结果中，高功能人群很大概率被性别主导地再次切开了

## E. 冲突点 / 缺失点 / 风险点

### 1. 版本优先级存在口径冲突

- `reports/advanced_subtyping/TECHNICAL_REPORT_CN.md` 的自动最优口径偏向 `PCA`
- `reports/advanced_subtyping/FINAL_RECOMMENDATION_CN.md` 的论文推荐口径偏向 `GroupMaskedFT + GMM4`
- 两者并非简单覆盖关系，而是“自动最优”和“手工推荐”的决策标准不同

### 2. 最新推荐模型存在显著性别驱动风险

- `GroupMaskedFT` 的 4 簇切分会把高功能群体按性别强烈分开
- 这会直接威胁“高功能-感官脆弱型 vs 高储备均衡型”的临床解释

### 3. 外部验证并非完全独立

- 虽然 ADL/IADL 没进入主聚类矩阵，但外部验证中的 `Fried frailty score`、`sarcopenia score`、`fall risk` 与输入中的握力、步速、步态等变量存在不同程度的构念重叠
- 这不等于结果无效，但意味着它们更适合被写成“related functional validators”，而不是“完全外部独立终点”

### 4. 地域/来源差异存在，但当前只够做探索

- 以 `GroupMaskedFT_GMM4_FINAL` 为例，`province / city / source` 与 cluster 的关联在统计上都非常强
- 但这些关联同时受到缺失、样本来源结构、中心分布差异影响
- 所以最多能写成“exploratory enrichment”，不能写成稳固的地域机制结论

### 5. 复现链在最新版推荐模型上不闭环

- `GroupMaskedFT_GMM4_FINAL_*` 结果文件存在
- 但当前 tracked scripts 中没有明确生成它们的流程
- 这是投稿前必须说明或补上的 reproducibility gap

### 6. 预测实验不适合做主线

- `PREDICTION_EXPERIMENTS*.md` 和对应输出说明：ADL/IADL 横断面预测在严格去功能重叠后性能明显下降
- 这类任务更适合当补充应用探索，不适合当论文主叙事

## F. 你接下来准备如何进一步抽取信息

我在阶段 1 及以后会按以下原则推进：

1. 论文定位上，以“连续 IC + 深度表征 + 无监督功能亚型 + 外部功能验证”为主，而不是“未来预测”。
2. 模型选择上，优先采用可复现且不被单一人口学变量主导的结果。
3. 叙事上，承认 `PCA` 是强基线，不把 deep model 包装成碾压式胜利。
4. 图表上，优先调用 `advanced_subtyping` 的英文图件，但把 `GMM4_FINAL` 放到敏感性分析位置。
5. 风险控制上，所有关于“机制”“临床意义”“多中心差异”的句子都要绑定具体证据和边界说明。

## 复盘与反思

这一阶段最大的收获不是“项目能不能写论文”，而是“该写哪条论文主线、哪些结果不能硬写”。最新推荐材料表面上最像投稿版，但它恰恰藏着最大的两个坑：一是复现链断裂，二是 4 簇结果可能被性别强烈驱动。把这两个问题提前拆出来，后面所有阶段都会更稳。

## 自动进入下一阶段

已根据上述盘点结果进入阶段 1：优先在“最能被现有证据支撑”的前提下选择最终论文定位。

