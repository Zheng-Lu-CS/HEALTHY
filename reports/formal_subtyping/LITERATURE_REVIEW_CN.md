# 后续论文方向文献对标（2026-03-13 检索）

## 1. 先说结论

如果把你们现在的目标概括成一句话，就是：

`用横断面老年综合评估表格数据，做出可信的连续化 IC 表征、可解释的亚型分层，并且把深度学习/大模型真正用在“恢复语义结构”上，而不是只换一个模型名。`

围绕这个目标，我这次更看重三类文献：

1. `和我们医学问题最接近`：Intrinsic Capacity、frailty、sarcopenia、ADL/IADL、老年综合评估。
2. `和我们数据形态最接近`：社区/队列老年人，主要是问卷+量表+结构化变量。
3. `和我们方法升级最接近`：机器学习、表格深度学习、文本-表格融合、LLM 辅助表示学习。

一个很重要的判断是：

- 老年医学/老年流行病学主流期刊已经接受机器学习，甚至明确出现了 AI 专刊。
- 但是这些期刊更买账的是：`临床问题清楚 + 基线扎实 + 解释充分 + 外部验证/敏感性分析完整`。
- 相比之下，`纯 LLM/纯 Transformer 炫技` 在 gerontology/geriatrics 期刊里并不占优势。
- 所以你们最稳的策略不是“全文都押在 LLM 上”，而是：
  - 主线：`FTTransformer / group-aware tabular encoder`
  - 辅线：`Tab2Text + BERT/LLM` 做语义增强
  - 强基线：`PCA / XGBoost / LCA`
  - 强解释：`连续 IC + 外部临床验证 + 去泄露分析`

## 2. 重点文献清单

下面这些文章我按“对我们现在最有用”的顺序来讲。这里的 `Q1/Q2/A区` 是按期刊通行声誉和常见分区印象来判断，真正投稿前还是建议你们按学校/医院认定目录再核一次。

### 2.1 The Gerontologist, 2025

**A machine learning approach for estimating intrinsic capacity age and its associations with multimorbidity and geroprotective agents**  
- 期刊：`The Gerontologist`
- 时间：`2025-11-10`
- 来源：
  - PubMed: <https://pubmed.ncbi.nlm.nih.gov/41042915/>
  - DOI: <https://doi.org/10.1093/geront/gnaf228>

**研究问题**
- 能不能把 IC 五域综合成一个类似“生物年龄/功能年龄”的连续指标，也就是 `IC-age`？
- 这个连续指标能不能反映多病共存、体力活动、吸烟等老年健康状态？

**数据情况**
- `48,068` 名 `>=60 岁` 老年人。
- 数据来自 `SHARE 2021–2022`。
- 典型的大规模老年结构化评估数据，和你们的“多量表+多疾病+多生活方式”结构非常像，只是样本量更大。

**方法**
- 用 `Random Forest regression` 预测 IC-age。
- 输入包括 IC 五域相关生物标志/功能变量 + 人口学信息。
- 输出不是分类，而是一个连续年龄样的分数。

**主要结果**
- 预测误差约 `5.3 years`，相关 `r = 0.55`。
- 手握力、认知、感觉相关变量贡献大。
- IC-age 能区分多病共存和健康生活方式差异。

**对我们的启发**
- 这篇文章非常关键，因为它说明：`IC 不一定非要做成 0/1 域受损数，也可以做成连续的功能年龄轴。`
- 你们现在已经做了 `IC_total_cont_100`，下一步完全可以升级成：
  - `IC functional age`
  - 或者 `IC latent axis / healthy aging axis`
- 这比简单的“高/中/低风险”更高级，也更容易写成 A 类论文的亮点。

**我们可以借鉴什么**
- 用连续目标替代粗分层。
- 把 IC 写成“综合健康老化表型”，而不是仅仅一个打分表。

**我们不要照抄什么**
- 它本质还是传统机器学习回归。
- 如果我们只是“再做一遍 RF/XGBoost 回归”，方法创新会不够。

---

### 2.2 Maturitas, 2025

**Interpretable machine learning models to predict decline in intrinsic capacity among older adults in China: a prospective cohort study**  
- 期刊：`Maturitas`
- 时间：`2025`
- 来源：
  - ScienceDirect: <https://www.sciencedirect.com/science/article/abs/pii/S0378512225004025>
  - DOI: <https://doi.org/10.1016/j.maturitas.2025.108594>

**研究问题**
- 能不能预测老年人未来 4 年内的 IC 下降？

**数据情况**
- `822` 名老年人。
- `CHARLS 2011 baseline + 2015 follow-up`。
- 这篇是中国人群，非常接近你们后面写作时的语境。

**方法**
- 比较 `5` 个机器学习模型。
- 做解释性建模。
- 最终 `XGBoost` 最优。

**主要结果**
- `44.6%` 的老年人出现 IC 下降。
- 最优模型测试集 `AUC = 0.715`。
- 只用了 `9` 个变量就建成了实用筛查模型。

**对我们的启发**
- 这类文章告诉我们：医学期刊是接受 ML 的，但他们要的是 `实用性 + 简洁性 + 可解释性`。
- 你们现在没有随访，因此不能直接复制“未来 IC decline 预测”。
- 但是可以借它的写法做：
  - `cross-sectional IC continuous score modeling`
  - `subtype discrimination`
  - `high-risk subtype screening`

**我们可以借鉴什么**
- 强调“可解释、能落地、适合基层筛查”。
- 用少量强变量做简化模型，作为深度模型之外的临床落地版。

**我们不要照抄什么**
- 不能硬做“预测未来下降”，因为你们没有未来标签。

---

### 2.3 Maturitas, 2026

**Patterns of intrinsic capacity impairment and their longitudinal association with possible sarcopenia in older adults: A latent class analysis**  
- 期刊：`Maturitas`
- 时间：`2026-01-22` online / `2026-03` issue
- 来源：
  - PubMed: <https://pubmed.ncbi.nlm.nih.gov/41579421/>
  - DOI: <https://doi.org/10.1016/j.maturitas.2026.108838>

**研究问题**
- 不同 IC 损伤模式，是否会对应未来“possible sarcopenia”的不同风险？

**数据情况**
- `1667` 名 `>=60 岁` 老年人。
- `CHARLS 2011–2015`。
- 先做 IC 模式，再看 3 年/5 年 possible sarcopenia。

**方法**
- `Latent Class Analysis (LCA)` 找 IC 模式。
- `Cox regression` 分析纵向风险。

**主要结果**
- 识别出 `4` 类 IC impairment patterns。
- 相比“相对健康类”，`运动下降`、`运动+感官下降`、`运动+心理+感官下降` 等模式有更高 possible sarcopenia 风险。
- 女性和 `60–70 岁` 人群中的高风险模式更明显。

**对我们的启发**
- 这是目前和你们“IC 分型 + 肌少症关联”最像的文章之一。
- 它直接告诉我们：`分型不是为了分型本身，而是为了和肌少症/衰弱/ADL 等外部结局建立桥。`
- 你们的 Excel 已经有肌少症、Fried、ADL、IADL、跌倒等变量，所以即使没有随访，也可以先做：
  - `亚型 × 肌少症/衰弱/ADL/IADL 横断面验证`

**我们可以借鉴什么**
- 簇命名方式：不是“cluster1/2/3”，而是“运动下降型”“运动-感官联合受损型”。
- 文章叙事：先做模式，再做临床意义验证。

**我们不要照抄什么**
- 他们是 LCA，方法比较传统。
- 你们如果还只是 LCA，方法上会显得太保守。

---

### 2.4 The Journals of Gerontology: Series A, 2023

**Machine Learning Models to Predict Future Frailty in Community-Dwelling Middle-Aged and Older Adults: The ELSA Cohort Study**  
- 期刊：`The Journals of Gerontology: Series A`
- 时间：`2023`
- 来源：
  - PMC: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10613015/>
  - DOI: <https://doi.org/10.1093/gerona/glad127>

**研究问题**
- 在社区中老年人里，能不能预测未来 frailty？
- 不平衡标签下，哪种机器学习模型更稳？

**数据情况**
- ELSA 队列。
- 基线 `2008–2009`，随访 `2012–2013`。
- 初始 `4637` 人，作者重点研究基线非 frail 人群。

**方法**
- 比较 `LR / RF / SVM / Neural Network / KNN / Naive Bayes`。
- 重点处理 `class imbalance`。

**主要结果**
- 类别平衡后性能更好。
- `Random Forest` 整体表现最好。
- 关键变量包括：肌力、平衡、自评健康等。

**对我们的启发**
- 这篇文章特别适合提醒我们：`医学 AI 文章不是模型越花哨越好，问题设定、样本不平衡、指标解释同样重要。`
- 你们虽然现在没有未来 frailty 标签，但已有 `Fried`、`衰弱快速筛查`，完全可以做：
  - `cluster external validation`
  - `IC continuous score external alignment`
  - 以及后续的横断面筛查模型

**我们可以借鉴什么**
- 类别不平衡处理思路。
- 把简单模型全做全，保证深度模型不是“空降冠军”。

**我们不要照抄什么**
- 你们不能直接讲“预测未来 frailty”，因为你们是横断面。

---

### 2.5 Psychogeriatrics, 2025

**Machine learning-based prediction of sarcopenia in community-dwelling middle-aged and older adults: findings from the CHARLS**  
- 期刊：`Psychogeriatrics`
- 时间：`2025-01`
- 来源：
  - PubMed: <https://pubmed.ncbi.nlm.nih.gov/39444246/>
  - DOI: <https://doi.org/10.1111/psyg.13205>

**研究问题**
- 问卷变量、血生化变量、二者结合，谁更适合筛查 sarcopenia？

**数据情况**
- `CHARLS 2011 + 2013`
- `N = 2934`
- `>=45 岁`
- 依据 `AWGS 2019` 定义 sarcopenia。

**方法**
- `5` 个机器学习模型。
- 分 `Q-based`、`Bio-based`、`combined` 三类输入。
- 还做了 temporal external validation。

**主要结果**
- 最优为 `XGBoost`
- `combined` 模型测试集 `AUROC = 0.759`
- 认知功能是重要预测因子之一，另外还有教育、慢病、IADL、血液指标等。

**对我们的启发**
- 这篇文章和你们非常贴近，因为你们现在就有：
  - 问卷/量表
  - ADL/IADL
  - 认知/心理
  - 肌少症相关项
- 这说明：`IC、认知、IADL、慢病负担` 与 sarcopenia 是可以被整合到同一分析框架里的。

**我们可以借鉴什么**
- 把 sarcopenia 当成外部验证终点或次级分析终点。
- 把“功能状态”和“肌少症风险”整合成一个更完整的老年综合画像。

**我们不要照抄什么**
- 如果只做 sarcopenia 分类，可能会把你们的大数据优势浪费掉。

---

### 2.6 Journal of Clinical Medicine, 2024

**Identification of Predictors of Sarcopenia in Older Adults Using Machine Learning: English Longitudinal Study of Ageing**  
- 期刊：`Journal of Clinical Medicine`
- 时间：`2024`
- 来源：
  - 期刊全文：<https://www.mdpi.com/2077-0383/13/22/6794>

**研究问题**
- 在老年人群中，能不能从大量变量里筛出最有效的 sarcopenia 预测因子？

**数据情况**
- `4994` 名老年人。
- Sarcopenia 患病率 `9.2%`。
- 类别明显不平衡。

**方法**
- 特征筛选：`MRMR`
- 分类器：`RUSBoosted Trees`
- 用少量特征替代全量特征。

**主要结果**
- 作者最终保留了 `9` 个最优特征。
- 强调在样本不平衡情况下，`RUSBoost` 很合适。
- 文章里明确说，少量强变量不损失太多精度。

**对我们的启发**
- 这篇文章非常适合你们以后做“精简筛查工具”。
- 也就是说，主论文可以是：
  - `深度学习分型`
- 衍生论文可以是：
  - `从分型结果反推最少变量筛查工具`

**我们可以借鉴什么**
- 先做 rich model，再做 sparse model。
- 这样投稿时会更像一个完整研究计划，而不是只有一个 fancy 模型。

---

### 2.7 Journal of Biomedical Informatics, 2025

**Leveraging heterogeneous tabular of EHRs with prompt learning for clinical prediction**  
- 期刊：`Journal of Biomedical Informatics`
- 时间：`2025-08`
- 来源：
  - PubMed: <https://pubmed.ncbi.nlm.nih.gov/40619074/>
  - DOI: <https://doi.org/10.1016/j.jbi.2025.104868>

**研究问题**
- 面对异构 EHR 表格，能不能先用 LLM 生成文本摘要，再和 Transformer 融合成更好的 patient representation？

**数据情况**
- `eICU-CRD` 公共数据 + 真实世界 `CECMed` 数据。
- 包含老年慢病患者。
- 虽然任务是 severity / mortality / LoS prediction，不是 IC 分型，但数据形态和你们非常接近：`异构结构化医疗表格`。

**方法**
- `Prompt learning` 引导 LLM 生成不同表格模块的文本摘要。
- `long text embedding` 得到统一文本向量。
- 再用 `cross-attention + self-attention` 做异构数据融合。

**主要结果**
- 相比多个 baseline 有优势。
- 作者做了模块级 ablation，证明每个组件有贡献。

**对我们的启发**
- 这是最直接支持你们“Tab2Text + BERT/LLM + 表格编码器”的方法学证据。
- 也说明：`LLM 不是不能发医学期刊，但最好作为表征增强/模态融合模块，而不是替代一切的黑盒。`

**我们可以借鉴什么**
- 你们完全可以把现在的路线升级成：
  - `文本塔：医学化 Tab2Text -> BERT/long text encoder`
  - `表格塔：FTTransformer`
  - `训练目标：对比学习/跨模态一致性`
  - `下游任务：subtyping + external validation`

**我们不要照抄什么**
- 这篇做的是预测任务。
- 你们不能直接机械移植；要把它改造成 `representation learning for subtyping`。

---

### 2.8 补充：BMC/开放获取方向的 cross-sectional subtype 文章

**Intrinsic capacity and health-promoting lifestyle in older adults: a latent class analysis**  
- 来源：PMC  
- 链接：<https://pmc.ncbi.nlm.nih.gov/articles/PMC12307196/>

**为什么值得看**
- 它是典型的横断面分型文章。
- `800` 名老年人，`5` 个社区，`ICOPE simple screening + lifestyle scale`。
- 最终做出 `3` 类：低心理-低健康型、相对健康型、低认知-低参与型。

**对我们的启发**
- 这种文章说明横断面数据照样可以发“pattern / class / subtype”。
- 但是方法过于传统，更多适合作为你们文章里的对照叙事，而不是主方法。

## 3. 从这些文献里能看出什么共性

### 3.1 数据共性

- 大多数研究都不是影像，而是 `结构化老年评估表`。
- 常见变量群：
  - 人口学
  - 慢病/多病共存
  - ADL/IADL
  - 认知/心理
  - 握力/步速/平衡
  - 营养/体重/BMI
- 这意味着你们现在的 `6025 x 361` 数据，其实是非常像主流 gerontology 研究的数据形态的。

### 3.2 问题设定共性

现有文章大多在做三件事：

1. `IC/衰弱/肌少症筛查`
2. `亚型识别 / latent class`
3. `未来风险预测`

你们现在没有随访，所以第 3 条暂时做不了，但前两条完全能做，而且可以做得更深。

### 3.3 方法共性

大部分文章还停留在：

- LCA
- Logistic / Cox / GLM
- RF / XGBoost / SVM
- SHAP / 简单特征选择

真正把 `LLM / prompt learning / tabular transformer / multimodal representation learning` 用到老年综合评估上的，还很少。

这恰恰意味着：

`你们的方法空间是空着的。`

## 4. 对我们最直接的写作策略建议

### 4.1 最稳的主线题目

最建议的主线不是“预测未来什么”，而是：

`Continuous intrinsic capacity modeling and deep representation-based subtyping in community-dwelling older adults`

或者更医学一点：

`Deep phenotyping of intrinsic capacity in older adults using continuous-domain scoring and multimodal representation learning`

### 4.2 最稳的技术路线

**主线模型**
- `FTTransformer / Tabular Transformer`
- 做 `group-aware` 输入：
  - demographics
  - disease burden
  - cognition
  - psychological
  - vitality
  - locomotion
  - sensory
  - region / center

**语义增强支线**
- `Tab2Text -> BERT/clinical text encoder`
- 不要让文本塔单独负责最终结论，而是作为语义补强。

**最值得做的 fancy 升级**
- `text-table contrastive learning`
- `masked modeling on tabular groups`
- `prototype-based clustering / deep clustering`

### 4.3 最该避免的坑

1. **只讲 silhouette，不讲医学意义**
   - 医学老师不关心簇形状好不好看，他们更关心 ADL/IADL/肌少症/衰弱有没有明显梯度。

2. **只讲 deep model，不和 PCA/XGBoost/LCA 比**
   - 这样很容易被问倒。

3. **把 LLM 当决定性计算器**
   - 这在医学稿子里风险很高。
   - 更稳的写法是：LLM 只负责 `semantic lifting`，真正的表示学习和分型还是由可复现模型完成。

4. **忽略信息泄露**
   - 如果外部验证里直接混入聚类时已经用过的高度重合量表，文章会非常危险。

## 5. 结合我们当前结果，我建议的“尽快发 A 类论文”路线

### 方向 A：主推，最适合现在立刻推进

**连续化 IC + FTTransformer 深度分型 + 外部临床验证**

具体写法：
- 第一步：构建 `continuous IC`
- 第二步：用 `FTTransformer` 学 128 维 embedding
- 第三步：做 `KMeans/GMM` 分型
- 第四步：用 `ADL/IADL/肌少症/Fried/跌倒/社会功能` 做独立外部验证
- 第五步：加入 `BERTText` 作为语义增强/消融对照

为什么这条线最稳：
- 和现有 gerontology 论文问题高度一致
- 又比传统 LCA/ML 明显更高级
- 你们现在已经有初步结果，可以继续深挖，不需要从零重来

### 方向 B：更 fancy，但要作为二阶段升级

**Tab2Text + BERT/LLM + FTTransformer 双塔对比学习**

核心思想：
- 把硬编码字段提升成更自然的临床语义描述
- 文本塔和表格塔学到一致的 patient representation
- 再在共享空间里聚类

为什么它适合当二阶段：
- 很新
- 很容易讲“语义信息不该被硬编码压扁”
- 但训练、调参、审稿解释都更难

### 方向 C：适合作为衍生短文/副产出

**从深度分型中反推最少变量的快速筛查模型**

比如：
- 先用 FTTransformer 做高质量分型
- 再用 XGBoost / LightGBM / sparse model 预测分型或低 IC 高风险亚型
- 生成“基层快速筛查版”

这条线很适合补一篇偏应用的文章。

## 6. 最后一句最实在的话

如果目标是“尽快发 A 类”，现在最值得押注的不是“让 LLM 直接决定医学结论”，而是：

`用深度表格表示学习做主线，用 LLM/BERT 解决语义硬编码问题，用外部临床量表把分型讲清楚。`

这样既有 fancy 的方法学，又不会在医学审稿里显得太飘。
