# 阶段 2：目标期刊筛选与风格学习

## A. 目标期刊比较

我把“最值得认真考虑”的期刊分成四档：

| 期刊 | 适配度 | 优点 | 主要风险 | 我的建议 |
|---|---|---|---|---|
| The Journals of Gerontology: Series A | 最高 | 医学老化、功能衰退、机器学习、分层与临床意义兼容；接受方法但不迷信方法 | 需要很强的问题清晰度、边界控制和强基线 | 主投 |
| Maturitas | 很高 | 最近确实在发 IC decline、LCA、健康老化、应用型 ML；对横断面与应用导向更友好 | 偏好更直接的临床或公共卫生落地价值 | 第一备选 |
| Archives of Gerontology and Geriatrics | 高 | 接收 geriatric assessment、异质性、机器学习、社区老年样本；对 cross-sectional phenotyping 容忍度较高 | 需要避免“只是换模型名”的方法堆砌 | 第二备选 |
| Age and Ageing | 中等偏低 | 顶级老年医学平台，若命中则影响大 | 对横断面、无外部队列、proxy IC、无长期结局的容忍度低 | 不建议作为当前主投 |

### 1. The Journals of Gerontology: Series A

来自期刊官方 `About` 页面与作者指南的关键信号：

- 期刊明确覆盖 aging 的 biological 和 medical sciences
- 强调与年龄相关疾病、功能和干预有关的工作
- 作者指南对 research reports 的结构和篇幅控制较严格，更偏 concise、result-driven 的写法

为什么适合我们：

- 你们的问题是“老年功能状态异质性”而不是纯 AI 工程
- 你们有多量表、多病共存、ADL/IADL、肌少症、衰弱、QoL 这些 geriatric medicine 审稿人熟悉的支撑变量
- 期刊已有 frailty ML 文章和 aging-AI 相关内容，不会天然排斥方法升级

### 2. Maturitas

官方 `Aims and Scope` 与 `Guide for Authors` 显示：

- 关注 midlife and aging
- 对 healthy aging、功能下降、筛查和应用性建模接受度较高
- 更强调 public health / prevention / clinically usable interpretation

为什么是很强备选：

- 它已经发表 IC decline prediction、IC pattern/LCA、健康老化等相关文章
- 对“连续 IC + 功能分层 + 外部验证”这类 story 比较友好
- 如果 JGSA 认为 deep novelty 不足，它会是更稳的落点

### 3. Archives of Gerontology and Geriatrics

从官方 guide 和期刊近年选题看：

- 覆盖实验、临床、流行病学与社会老年学
- 接收 geriatric assessment、frailty、functional decline、multimorbidity、heterogeneity
- 但明确不喜欢“只有描述、没有更深解释”的稿件

为什么适合作为第二备选：

- 你们的数据形态和它经常收的 geriatric epidemiology 文章很接近
- 只要把“PCA 强基线 + deep model 边界 + 外部功能意义”讲清楚，适配度不错

### 4. Age and Ageing

官方 `About` 页面强调：

- 面向改变临床实践与老年照护体系的高影响研究
- 选题门槛更高，更偏向强设计、强外部验证或更直接可转化的研究

为什么暂时不建议主投：

- 你们当前没有纵向结局、没有外部验证队列、IC 仍是 proxy/approximation
- 如果硬投，最大风险不是 desk reject，而是“你们问题对，但证据强度还不够”

## B. 最终主投期刊与备选期刊

### 主投

**The Journals of Gerontology: Series A: Biological Sciences and Medical Sciences**

### 备选

1. **Maturitas**
2. **Archives of Gerontology and Geriatrics**

### 不建议当前主投

- `Age and Ageing`

## C. 该期刊写作风格提炼

以下风格提炼，以 JGSA 为主，同时参考其近似题材论文的实际写法。

### 1. 摘要节奏

常见写法：

- `Background and Objectives`
- `Research Design and Methods`
- `Results`
- `Discussion and Implications`

适合我们的原因：

- 这种结构允许你先用一句话把老年医学问题讲清楚，再介绍方法
- 最后还留有一个“Implications”位置，用来收束临床意义，但不会逼你夸大成因果

### 2. 引言风格

JGSA 更喜欢：

1. 先讲老龄化/功能维持/IC 的临床问题
2. 再讲现有做法的问题
3. 最后自然引出你们的方法与研究目的

不喜欢：

- 上来先讲 Transformer、LLM、self-supervision 多高级
- 或者用大段文献综述淹没问题本身

### 3. 结果呈现节奏

更合适的节奏是：

1. 数据与样本质量
2. 候选方法比较
3. 主模型的亚型结构
4. 外部功能意义验证
5. 敏感性分析与边界

而不是：

1. 先一口气展示所有 fancy 模型
2. 再到最后才说哪个结果能解释

### 4. 讨论写法

更像 JGSA 的讨论应当：

1. 先精确回收主发现
2. 再与 IC / frailty / LCA / functional aging 文献对齐
3. 讨论临床识别价值与方法学含义
4. 最后非常诚实地说局限性

### 5. 常见拒稿点

结合官方定位和近似文章，我认为最可能的拒稿点是：

1. 研究问题看似新，实则只是“换模型”
2. 结果只有内部聚类指标，没有临床外部意义
3. baseline 太强而作者不诚实汇报
4. 外部验证构念重叠严重却不承认
5. 横断面文章却使用了过强的因果或预测语言

## D. 代表性论文学习清单

### 1. JGSA 2023

**Machine Learning Models to Predict Future Frailty in Community-Dwelling Middle-Aged and Older Adults: The ELSA Cohort Study**

- 用途：学习工程规范和如何诚实比较多个 baseline
- 可借鉴：
  - 把 class imbalance、预处理、变量筛选写得很规范
  - 不只报一个指标
- 要避免：
  - 我们没有随访，不能复制其“future prediction”口径

### 2. The Gerontologist 2025

**A machine learning approach for estimating intrinsic capacity age and its associations with multimorbidity and geroprotective agents**

- 用途：学习如何把 IC 从“域受损计数”提升到“连续 aging biomarker”
- 可借鉴：
  - 连续表型的故事力
  - 把 IC 与多病共存、生活方式联系起来
- 要避免：
  - 不能仅仅复制一个回归器然后改名

### 3. BMC Geriatrics 2024

**Factors associated with intrinsic capacity impairment in hospitalized older adults: a latent class analysis**

- 用途：学习经典 IC 分型文章的临床叙事框架
- 可借鉴：
  - 先做 pattern，再做 ADL/IADL/衰弱/跌倒的外部验证
  - 命名风格清晰
- 要避免：
  - 不能停留在传统 LCA 的方法层级

### 4. Clinical Interventions in Aging 2025

**Patterns and Factors of Intrinsic Capacity Impairment in Older Adults with Chronic Diseases: A Latent Class Analysis**

- 用途：学习中国大样本 IC pattern 文章如何组织结果
- 可借鉴：
  - 大样本、三类模式、影响因素组织方式
- 要避免：
  - 它的输入仍主要是人工定义量表，方法升级空间就是我们的切入点

### 5. BMC Geriatrics 2022

**Intrinsic capacity of older people in the community using WHO Integrated Care for Older People (ICOPE) framework: a cross-sectional study**

- 用途：学习如何在 WHO ICOPE 框架下为 proxy/approximate IC 提供医学合法性
- 可借鉴：
  - 不把自己包装成“重新定义 IC”
  - 明确站在 ICOPE 框架内做现实世界扩展

## E. 这些论文中可借鉴之处

1. 问题意识要先于方法名
2. 亚型必须回到 ADL/IADL、衰弱、生活质量这些 geriatric outcome space
3. 强基线不能省
4. 讨论里必须承认横断面边界
5. 如果方法升级不明显，也要把“为什么还值得写”说清楚

## F. 这些论文中需要避免的问题

1. 不能把 cluster 当结论本身
2. 不能把 overlapping constructs 当完全独立外部验证
3. 不能把 cross-sectional ordering 写成真实 disease trajectory
4. 不能用“AI”本身替代医学问题

## G. 我们论文应该模仿的叙事和应避免的叙事

### 应模仿

- `WHO/ICOPE 问题锚点 -> 现有分型不足 -> 连续 IC + 表征学习 -> clinically meaningful subtype`
- `强基线同台 -> 主模型优势克制表达 -> 敏感性分析解释边界`

### 应避免

- `我们用了 deep/LLM，所以一定更先进`
- `4 个簇很好看，所以一定更真实`
- `横断面排序 = 真实进展路径`

## 复盘与反思

阶段 2 最关键的收获是：主投期刊不应由“哪本期刊等级最高”决定，而应由“哪本期刊最能接住我们这种有方法升级、但必须非常克制解释的横断面功能分型稿”决定。综合看，JGSA 是最合适的平衡点；Age and Ageing 虽然诱人，但当前证据强度并不匹配。

## 自动进入下一阶段

已进入阶段 3：围绕 JGSA 风格与当前最稳定位，设计论文蓝图。

## 本阶段外部来源

- JGSA About: https://academic.oup.com/biomedgerontology/pages/About
- JGSA Author Guidelines: https://academic.oup.com/biomedgerontology/pages/Author_Guidelines
- Maturitas Aims and Scope: https://www.sciencedirect.com/journal/maturitas/about/aims-and-scope
- Maturitas Guide for Authors: https://www.sciencedirect.com/journal/maturitas/publish/guide-for-authors
- Age and Ageing About: https://academic.oup.com/ageing/pages/About
- Archives of Gerontology and Geriatrics Guide for Authors: https://www.sciencedirect.com/journal/archives-of-gerontology-and-geriatrics/publish/guide-for-authors

