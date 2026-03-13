# 正式分型后处理技术报告

## 1. 这一步在做什么
- 这份报告不是重新训练模型，而是在已有正式分型结果上做“可汇报版本”的后处理。
- 核心目标有三个：
  1. 重新用固定随机种子生成可复现的聚类标签，避免旧文件里出现标签不一致的问题。
  2. 把所有正式汇报图中的文字统一改成英文，彻底绕开中文字体显示问题。
  3. 生成一份中文技术说明，明确三条路线各自擅长什么、短板是什么、下一步该怎么推进。

## 2. 本次后处理使用的输入
- 连续化 IC 结果：`outputs/formal_subtyping/IC_continuous_scores.csv`
- 三条 128 维表征：`bert_embedding_128.npy`、`ft_embedding_128.npy`、`pca_embedding_128.npy`
- 最优 k 值来自已有正式实验：BERTText=4，FTTransformer=5，PCA=4
- 后处理统一用 `random_state=2026, n_init=20` 重新做 KMeans，并按 Overall IC 从低到高重排 cluster_id。

## 3. 连续化 IC 的含义
- 认知：使用 `认知-总分 / 30`，近似 MMSE 连续化。
- 心理：使用 `1 - 心理-总分 / 15`，再把焦虑抑郁诊断作为 15% 惩罚项。
- 活力：综合 `活力-总分`、营养描述、BMI 偏离 22 的程度、小腿围。
- 运动：综合 `运动-总分`、运动完成情况、步态异常、4 米步速、握力。
- 感官：综合听力/视力障碍、筛查结果、是否影响日常。
- 最终每个维度都是 0–100 分，总分是五域平均。这样做的价值是：不再把老人强行压成“受损/未受损”两类，而是保留程度信息。

## 4. 三条分型路线怎么理解
- `BERTText`：先把样本字段用 `编码.docx` 映射成更可读的医学文本，再用 `bert-base-chinese` 编码，最后压到 128 维。
- `FTTransformer`：直接对表格做 Transformer 风格的自监督重建，输出 128 维表示。
- `PCA`：对同一套核心特征做统一预处理后直接降到 128 维，作为强基线。

## 5. 统一比较结果
| method        |   k |   silhouette |   calinski_harabasz |   davies_bouldin |   stability_ari |   external_separation |   min_cluster_ratio |   max_cluster_ratio | cluster_sizes              |
|:--------------|----:|-------------:|--------------------:|-----------------:|----------------:|----------------------:|--------------------:|--------------------:|:---------------------------|
| PCA           |   4 |    0.0769941 |             469.252 |          2.77785 |        0.606424 |              0.234876 |            0.193361 |            0.299087 | 1165, 1802, 1382, 1676     |
| FTTransformer |   5 |    0.0721261 |             331.158 |          3.3602  |        0.996298 |              0.221359 |            0.138257 |            0.260249 | 928, 1223, 833, 1473, 1568 |
| BERTText      |   4 |    0.264103  |            8593.02  |          1.90421 |        0.99895  |              0.037909 |            0.239502 |            0.255436 | 1443, 1514, 1539, 1529     |

解释：
- 纯几何聚类质量最强的是 `BERTText`，它的 silhouette=0.2641。
- 外部临床区分度最强的是 `PCA`，external_separation=0.2349。
- 作为下一阶段主推深度学习路线，我推荐 `FTTransformer`。理由不是“它绝对第一”，而是它在外部临床分离度上已经接近最强基线，同时保留了明确的深度模型增量空间。

## 6. 结果怎么解读
- `BERTText`：簇边界最清楚，说明‘把硬编码字段翻译成自然语言再编码’这条路是有效的；但它对 ADL/IADL/肌少症/衰弱等外部量表的分离度偏低，说明当前文本路线更像语义整理，而不是最强临床分型主线。
- `FTTransformer`：silhouette 不高，但 external_separation 很强，而且 5 个簇都不是小簇，比较适合讲“临床异质性分层”。这条路线最适合作为你们后面冲 A 类论文的深度学习主线。
- `PCA`：基线依然很强，说明这批数据本身结构性很强。这个结果很重要，因为它逼着我们后面的 fancy 方法必须拿出真正的增益，而不是只换一个更复杂的模型名字。

## 7. BERTText 的分型画像
- Cluster 0：Overall IC=68.6。 这是中间型/过渡型簇。 主要特征：感官维度较差。 外部临床画像：Sarcopenia=1.62，Fried=1.18，ADL=91.8，IADL=11.0，Fall risk=31.4。
- Cluster 1：Overall IC=75.8。 这是中间型/过渡型簇。 外部临床画像：Sarcopenia=0.82，Fried=1.06，ADL=97.2，IADL=9.2，Fall risk=6.7。
- Cluster 2：Overall IC=76.1。 这是中间型/过渡型簇。 外部临床画像：Sarcopenia=1.73，Fried=1.40，ADL=92.9，IADL=10.2，Fall risk=30.7。
- Cluster 3：Overall IC=77.3。 这是中间型/过渡型簇。 外部临床画像：Sarcopenia=1.11，Fried=1.00，ADL=95.1，IADL=9.6，Fall risk=27.7。
- BERTText 的簇几何边界最清楚，但外部临床量表分离度偏低，说明纯文本语义目前更像‘语义整理器’，还不是最强的临床分型器。

## 7. FTTransformer 的分型画像
- Cluster 0：Overall IC=59.2。 这是最受损的整体低功能簇。 主要特征：运动能力明显偏弱，认知维度偏弱，心理维度偏弱。 外部临床画像：Sarcopenia=4.13，Fried=2.28，ADL=75.8，IADL=15.1，Fall risk=46.8。
- Cluster 1：Overall IC=71.5。 这是中间型/过渡型簇。 主要特征：感官维度较差。 外部临床画像：Sarcopenia=1.14，Fried=1.08，ADL=98.2，IADL=8.9，Fall risk=23.4。
- Cluster 2：Overall IC=74.3。 这是中间型/过渡型簇。 主要特征：运动能力保持较好，感官维度较差。 外部临床画像：Sarcopenia=0.70，Fried=0.72，ADL=94.8，IADL=10.3，Fall risk=32.9。
- Cluster 3：Overall IC=76.5。 这是中间型/过渡型簇。 外部临床画像：Sarcopenia=0.70，Fried=1.02，ADL=98.2，IADL=8.9，Fall risk=5.6。
- Cluster 4：Overall IC=84.2。 这是整体功能最高的一组。 主要特征：感官维度保持较好。 外部临床画像：Sarcopenia=0.70，Fried=1.00，ADL=98.4，IADL=8.5，Fall risk=23.5。
- FTTransformer 的分型更像一条‘整体功能下降 + 感官分化’的谱系，因此更适合做主线深度学习分型。

## 7. PCA 的分型画像
- Cluster 0：Overall IC=58.2。 这是最受损的整体低功能簇。 主要特征：运动能力明显偏弱，认知维度偏弱，心理维度偏弱。 外部临床画像：Sarcopenia=3.96，Fried=2.24，ADL=78.4，IADL=14.8，Fall risk=42.6。
- Cluster 1：Overall IC=73.8。 这是中间型/过渡型簇。 主要特征：运动能力保持较好，感官维度较差。 外部临床画像：Sarcopenia=0.79，Fried=0.81，ADL=97.1，IADL=9.3，Fall risk=27.5。
- Cluster 2：Overall IC=77.9。 这是中间型/过渡型簇。 外部临床画像：Sarcopenia=0.57，Fried=0.95，ADL=98.8，IADL=8.6，Fall risk=4.9。
- Cluster 3：Overall IC=83.9。 这是整体功能最高的一组。 主要特征：感官维度保持较好。 外部临床画像：Sarcopenia=0.72，Fried=1.03，ADL=98.3，IADL=8.5，Fall risk=23.7。
- PCA 是非常强的基线，说明这批数据本身已经带有较强的低秩结构。

## 8. 为什么主推 FTTransformer，而不是直接拿 PCA 发
- 因为现在项目的目标不是只做一个 baseline 分层，而是要走一条“深度学习/大模型辅助”的论文路线。
- PCA 必须保留，并且要如实承认它很强；这会让文章更可信。
- 但真正能继续往前扩展成 fancy 方法的，是 FTTransformer 这条线：
  1. 它可以自然接上缺失掩码预训练。
  2. 它可以接入 group-aware token、region token、disease token 等语义结构。
  3. 它可以和文本/BERT 表征做跨模态对比学习，而不是停留在单一表格模型。

## 9. 下一步最值得做的事
- 先把 FTTransformer 作为主线模型，做 group-aware 输入和 masked modeling。
- 再把 BERTText 从‘单独聚类’升级成‘文本-表格双塔对比学习’，而不是让文本路线单打独斗。
- 聚类评价不能只看 silhouette，必须同时汇报 external_separation、cluster size balance、以及关键外部量表的组间检验。
- 论文写法上要坚持：深度模型不是为了炫，而是为了更好地恢复被硬编码压扁的语义结构。
