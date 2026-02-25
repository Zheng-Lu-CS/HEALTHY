# Workflow Report

## Purpose
This report documents the end-to-end pipeline workflow, outputs, and key findings from the current run.

## Workflow Overview
1. Load raw XLSX and parse coding doc (DOCX).
2. Normalize explicit missing tokens (ND/UK/NA).
3. D1 data audit: missingness, outliers, category and region distributions.
4. D2 IC proxy scoring: 5 domains, total score, level mapping.
5. D3 embedding + clustering + profiles + UMAP + radar.
6. D4 feature importance (IC + clusters) with SHAP summaries.
7. LLM-adjacent outputs (sample cards, rule-based cluster names).

## Key Files and Meaning
| Path | Meaning |
|---|---|
| `data/*.xlsx` | Raw survey data |
| `data/??.docx` | Codebook for categorical values |
| `figures/*.png` | All visual figures (D1?D4) |
| `reports/D1_data_audit_report.md` | Data quality audit report |
| `reports/D2_IC_rules.md` | IC proxy rule specification |
| `reports/D3_cluster_names.md` | Rule-based cluster naming |
| `figures/FIGURE_NOTES.md` | Text+math explanations for all figures |
| `reports/TECHNICAL_REPORT.md` | Font fix + structure change report |
| `outputs/D2_with_IC.csv` | Raw data + IC domain flags + totals |
| `outputs/D3_embeddings.parquet` | Learned embeddings (32D) |
| `outputs/D3_cluster_assignments.csv` | Cluster ID per sample |
| `outputs/D3_cluster_profiles.xlsx` | Cluster summary tables |
| `outputs/D4_feature_importance_IC.csv` | SHAP feature importance for IC |
| `outputs/D4_feature_importance_cluster.csv` | SHAP feature importance for clusters |
| `outputs/sample_card.txt` | Template-based sample cards |

## Data Snapshot
- Samples (n): 6025
- Features (p): 361

## Missingness Summary
Missing-rate bins (columns):
| bin     |   columns |
|:--------|----------:|
| <=5%    |       206 |
| 5-20%   |         9 |
| 20-50%  |        44 |
| 50-80%  |        27 |
| 80-95%  |        24 |
| 95-100% |        51 |

## IC Distribution
IC_total (0?5) distribution:
|   IC_total |   count |   proportion |
|-----------:|--------:|-------------:|
|          0 |     404 |    0.0670539 |
|          1 |    1467 |    0.243485  |
|          2 |    1773 |    0.294274  |
|          3 |    1313 |    0.217925  |
|          4 |     729 |    0.120996  |
|          5 |     339 |    0.0562656 |

IC_level distribution:
| IC_level   |   count |   proportion |
|:-----------|--------:|-------------:|
| Medium     |    3086 |     0.512199 |
| Low        |    1871 |     0.310539 |
| High       |    1068 |     0.177261 |

## Cluster Sizes
|   cluster_id |   count |   proportion |
|-------------:|--------:|-------------:|
|            0 |    1415 |   0.234855   |
|            1 |     843 |   0.139917   |
|            2 |    1676 |   0.278174   |
|            3 |     459 |   0.0761826  |
|            4 |    1595 |   0.26473    |
|            5 |      37 |   0.00614108 |

## Top Features for IC_level (SHAP)
| feature                                                                                                      |   mean_abs_shap |
|:-------------------------------------------------------------------------------------------------------------|----------------:|
| 生命质量评估-12）在过去4个星期里，有多少时间由于您身体健康或情绪问题而妨碍您的社交活动（比如探亲、访友等）？ |        1.76106  |
| Fried衰弱表型评估-握力左手第一次                                                                             |        1.60169  |
| 区=东城区                                                                                                    |        1.12315  |
| 跌倒评估-总分                                                                                                |        0.892021 |
| 省=广西壮族自治区                                                                                            |        0.683094 |
| Fried衰弱表型评估-握力左手第二次                                                                             |        0.634494 |
| 区=Other                                                                                                     |        0.481909 |
| 省=新疆维吾尔自治区                                                                                          |        0.383338 |
| 区=中原区                                                                                                    |        0.27877  |
| Fried衰弱表型评估-握力右手第一次值                                                                           |        0.167607 |
| Fried衰弱表型评估-握力左手最大值                                                                             |        0.166246 |
| 肌少症评估-1举起或搬运10磅物体（约45kg）是否存在困难？                                                       |        0.150101 |
| 文化程度=初中                                                                                                |        0.146401 |
| 文化程度=高中/中专                                                                                           |        0.143215 |
| 文化程度=大专/本科                                                                                           |        0.134894 |

## Top Features for Cluster ID (SHAP)
| feature                                   |   mean_abs_shap |
|:------------------------------------------|----------------:|
| Fried衰弱表型评估-握力左手最大值          |        1.73384  |
| Fried衰弱表型评估-4米行走第二次           |        1.67676  |
| 是否帕金森=0.0                            |        0.894368 |
| 听力障碍是否影响日常=是                   |        0.53661  |
| 运动-双脚前后成一直线站立10秒-保持时间为: |        0.451883 |
| 家庭类型=1.0                              |        0.432437 |
| 高血压发病年数=25.0                       |        0.408816 |
| Fried衰弱表型评估-握力右手第二次值        |        0.404792 |
| 是否白内障=0.0                            |        0.368033 |
| 健康状态=一般                             |        0.34109  |
| 运动-总分                                 |        0.326932 |
| 是否前列腺病（问男）=1.0                  |        0.289867 |
| 区=鼓楼区                                 |        0.287158 |
| 运动-双脚半前后位站立10秒-保持时间为:     |        0.232025 |
| 受试来源=1.0                              |        0.231719 |

## What We Can Conclude Now
- We have a complete, reproducible cross-sectional IC proxy pipeline without longitudinal outcomes.
- IC domain impairments and total IC scores can stratify participants by functional risk at baseline.
- Unsupervised clustering reveals multiple subtypes with distinct IC domain profiles (see radar plot + profiles).
- Feature importance identifies variables most associated with IC level and subtype separation (see SHAP tables).
- These outputs support a report/paper: data quality, proxy IC scoring, subtype discovery, and interpretability.
