# CLUSTER_MODEL_CARD

## Embedding
- Source: outputs/D3_embeddings.parquet (32D AutoEncoder embedding)

## K Selection
- Metrics used: silhouette, Calinski–Harabasz, stability ARI.
- K chosen: best within 3–6 by silhouette + stability.

|   k |   silhouette |   calinski_harabasz |   stability_ari |
|----:|-------------:|--------------------:|----------------:|
|   2 |    0.199181  |             776.77  |        0.995173 |
|   3 |    0.0816978 |             601.844 |        0.999698 |
|   4 |    0.0881538 |             522.171 |        0.881405 |
|   5 |    0.0813431 |             481.29  |        0.933518 |
|   6 |    0.0891018 |             460.492 |        0.973434 |
|   7 |    0.0904192 |             436.236 |        0.940066 |
|   8 |    0.0912913 |             403.459 |        0.765659 |
|   9 |    0.095241  |             389.228 |        0.731251 |
|  10 |    0.0950726 |             376.354 |        0.872488 |

## Davies–Bouldin Index (lower is better)
|   k |   davies_bouldin |
|----:|-----------------:|
|   2 |          2.50757 |
|   3 |          2.59697 |
|   4 |          2.57905 |
|   5 |          2.58866 |
|   6 |          2.35214 |
|   7 |          2.113   |
|   8 |          2.08317 |
|   9 |          2.09167 |
|  10 |          1.99533 |

## Cluster Sizes
|   cluster_id |   count |
|-------------:|--------:|
|            0 |    1415 |
|            1 |     843 |
|            2 |    1676 |
|            3 |     459 |
|            4 |    1595 |
|            5 |      37 |

## Small-Cluster Policy
- Clusters with n<50 should be treated as rare/edge subtypes.
- Avoid causal over-interpretation; consider merging by distance if needed for reporting.