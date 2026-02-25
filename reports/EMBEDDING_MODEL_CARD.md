# EMBEDDING_MODEL_CARD

## Input Features
- Total features after encoding: 989
- Numeric features: 57
- Categorical features (one-hot): 241

## Preprocessing
- Numeric: z-score standardization (mean=0, std=1).
- Categorical: one-hot encoding; rare categories (<0.5%) -> 'Other'.
- Missing: numeric median; categorical 'Missing'.

## Model
- AutoEncoder (MLP)
- Encoder: input -> 256 -> 128 -> latent
- Decoder: latent -> 128 -> 256 -> input
- Latent dimension: 32
- Objective: reconstruction (MSE)
- Epochs: 40; batch size: 256; lr: 1e-3
- Random seed: 42

## Stability (Current)
- Cluster stability is evaluated via ARI across multiple random seeds in KMeans (see D3_cluster_metrics.csv).
- Full embedding stability across multiple AutoEncoder trainings is not yet computed in this run.