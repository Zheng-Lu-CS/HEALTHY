# SHAP_INTERPRETATION_GUIDE

## What is SHAP?
SHAP (SHapley Additive exPlanations) assigns each feature a contribution to the model's prediction, based on game-theoretic Shapley values.

## Mean |SHAP|
mean(|SHAP|) = (1/m) * Σ_i |SHAP_{i,f}|. Larger values indicate stronger overall influence.

## Scale Caution
SHAP values are on the model-output scale (e.g., log-odds or probability). They are not fixed to 0–1. Compare within the same model only.

## Interpretation Boundaries
High importance for region or center indicates association, not causality. It may reflect center composition, sampling, or true underlying differences.

## Directional Explanations
See `reports/SHAP_DEPENDENCE_SUMMARY.md` and `figures/D4_dependence_IC_*.png` for directionality plots.