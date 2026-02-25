# HEALTHY Project

This repository contains a reproducible pipeline for cross-sectional IC proxy scoring, embedding, clustering, and factor interpretation.

## Structure
- `data/` raw input files (XLSX, DOCX, PDF)
- `scripts/` pipeline scripts
- `configs/` configuration outputs
- `reports/` markdown reports (D1–D3 + technical notes)
- `outputs/` data artifacts (CSV/Parquet/XLSX)
- `figures/` generated figures (PNG)
- `scratch/` temporary developer artifacts

## Quick Start
```powershell
python scripts\run_pipeline.py
```

## Visualization (Font-Safe)
```powershell
python scripts\visualize_figures.py
python scripts\describe_figures.py
```

## Outputs
- `reports/D1_data_audit_report.md`
- `reports/D2_IC_rules.md`
- `reports/D3_cluster_names.md`
- `reports/WORKFLOW_REPORT.md`
- `reports/TECHNICAL_REPORT.md`
- `outputs/D2_with_IC.csv`
- `outputs/D3_embeddings.parquet`
- `outputs/D3_cluster_assignments.csv`
- `outputs/D3_cluster_profiles.xlsx`
- `outputs/D3_cluster_metrics.csv`
- `outputs/D4_feature_importance_IC.csv`
- `outputs/D4_feature_importance_cluster.csv`
- `outputs/sample_card.txt`
- `figures/*.png`
- `figures/FIGURE_NOTES.md`

## Notes
- Chinese fonts are auto-selected to avoid garbled plot labels.
- Random seeds are fixed for reproducibility.
