# Technical Report: Fixing Plot Legend Garbling and Project Restructure

## Summary
- Issue: Most plot legends and Chinese labels were garbled or missing in PNG outputs.
- Root cause: Matplotlib default font did not include CJK glyphs on this machine.
- Fix: Enforced CJK-capable font selection and standardized plotting style before rendering.
- Result: All plots regenerated with proper Chinese text rendering and visible legends.

## Environment
- OS: Windows
- Working directory: `c:\Users\华为\Desktop\深度学习\系统学习2026\project\HEALTHY`
- Pipeline script: `scripts/run_pipeline.py`

## Font Detection and Selection
- Preferred fonts list: Microsoft YaHei, SimHei, Noto Sans CJK SC, Source Han Sans SC, PingFang SC, Arial Unicode MS
- Installed candidates found: Microsoft YaHei, SimHei
- Selected font: Microsoft YaHei

## Implementation Details
- Added `configure_plot_style()` to select a CJK-capable font.
- Set `axes.unicode_minus = False` to avoid minus-sign issues.
- Applied a consistent seaborn theme.
- Called `configure_plot_style()` at pipeline startup before any plotting.
- Regenerated all figures with the new settings.

## Files Updated
- `scripts/run_pipeline.py`
- `README.md`

## Project Structure Changes
- Added `reports/` for narrative outputs.
- Added `outputs/` for data artifacts.
- Kept `figures/` for PNG outputs.
- Kept `configs/` for configuration files.
- Kept `scripts/` for pipeline code.
- Kept `data/` for raw inputs.
- Added `scratch/` for temporary developer artifacts.

## Output Regeneration
- All figures were regenerated using the selected CJK font.
- Location: `figures/*.png`

## Verification Checklist
- D1 bar charts show Chinese titles and labels correctly.
- D3 radar chart and UMAP figure display legends and labels without garbling.
- D4 SHAP summaries render axis labels and legends correctly.

## Optional Next Steps
- Add an English-only export mode for plots.
- Export vector figures (SVG/PDF) for publication.
