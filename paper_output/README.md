# Research Paper — Compilation Instructions

## Files Included
- `main.tex`         — Full IEEE journal LaTeX source (IEEEtran class)
- `figures/`         — All figures referenced in the paper (PNG + PDF)
  - fig1_model_comparison_bars  — 4-metric bar chart (all models)
  - fig2_training_curves        — Loss curves (LSTM, GRU, STFormer, Transformer)
  - fig3_hyperparam_sensitivity — Hidden dim & LR sensitivity
  - fig4_stformer_architecture  — STFormer architecture diagram
  - fig5_radar_chart            — Multi-metric radar comparison
  - fig6_eda_overview           — METR-LA dataset EDA overview
  - fig7_network_structure      — Sensor network structure
  - fig8_temporal_decomp        — Temporal decomposition & data quality

## Compilation (requires TeX Live / MiKTeX)

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use Overleaf:
1. Upload main.tex and the figures/ folder.
2. Set compiler to pdfLaTeX.
3. Click Compile.

## IEEEtran Class
Download from: https://www.ctan.org/pkg/ieeetran
Or available by default in any modern LaTeX distribution.

## Real Metrics Used
| Model          | MAE   | RMSE  | MAPE  | R2    |
|----------------|-------|-------|-------|-------|
| LSTM           | 5.077 | 9.942 | 10.28 | 0.810 |
| GRU            | 4.986 | 9.846 | 10.30 | 0.813 |
| STFormer       | 4.552 | 8.653 | 9.28  | 0.856 |
| PyTorch Trans. | 4.829 | 9.622 | 10.34 | 0.822 |
| XGBoost        | 3.438 | 7.387 | 6.63  | 0.895 |
| Random Forest  | 5.774 | 11.19 | 12.33 | 0.759 |
