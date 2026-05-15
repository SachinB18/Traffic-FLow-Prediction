# Image Optimization Guide for Enhanced Research Paper

## Overview

The enhanced paper recommends **4 key images** (instead of 6-7 in original) to meet the "few but on point" criterion while maintaining comprehensive visual support.

## Images to KEEP ✅

### 1. **fig6_eda_overview.png** - Exploratory Data Analysis (ESSENTIAL)
**Section**: Methodology (Dataset Description)  
**Line Reference**: Line 323-327  
**Purpose**: Data distribution and temporal patterns  
**Content should show**:
- Speed distribution histogram (left-skewed, median > mean)
- Hourly heatmap (24-hour rush-hour patterns)
- Weekly heatmap (weekday vs weekend differences)
- Key statistics overlay

**Why Essential**: 
- Validates dataset appropriateness for DL
- Shows temporal patterns (24h, 7d periodicity)
- Justifies preprocessing choices (non-stationarity)
- Cannot be easily described textually

**Rubric Mapping**: Q5(ii) - Dataset properties and characteristics

**Suggested Dimensions**: 
```latex
\includegraphics[width=\linewidth]{figures/fig6_eda_overview.png}
```
(Full page width for readability)

---

### 2. **fig4_stformer_architecture.png** - System Architecture (ESSENTIAL)
**Section**: System Architecture and Model Design  
**Line Reference**: Line 346  
**Purpose**: Visual representation of all 6 models and data flow  
**Content should show**:
- Input data (B×T×N tensors)
- Three DL branches: LSTM → hidden → output
- GRU → hidden → output  
- STFormer (dual-branch, spatial attention, fusion)
- Two tree model branches: feature flattening → predictions
- All labeled with layer dimensions

**Why Essential**:
- Clarifies architecture differences between models
- Shows parameter flow (dimensions at each layer)
- Critical for Q6 (Model Architecture requirement)
- Enables visual comparison of 6 approaches

**Rubric Mapping**: Q6 - Architecture diagram with all layers labeled

**Suggested Dimensions**:
```latex
\includegraphics[width=\linewidth]{figures/fig4_stformer_architecture.png}
```

**Recommendation**: If existing image shows only STFormer, create a composite that includes:
- Left side: STFormer detailed architecture (dual branches, spatial attention)
- Right side: LSTM/GRU/Transformer sketches for comparison
- Bottom: Tree model input/output flow

---

### 3. **fig2_training_curves.png** - Training Dynamics (ESSENTIAL)
**Section**: Results - Training Dynamics  
**Line Reference**: Line 616  
**Purpose**: Convergence comparison across DL models  
**Content should show**:
- X-axis: Epoch (0-100)
- Y-axis: MSE loss (log or linear scale)
- 4 lines: LSTM (solid blue), GRU (solid green), STFormer (solid red), Transformer (solid orange)
- Dashed versions for validation loss
- Red dotted vertical line: convergence epoch for each model
- Legend: "Train (solid)", "Val (dashed)"
- Annotations: "62 epochs", "49 epochs", "100 epochs", "56 epochs" with best val loss

**Why Essential**:
- Q8(ii) requirement: Loss curves with labeled axes
- Shows convergence speed differences (GRU fastest, STFormer slowest)
- Validates training stability and early stopping
- Cannot describe trade-offs textually

**Rubric Mapping**: Q8(ii) - Training and validation loss curves

**Suggested Dimensions**:
```latex
\includegraphics[width=\linewidth]{figures/fig2_training_curves.png}
```

**Critical Details**:
- Axes MUST be labeled: "Epoch" (x), "MSE Loss" (y)
- Color-coding consistent across figures
- Legend must distinguish train/val
- Grid lines recommended for readability

---

### 4. **fig1_model_comparison_bars.png** - Results Comparison (ESSENTIAL)
**Section**: Results - Overall Performance  
**Line Reference**: Line 599  
**Purpose**: Four-metric comparison across all 6 models  
**Content should show**:
- X-axis: Models (6 bars per metric group)
- Y-axis: Metric value (standardized scale, "lower is better" noted)
- Grouped bars (or subplots) for each metric:
  - MAE (mph): 3.44, 4.55, 4.83, 4.99, 5.08, 5.77
  - RMSE (mph): 7.39, 8.65, 9.62, 9.85, 9.94, 11.19
  - MAPE (%): 6.63, 9.28, 10.34, 10.30, 10.28, 12.33
  - R²: 0.895, 0.856, 0.822, 0.813, 0.810, 0.759
- Gold/highlighted border on STFormer (best DL)
- Different colors for: DL models (blues/reds), Ensemble (greens/grays)
- Legend identifying model types

**Why Essential**:
- Q8(v) requirement: Bar chart comparing baseline vs improved
- Shows all 4 required metrics simultaneously
- Enables quick visual assessment of trade-offs
- STFormer dominance among DL models apparent
- XGBoost overall winner visible

**Rubric Mapping**: Q8(v) - Bar chart of baseline vs improved metrics

**Suggested Dimensions**:
```latex
\includegraphics[width=0.8\linewidth]{figures/fig1_model_comparison_bars.png}
```
(Slightly narrower for 2-column layout)

**Recommended Format**:
- Subplots: 2×2 grid (one per metric)
- OR Single stacked/grouped bar chart with metric labels
- Ensure no bar cutoff at plot edges

---

## Images to REMOVE or CONVERT ❌

### ❌ **fig5_radar_chart.png** - REMOVE
**Reason**: Redundant with Fig. 1 (bar chart)  
**Content**: Normalized metrics in radar format  
**Why problematic**:
- Hard to read in print (small angles difficult to compare)
- Bar chart conveys same information more clearly
- Wastes space (roughly 1 full plot area)
- Not required by rubric

**Alternative**: Replace with text summary if radar adds insights:
```
"Radar analysis shows STFormer's balanced profile across metrics..."
```

---

### ❌ **fig7_network_structure.png** - REMOVE
**Reason**: Non-essential for core narrative  
**Content**: Sensor network topology (degree distribution, hub sensors)  
**Why problematic**:
- Network analysis not required by rubric
- Doesn't directly support any RQ (RQ1-3 focus on forecasting, not topology)
- Adds 1-2 pages without advancing key arguments
- EDA figure (fig6) already shows dataset overview

**Alternative**: If topology is important, integrate into fig6 or describe in text

---

### ❌ **fig3_hyperparam_sensitivity.png** - CONVERT TO TABLES
**Reason**: Better represented as Tables (already done: Table 7, Table 8)  
**Content**: Hidden dim sensitivity, learning rate sensitivity curves  
**Conversion**:
- Keep as Table 7 (LSTM hidden dim): 4 rows × 3 columns
- Keep as Table 8 (LR sensitivity): 5 rows × 3 columns
- Add brief textual analysis after each table

**Advantage of tables**:
- Exact values readable (not approximate from graph)
- Rubric Q9 explicitly asks for "table or graph"
- Tables save space vs. 2×2 subplot figure
- Easier to reference specific values in text

---

## UPDATED FIGURE LIST FOR FINAL PAPER

### Figures to Include (4 total):

1. **fig1_model_comparison_bars.png** (Page 8)
   - Location: Section 6.1 (Line 599)
   - Width: 0.8\linewidth (narrower for balance)
   - Caption: "Four-metric comparison across six models. Highlighting shows STFormer's dominance among DL approaches; XGBoost achieves lowest MAE overall on 1-step prediction."

2. **fig2_training_curves.png** (Page 9)
   - Location: Section 6.3 (Line 616)
   - Width: \linewidth (full width)
   - Caption: "Training (solid) and validation (dashed) MSE loss evolution. STFormer convergence slower initially but achieves lowest final loss; early stopping prevents overfitting across all models."

3. **fig4_stformer_architecture.png** (Page 5)
   - Location: Section 4.0 (Line 346)
   - Width: \linewidth (full width)
   - Caption: "Complete system architecture: Deep learning models (LSTM, GRU, STFormer, Transformer) process B×T×N temporal sequences; tree models (XGBoost, RF) operate on flattened 2,484-dimensional feature vectors."

4. **fig6_eda_overview.png** (Page 3)
   - Location: Section 3.1 Dataset (Line 327)
   - Width: \linewidth (full width)
   - Caption: "METR-LA dataset characteristics: (left) speed distribution showing left-skew (median > mean) indicating congestion events; (middle) hourly heatmap revealing 24-hour periodicity with rush-hour troughs; (right) weekly heatmap showing weekday vs. weekend differences."

---

## PAGE COUNT IMPACT

**With 4 figures**: ~8-10 pages  
**Original with 6-7 figures**: 10-12 pages  
**Optimized with tables**: Still ~9 pages (better balance)

---

## CAPTION REQUIREMENTS

Each caption must include:
1. **What**: Clear description of visualization
2. **Why**: How it supports rubric requirement
3. **Interpretation**: Key finding from the figure

**Example (Good):**
```
Fig. 1 - Four-metric comparison: STFormer (red bar, gold border) achieves best 
MAE among DL models (4.55 mph, 10.4% improvement over LSTM), but XGBoost 
(leftmost) dominates overall on 1-step prediction, highlighting horizon-dependent 
model selection.
```

**Example (Poor):**
```
Fig. 1 - Model comparison.
```

---

## FIGURE RESOLUTION & FORMAT

### Technical Specifications:
- **Format**: PNG or PDF (PDFs preferred for LaTeX)
- **Resolution**: 300 DPI minimum (print quality)
- **Color space**: RGB (not CMYK for screen display)
- **Compression**: Lossless PNG preferred

### Size Guidelines:
- **Single-column width**: Max 3.4 inches (8.6 cm)
- **Full-page width**: Max 7 inches (17.8 cm)
- **Height**: Keep under 5 inches (12.7 cm) to fit with caption

### File Size:
- Target: <1 MB per figure (total 4 figures < 4 MB)
- If >2 MB: Compress using:
  ```
  convert fig1.png -quality 85 -strip fig1_compressed.png
  ```

---

## VERIFICATION CHECKLIST

- [ ] All 4 figures present in `figures/` directory
- [ ] Filenames match LaTeX references exactly (case-sensitive)
- [ ] All axes labeled with units (e.g., "MAE (mph)", "Epoch")
- [ ] Legends present and legible
- [ ] Color-blind safe (avoid red-green only schemes)
- [ ] Resolution adequate for printing (300 DPI)
- [ ] Captions match rubric requirements
- [ ] Cross-references in text (e.g., "Figure 1 shows...")
- [ ] No placeholder/Lorem ipsum text in figures
- [ ] All figures cited in main text (not orphaned)
- [ ] Table placement doesn't break section flow

---

## INTEGRATION WITH PAPER TEXT

### Check Cross-References:
```latex
% Line 599: First mention of fig1
See Figure~\ref{fig:bars} for metric comparison.

% Line 616: First mention of fig2
Figure~\ref{fig:curves} depicts convergence patterns.

% Line 327: First mention of fig6
Figure~\ref{fig:eda} illustrates temporal periodicity.

% Line 346: First mention of fig4
Figure~\ref{fig:sysarch} presents the complete architecture.
```

### Ensure Labels Match:
- `\label{fig:bars}` for fig1
- `\label{fig:curves}` for fig2
- `\label{fig:eda}` for fig6
- `\label{fig:sysarch}` for fig4

---

## FINAL SUMMARY

| Figure | Rubric Q | Page | Status |
|--------|----------|------|--------|
| fig1_comparison_bars | Q8(v) | 8 | ✅ KEEP |
| fig2_training_curves | Q8(ii) | 9 | ✅ KEEP |
| fig4_architecture | Q6 | 5 | ✅ KEEP |
| fig6_eda_overview | Q5(iv) | 3 | ✅ KEEP |
| fig5_radar_chart | -- | -- | ❌ REMOVE |
| fig7_network_structure | -- | -- | ❌ REMOVE |
| fig3_hyperparam_tables | Q9 | 7 | ✅ TABLES |

**Result**: 4 figures + 2 tables = Optimal "few but on point" coverage

