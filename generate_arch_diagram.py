#!/usr/bin/env python
"""Generate System Architecture Diagram for METR-LA Traffic Prediction Project"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle

# Configuration
OUT = r'c:\Users\LOQ 15IRX9\Downloads\DL_Project_2\paper_output\figures'
os.makedirs(OUT, exist_ok=True)

# Create Figure
fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis('off')
fig.patch.set_facecolor('white')

def draw_box(ax, x, y, w, h, text, fc='#E3F2FD', ec='#0033CC', fs=10, bold=True):
    """Draw a rounded box with text"""
    r = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.1', 
                       fc=fc, ec=ec, lw=2, zorder=1)
    ax.add_patch(r)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, 
            fontweight='bold' if bold else 'normal', multialignment='center', zorder=2)

def draw_arrow(ax, x1, y1, x2, y2, color='#555', style='->', width=1.5):
    """Draw an arrow between two points"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=width, zorder=0))

# ===== TITLE =====
ax.text(8, 10.6, 'Spatio-Temporal Traffic Flow Prediction System Architecture', 
        fontsize=18, fontweight='bold', ha='center')
ax.text(8, 10.15, 'METR-LA Dataset | 207 Sensors | 12-Step History → 1-Step Prediction', 
        fontsize=11, ha='center', style='italic', color='#666666')

# ===== LAYER 1: RAW DATA =====
draw_box(ax, 2, 9, 3.2, 0.75, 'Raw METR-LA Data\n207 × 34,272 timesteps\n(119 days, 5-min intervals)', 
         fc='#E3F2FD', ec='#0033CC', fs=9.5)

# ===== LAYER 2: PREPROCESSING =====
preproc_x = [4.5, 7, 9.5, 12]
preproc_labels = ['Missing Value\nHandling\n(Fwd/Bwd Fill)', 
                  'Normalization\n(MinMax)\n[0, 1]', 
                  'Train/Val/Test\nSplit\n(70/10/20)',
                  'Sliding Window\nSequencing\n(T_in=12, T_out=1)']
preproc_colors = ['#FFE0B2', '#FFE0B2', '#FFE0B2', '#FFE0B2']
preproc_edges = ['#FF6F00', '#FF6F00', '#FF6F00', '#FF6F00']

for x, label, fc, ec in zip(preproc_x, preproc_labels, preproc_colors, preproc_edges):
    draw_box(ax, x, 8.2, 2.2, 0.9, label, fc=fc, ec=ec, fs=8.5)
    draw_arrow(ax, x-1.1, 8.2, x-0.6, 8.2, color=ec)

draw_arrow(ax, 2, 8.6, 3.5, 8.6, color='#FF6F00')

# ===== LAYER 3: PROCESSED DATASETS =====
datasets = [
    (3.5, 'Training Dataset\n[23,978 × 12 × 207]\n70% of data'),
    (8, 'Validation Dataset\n[3,425 × 12 × 207]\n10% of data'),
    (12.5, 'Test Dataset\n[6,853 × 12 × 207]\n20% of data')
]

for x, label in datasets:
    draw_box(ax, x, 6.8, 2.5, 0.95, label, fc='#C8E6C9', ec='#2E7D32', fs=8.5)
    draw_arrow(ax, x, 7.7, x, 7.3, color='#2E7D32')

# ===== LAYER 4: SIX MODELS =====
model_specs = [
    (0.9, 'LSTM\n(From Scratch)\n138,191 params\n2 layers, 256-dim', '#BBDEFB', '#0033CC'),
    (2.5, 'GRU\n(From Scratch)\n112,399 params\n2 layers, 256-dim', '#BBDEFB', '#0033CC'),
    (4.1, 'STFormer ⭐\n(From Scratch)\n658,895 params\nDual-branch\nGraph-free Attn', '#FFCCBC', '#FF6F00'),
    (6.2, 'PyTorch\nTransformer\n(Library)\n131,023 params\nTemporal-only', '#FFCCBC', '#FF6F00'),
    (8.8, 'XGBoost\n(Library)\n~50K params\n200 trees\nGPU-enabled', '#F8BBD0', '#C2185B'),
    (10.8, 'Random\nForest\n(Library)\n~30K params\n100 trees\nParallel', '#F8BBD0', '#C2185B'),
]

for x, label, fc, ec in model_specs:
    draw_box(ax, x, 5.5, 1.4, 1.2, label, fc=fc, ec=ec, fs=7.5)
    draw_arrow(ax, x, 6.3, x, 5.95, color=ec)

# ===== LAYER 5: TRAINING & EVALUATION =====
draw_box(ax, 3.5, 4, 3.2, 1.0, 
         'Training Process\n• Optimizer: Adam\n• Loss: MSE\n• Scheduler: ReduceLROnPlateau\n• Gradient Clip: 1.0\n• Batch: 32, Epochs: 100',
         fc='#E3F2FD', ec='#1565C0', fs=7.5)

draw_box(ax, 8.5, 4, 3.2, 1.0,
         'Evaluation Metrics\n• MAE (Mean Abs Error)\n• RMSE (Root Mean Sq)\n• MAPE (Mean Abs %)\n• R² (Coeff Determ)\n• (Inverse-transformed)',
         fc='#E3F2FD', ec='#1565C0', fs=7.5)

# Arrows from models to training/evaluation
for x in [0.9, 2.5, 4.1]:
    draw_arrow(ax, x, 5.0, 3.5, 4.45, color='#1565C0', width=1.2)
for x in [6.2, 8.8, 10.8]:
    draw_arrow(ax, x, 5.0, 8.5, 4.45, color='#1565C0', width=1.2)

# ===== LAYER 6: RESULTS =====
results_text = ('🏆 FINAL RESULTS (Test Set - 1-Step Prediction):\n'
                '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
                '① XGBoost (Best): MAE 3.44 mph | RMSE 7.39 | MAPE 6.63% | R² 0.895\n'
                '② STFormer (Best DL): MAE 4.55 mph | RMSE 8.65 | MAPE 9.28% | R² 0.856 ✓ 10.4% vs LSTM\n'
                '③ Transformer: MAE 4.83 mph | GRU: MAE 4.99 mph | LSTM: MAE 5.08 mph | RF: MAE 5.77 mph\n'
                '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
                '📊 Key Findings: Spatial attention +5.8% | Transformers > RNNs | 1-step recency dominant')

draw_box(ax, 8, 1.5, 14.5, 1.8, results_text, fc='#C8E6C9', ec='#1B5E20', fs=7.5, bold=False)

# Arrows to results
draw_arrow(ax, 3.5, 3.5, 6, 2.3, color='#1B5E20', width=2)
draw_arrow(ax, 8.5, 3.5, 10, 2.3, color='#1B5E20', width=2)

# ===== LEGEND =====
legend_y = 0.6
ax.text(1, legend_y + 0.3, 'Legend:', fontsize=10, fontweight='bold')

legend_items = [
    ('#BBDEFB', '#0033CC', 'RNNs (LSTM/GRU) — From Scratch'),
    ('#FFCCBC', '#FF6F00', 'Transformers / Novel Models'),
    ('#F8BBD0', '#C2185B', 'Ensemble Methods (XGBoost/RF)'),
]

for i, (fc, ec, desc) in enumerate(legend_items):
    y_pos = legend_y - (i * 0.25)
    rect = Rectangle((1, y_pos-0.08), 0.15, 0.15, fc=fc, ec=ec, lw=1.5, zorder=1)
    ax.add_patch(rect)
    ax.text(1.3, y_pos+0.015, desc, fontsize=8.5, va='center', zorder=2)

plt.tight_layout()

# Save PNG (300 DPI for publication quality)
png_path = os.path.join(OUT, 'system_architecture_diagram.png')
plt.savefig(png_path, bbox_inches='tight', dpi=300, facecolor='white')

# Save PDF (vector format for presentations)
pdf_path = os.path.join(OUT, 'system_architecture_diagram.pdf')
plt.savefig(pdf_path, bbox_inches='tight', dpi=300, facecolor='white')

plt.close()

# Print confirmation
print('✅ System Architecture Diagram generated successfully!')
print(f'   📊 PNG (300 DPI): {png_path}')
print(f'   📄 PDF (Vector):   {pdf_path}')
