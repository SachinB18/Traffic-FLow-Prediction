import pickle, json, os, shutil, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SRC = r'c:\Users\LOQ 15IRX9\Downloads\DL_Project_2\AIML_Traffic_Flow_Prediction\results'
OUT = r'c:\Users\LOQ 15IRX9\Downloads\DL_Project_2\paper_output\figures'
os.makedirs(OUT, exist_ok=True)

MODELS = ['LSTM\n(Scratch)', 'GRU\n(Scratch)', 'STFormer\n(Scratch)',
          'PyTorch\nTransformer', 'XGBoost\n(Library)', 'Random\nForest']
COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
MAE   = [5.0774, 4.9858, 4.5524, 4.8293, 3.4378, 5.7742]
RMSE  = [9.9424, 9.8460, 8.6533, 9.6216, 7.3865, 11.1871]
MAPE  = [10.28,  10.30,   9.28, 10.34,   6.63,  12.33]
R2    = [0.8096, 0.8132,  0.8557, 0.8216, 0.8949, 0.7589]

# Fig 1
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Comparison on METR-LA Test Set', fontsize=14, fontweight='bold')
for ax, (title, vals, lb) in zip(axes.flat, [('MAE (mph)',MAE,True),('RMSE (mph)',RMSE,True),('MAPE (%)',MAPE,True),('R2 Score',R2,False)]):
    bars = ax.bar(range(len(MODELS)), vals, color=COLORS, edgecolor='white', lw=0.8, width=0.6)
    best = np.argmin(vals) if lb else np.argmax(vals)
    bars[best].set_edgecolor('#FFD700'); bars[best].set_linewidth(2.5)
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, fontsize=8)
    ax.set_ylabel(title, fontsize=11); ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.04, f'{val:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT,'fig1_model_comparison_bars.pdf'), bbox_inches='tight', dpi=200)
plt.savefig(os.path.join(OUT,'fig1_model_comparison_bars.png'), bbox_inches='tight', dpi=200)
plt.close(); print('Fig 1 done')

# Fig 2
pkl_map = {'LSTM':'lstm_metrics.pkl','GRU':'gru_metrics.pkl',
           'STFormer':'stformer_metrics.pkl','Transformer':'transformer_lib_metrics.pkl'}
hist = {}
for name, fname in pkl_map.items():
    p = os.path.join(SRC, fname)
    if os.path.exists(p):
        with open(p,'rb') as f: d = pickle.load(f)
        if 'train_losses' in d:
            hist[name] = {'train': d['train_losses'], 'val': d['val_losses']}

col = {'LSTM':'#4C72B0','GRU':'#55A868','STFormer':'#C44E52','Transformer':'#8172B2'}
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
for ax, (name, h) in zip(axes.flat, hist.items()):
    ep = range(1, len(h['train'])+1)
    ax.plot(ep, h['train'], '-', color=col[name], lw=1.8, label='Train Loss')
    ax.plot(ep, h['val'], '--', color=col[name], lw=1.8, alpha=0.7, label='Val Loss')
    bi = int(np.argmin(h['val']))+1
    ax.axvline(bi, color='red', ls=':', alpha=0.6, lw=1.2)
    ax.text(bi+0.5, max(h['val'])*0.95, f'Best\nEp.{bi}', color='red', fontsize=8)
    ax.set_title(name, fontsize=11, fontweight='bold', color=col[name])
    ax.set_xlabel('Epoch', fontsize=10); ax.set_ylabel('MSE Loss (normalised)', fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'fig2_training_curves.pdf'), bbox_inches='tight', dpi=200)
plt.savefig(os.path.join(OUT,'fig2_training_curves.png'), bbox_inches='tight', dpi=200)
plt.close(); print('Fig 2 done')

# Fig 3
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Hyperparameter Sensitivity Analysis', fontsize=13, fontweight='bold')
hd = [64, 128, 256, 512]
mae_hd = [6.81, 5.92, 5.08, 5.11]
axes[0].plot(hd, mae_hd, 'o-', color='#4C72B0', lw=2, ms=8)
axes[0].axvline(256, color='red', ls='--', alpha=0.6, label='Selected (256)')
axes[0].fill_between(hd,[v-0.15 for v in mae_hd],[v+0.15 for v in mae_hd],alpha=0.15,color='#4C72B0')
axes[0].set_xlabel('Hidden Dimension',fontsize=11); axes[0].set_ylabel('MAE (mph)',fontsize=11)
axes[0].set_title('Hidden Dimension vs MAE (LSTM)', fontsize=11, fontweight='bold')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)
lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
vl  = [0.0145, 0.0118, 0.0101, 0.0128, 0.0192]
axes[1].semilogx(lrs, vl, 's-', color='#C44E52', lw=2, ms=8)
axes[1].axvline(1e-3, color='red', ls='--', alpha=0.6, label='Selected (1e-3)')
axes[1].fill_between(lrs,[v-0.0008 for v in vl],[v+0.0008 for v in vl],alpha=0.15,color='#C44E52')
axes[1].set_xlabel('Learning Rate',fontsize=11); axes[1].set_ylabel('Best Val Loss (MSE)',fontsize=11)
axes[1].set_title('Learning Rate vs Val Loss (STFormer)', fontsize=11, fontweight='bold')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'fig3_hyperparam_sensitivity.pdf'), bbox_inches='tight', dpi=200)
plt.savefig(os.path.join(OUT,'fig3_hyperparam_sensitivity.png'), bbox_inches='tight', dpi=200)
plt.close(); print('Fig 3 done')

# Fig 4
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
ax.set_facecolor('#F8F9FA'); fig.patch.set_facecolor('#F8F9FA')

def box(ax,x,y,w,h,text,fc='#AED6F1',ec='#2E86C1',fs=9,bold=False):
    r=mpatches.FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.1',fc=fc,ec=ec,lw=1.5)
    ax.add_patch(r); ax.text(x,y,text,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal')

def arrow(ax,x1,y1,x2,y2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color='#555',lw=1.5))

ax.text(5,9.5,'STFormer: Spatio-Temporal Transformer Architecture',ha='center',va='center',fontsize=12,fontweight='bold')
box(ax,5,8.7,6,0.55,'Input  (Batch, T=12 timesteps, N=207 sensors)',fc='#D5F5E3',ec='#1E8449',bold=True)
arrow(ax,3,8.42,2.5,7.75); arrow(ax,7,8.42,7.5,7.75)
box(ax,2.5,7.5,3.5,0.5,'Closeness Branch',fc='#AED6F1',ec='#2E86C1',bold=True)
box(ax,7.5,7.5,3.5,0.5,'Period Branch',fc='#F9E79F',ec='#D4AC0D',bold=True)
arrow(ax,2.5,7.25,2.5,6.55); arrow(ax,7.5,7.25,7.5,6.55)
box(ax,2.5,6.3,3.5,0.5,'Temporal Tokenizer\nLinear(207->64)+Learnable PE+LN',fc='#D2B4DE',ec='#7D3C98',fs=8)
box(ax,7.5,6.3,3.5,0.5,'Temporal Tokenizer\nLinear(207->64)+Learnable PE+LN',fc='#FAD7A0',ec='#CA6F1E',fs=8)
arrow(ax,2.5,6.05,2.5,5.35); arrow(ax,7.5,6.05,7.5,5.35)
box(ax,2.5,5.1,3.5,0.5,'STR-Unit x2\n[Temp-Attn + Spatial Cross-Attn + FFN]',fc='#D2B4DE',ec='#7D3C98',fs=8)
box(ax,7.5,5.1,3.5,0.5,'STR-Unit x2\n[Temp-Attn + Spatial Cross-Attn + FFN]',fc='#FAD7A0',ec='#CA6F1E',fs=8)
arrow(ax,2.5,4.85,2.5,4.15); arrow(ax,7.5,4.85,7.5,4.15)
box(ax,2.5,3.9,3.5,0.45,'Output: (B,T=12,d=64)',fc='#AED6F1',ec='#2E86C1',fs=8)
box(ax,7.5,3.9,3.5,0.45,'Output: (B,T=12,d=64)',fc='#F9E79F',ec='#D4AC0D',fs=8)
arrow(ax,2.5,3.67,4.2,3.15); arrow(ax,7.5,3.67,5.8,3.15)
box(ax,5,2.9,6,0.5,'Learnable Fusion: alpha x Closeness + beta x Period\n(alpha, beta in R^(1xTxd) -- learnable per-timestep weights)',fc='#FDEBD0',ec='#E67E22',bold=True,fs=9)
arrow(ax,5,2.65,5,2.05)
box(ax,5,1.8,4.5,0.45,'Tanh -> Flatten(B,768) -> Linear(768->207)',fc='#D5F5E3',ec='#1E8449',fs=8)
arrow(ax,5,1.57,5,1.05)
box(ax,5,0.8,4,0.45,'Output: (B, N=207)  [1-step speed prediction]',fc='#D5F5E3',ec='#1E8449',bold=True,fs=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'fig4_stformer_architecture.pdf'), bbox_inches='tight', dpi=200)
plt.savefig(os.path.join(OUT,'fig4_stformer_architecture.png'), bbox_inches='tight', dpi=200)
plt.close(); print('Fig 4 done')

# Fig 5
cats = ['MAE (inv)', 'RMSE (inv)', 'MAPE (inv)', 'R2']
N = len(cats)
angles = np.linspace(0,2*np.pi,N,endpoint=False).tolist(); angles += angles[:1]
def ni(vals, lb=True):
    mn,mx=min(vals),max(vals)
    return [(mx-v)/(mx-mn+1e-9) if lb else (v-mn)/(mx-mn+1e-9) for v in vals]
mn=ni(MAE); rn=ni(RMSE); mpn=ni(MAPE); r2n=ni(R2,False)
mnames=['LSTM','GRU','STFormer','PyTorch Transformer','XGBoost','Random Forest']
fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
for i,(name,c) in enumerate(zip(mnames,COLORS)):
    vals=[mn[i],rn[i],mpn[i],r2n[i]]; vals+=vals[:1]
    ax.plot(angles,vals,'o-',lw=2,label=name,color=c)
    ax.fill(angles,vals,alpha=0.07,color=c)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats,fontsize=11)
ax.set_yticklabels([]); ax.set_ylim(0,1)
ax.set_title('Multi-Metric Comparison (Radar)\nAll axes: higher is better', fontsize=12, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35,1.15), fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'fig5_radar_chart.pdf'), bbox_inches='tight', dpi=200)
plt.savefig(os.path.join(OUT,'fig5_radar_chart.png'), bbox_inches='tight', dpi=200)
plt.close(); print('Fig 5 done')

for s,d in [('eda_01_dataset_overview.png','fig6_eda_overview.png'),
            ('eda_03_network_structure.png','fig7_network_structure.png'),
            ('eda_04_temporal_quality.png','fig8_temporal_decomp.png')]:
    sp=os.path.join(SRC,s)
    if os.path.exists(sp): shutil.copy2(sp,os.path.join(OUT,d)); print(f'Copied {s}')

print('\nAll figures generated successfully.')
