"""Figures for the staged OneK1K models (thesis Figure 17) and age-group ROC/PR
analysis (Figure 16) from results/onek1k/staged_onek1k_results.json and
staged_onek1k_oof.csv."""
import json, sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.environ.get('TFM_DATA', os.path.join(ROOT, 'data'))
OUT = os.environ.get('TFM_OUT', os.path.join(ROOT, 'results', 'onek1k'))
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
matplotlib.rcParams['svg.fonttype'] = 'none'

NAVY = '#1a1a4e'
BLUE = '#2E5FD7'
GREEN = '#1B9E77'
ORANGE = '#E08214'
RED = '#D73027'
GRAY = '#888888'

res = json.load(open(os.path.join(OUT, 'staged_onek1k_results.json')))
R = res['results']
oof = pd.read_csv(os.path.join(OUT, 'staged_onek1k_oof.csv'))
y = oof['age'].values

stages = ['EN_full', 'EN_concordant', 'XGB', 'STACK']
labels = ['EN baseline', 'EN concordant', 'XGBoost', 'Stacked']
colors = [BLUE, GREEN, ORANGE, RED]
r2 = [R[s]['R2'] for s in stages]
mae = [R[s]['MAE'] for s in stages]
base_r2, base_mae = r2[0], mae[0]

# ---------------- FIGURE 3 ----------------
fig, axes = plt.subplots(1, 3, figsize=(20, 7.2))
fig.suptitle('Model Improvement Pipeline: OneK1K Age Clock (n=981, ages 19-97)',
             fontsize=17, fontweight='bold', color=NAVY, y=0.98)

# Panel A: R2 bars
ax = axes[0]
x = np.arange(4)
ax.bar(x, r2, color=colors, width=0.62, zorder=3)
ax.axhline(base_r2, color=GRAY, ls=':', lw=1.4, zorder=2)
for i, v in enumerate(r2):
    ax.text(i, v + 0.008, f'R²={v:.3f}', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color=NAVY)
for i in range(1, 4):
    gain = (r2[i] - base_r2) / abs(base_r2) * 100
    ax.text(i, r2[i] - 0.035, f'+{gain:.1f}%' if gain >= 0 else f'{gain:.1f}%',
            ha='center', va='top', fontsize=11, fontweight='bold',
            color='#0a7a3d' if gain >= 0 else '#b2182b')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10.5, rotation=18, ha='right')
ax.set_ylabel('Pooled out-of-fold R²', fontsize=12)
ax.set_title('A.  R² improvement by pipeline stage', fontsize=13, fontweight='bold',
             color=NAVY, loc='left')
ax.set_ylim(0, max(r2) + 0.34)  # headroom so the stage box clears the bar labels
ax.grid(axis='y', color='#dddddd', lw=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
desc = ('Stage 1: concordance filter\n→ cross-cohort robust panel\n\n'
        'Stage 2: XGBoost\n→ gradient-boosted trees,\nrandomised hyperparam.\n\n'
        'Stage 3: stacked ensemble\n→ Ridge meta-learner')
ax.text(0.97, 0.97, desc, transform=ax.transAxes, ha='right', va='top', fontsize=8.6,
        color='#0a5a2c', bbox=dict(boxstyle='round,pad=0.5', fc='#f2fbf6', ec='#1B9E77', lw=1.2))

# Panel B: MAE bars
ax = axes[1]
ax.bar(x, mae, color=colors, width=0.62, zorder=3)
ax.axhline(base_mae, color=GRAY, ls=':', lw=1.4, zorder=2)
for i, v in enumerate(mae):
    ax.text(i, v + 0.12, f'{v:.2f} y', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color=NAVY)
for i in range(1, 4):
    d = mae[i] - base_mae
    ax.text(i, mae[i] - 0.45, f'{d:+.2f} y'.replace('-', '−'), ha='center', va='top',
            fontsize=11, fontweight='bold', color='#0a7a3d' if d <= 0 else '#b2182b')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10.5, rotation=18, ha='right')
ax.set_ylabel('Pooled MAE (years)', fontsize=12)
ax.set_title('B.  MAE reduction by pipeline stage', fontsize=13, fontweight='bold',
             color=NAVY, loc='left')
ax.set_ylim(0, max(mae) * 1.22)
ax.grid(axis='y', color='#dddddd', lw=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

# Panel C: scatter EN vs STACK
ax = axes[2]
p_en, p_st = oof['pred_EN_full'].values, oof['pred_STACK'].values
ax.scatter(y, p_en, s=14, alpha=0.35, color=BLUE, edgecolors='none',
           label=f'EN baseline (R²={r2[0]:.3f})')
ax.scatter(y, p_st, s=14, alpha=0.35, color=RED, edgecolors='none',
           label=f'Stacked (R²={r2[3]:.3f})')
lims = [15, 100]
ax.plot(lims, lims, '--', color=GRAY, lw=1.3, label='Perfect prediction')
for p, c in [(p_en, BLUE), (p_st, RED)]:
    b, a = np.polyfit(y, p, 1)
    xs = np.array([y.min(), y.max()])
    ax.plot(xs, a + b * xs, color=c, lw=2.2)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('True chronological age (years)', fontsize=12)
ax.set_ylabel('Predicted age (years)', fontsize=12)
ax.set_title('C.  Predicted vs true age scatter', fontsize=13, fontweight='bold',
             color=NAVY, loc='left')
ax.legend(loc='upper left', fontsize=9.5, framealpha=0.95)
ax.grid(color='#dddddd', lw=0.7)
ax.set_axisbelow(True)
dr2 = r2[3] - base_r2
dmae = mae[3] - base_mae
ax.text(0.97, 0.05, f'ΔR² = {dr2:+.3f} ({dr2 / abs(base_r2) * 100:+.1f}%)\nΔMAE = {dmae:+.1f} y'.replace('-', '−'),
        transform=ax.transAxes, ha='right', va='bottom', fontsize=11, fontweight='bold',
        color=RED, bbox=dict(boxstyle='round,pad=0.45', fc='#fdf3f2', ec=RED, lw=1.4))
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT, 'fig_staged_models.png'), dpi=200, facecolor='white')
fig.savefig(os.path.join(OUT, 'fig_staged_models.svg'), facecolor='white')
plt.close(fig)
print('figure3 saved')

# ---------------- FIGURE 17 ----------------
med = float(np.median(y))
bins = [
    ('Young (<45) vs Old (≥70)', (y < 45) | (y >= 70), (y >= 70),
     'Extreme contrast: clear biological difference\nexpected between young adults and older seniors'),
    ('Young (<55) vs Old (≥65)', (y < 55) | (y >= 65), (y >= 65),
     'Clinical decade gap: most relevant for\nintervention trials targeting 65+ adults'),
    ('Below vs Above Median Age', np.ones(len(y), bool), (y >= med),
     f'Balanced split at median age ({med:.1f} y).\nToughest test: smallest decision margin'),
]
models = [('Elastic Net (full)', 'pred_EN_full', BLUE),
          ('Elastic Net (concordant)', 'pred_EN_concordant', GREEN),
          ('XGBoost', 'pred_XGB', ORANGE),
          ('Stacked ensemble', 'pred_STACK', RED)]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('ROC Analysis: Polyamine–CIRBP Panel Age-Group Discrimination\n'
             'OneK1K pseudobulk (n=981, ages 19-97)  |  Classifier score = predicted age',
             fontsize=16, fontweight='bold', color=NAVY, y=0.995)

auc_table = {}
for col, (btitle, mask, pos, note) in enumerate(bins):
    yb = pos[mask].astype(int)
    prev = yb.mean()
    # ROC
    ax = axes[0, col]
    for mname, colname, c in models:
        s = oof[colname].values[mask]
        fpr, tpr, _ = roc_curve(yb, s)
        a = auc(fpr, tpr)
        auc_table[(btitle, mname)] = a
        ax.plot(fpr, tpr, color=c, lw=2.0, label=f'{mname}  (AUC={a:.3f})')
    ax.plot([0, 1], [0, 1], '--', color=GRAY, lw=1.2, label='Chance (AUC=0.500)')
    ax.text(0.03, 0.97, note, transform=ax.transAxes, ha='left', va='top', fontsize=8.6,
            color='#444444', bbox=dict(boxstyle='round,pad=0.4', fc='#f4f4f4', ec='#bbbbbb', lw=0.9))
    ax.set_title(f'ROC — {btitle}', fontsize=12.5, fontweight='bold', color=NAVY)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc='lower right', fontsize=8.8, framealpha=0.95)
    ax.grid(color='#dddddd', lw=0.7); ax.set_axisbelow(True)
    # PR
    ax = axes[1, col]
    for mname, colname, c in models:
        s = oof[colname].values[mask]
        prec, rec, _ = precision_recall_curve(yb, s)
        ap = average_precision_score(yb, s)
        auc_table[(btitle, mname + '_AP')] = ap
        ax.plot(rec, prec, color=c, lw=2.0, label=f'{mname}  (AP={ap:.3f})')
    ax.axhline(prev, color=GRAY, ls='--', lw=1.2, label=f'No-skill baseline (prev={prev:.2f})')
    ax.set_title(f'PR — {btitle}', fontsize=12.5, fontweight='bold', color=NAVY)
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc='upper right', fontsize=8.8, framealpha=0.95)
    ax.grid(color='#dddddd', lw=0.7); ax.set_axisbelow(True)

fig.text(0.5, 0.012,
         'Note: ROC/AUC quantifies binary discrimination performance of the regression clock. '
         'The primary performance metric remains pooled R² and MAE (regression).\n'
         'AUC is computed using out-of-fold predicted ages from the 10-fold KFold cross-validation '
         'pipeline on the OneK1K pseudobulk data (n=981).',
         ha='center', va='bottom', fontsize=9.5, style='italic', color='#555555')
fig.tight_layout(rect=[0, 0.045, 1, 0.96])
fig.savefig(os.path.join(OUT, 'fig_roc_pr.png'), dpi=200, facecolor='white')
fig.savefig(os.path.join(OUT, 'fig_roc_pr.svg'), facecolor='white')
plt.close(fig)
print('figure17 saved')

# dump AUC table for the text rewrite
out = {}
for (b, m), v in auc_table.items():
    out.setdefault(b, {})[m] = round(v, 3)
with open(os.path.join(OUT, 'roc_pr_aucs.json'), 'w') as f:
    json.dump({'median_age': med, 'aucs': out}, f, indent=2)
print(json.dumps({'median_age': med, 'aucs': out}, indent=2))
