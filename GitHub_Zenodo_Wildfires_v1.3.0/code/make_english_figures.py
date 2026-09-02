from pathlib import Path
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
FIG = BASE / 'figures'
FIG.mkdir(exist_ok=True)

series = pd.read_csv(BASE / 'data' / 'serie_modelizacion_2005_2025.csv')
fit_ext = pd.read_csv(BASE / 'results' / 'classical' / 'predicciones_ajuste_y_externo_figura.csv')
ext_seq = pd.read_csv(BASE / 'results' / 'classical' / 'validacion_externa_secuencial.csv')
metrics_count = pd.read_csv(BASE / 'results' / 'metricas_rodantes_conteo_v13.csv')
post = pd.read_csv(BASE / 'results' / 'coeficientes_posteriores_nb_ar1_HDP2.csv')
loadings = pd.read_csv(BASE / 'results' / 'cargas_pca_count.csv')
grid = pd.read_csv(BASE / 'results' / 'classical' / 'metricas_grid_arima_y_baselines.csv')
pros_arima = pd.read_csv(BASE / 'results' / 'prospective_2026' / 'frozen_arima_2026_forecast.csv')
pros_snapshot = pd.read_csv(BASE / 'results' / 'prospective_2026' / 'miteco_2026_snapshot_used_in_manuscript.csv')

# 1. Evolution figure with frozen prospective 2026 snapshot
# Panel A preserves the annual modelling view; Panel B shows the genuinely
# prospective sequence of date-stamped 2026 observations without treating them as
# a completed annual outcome.
fig, (ax, axb) = plt.subplots(2, 1, figsize=(11.2, 8.2), gridspec_kw={'height_ratios':[2.05, 1.0]})
obs_c, ar_c, ax_c = '#1f77b4', '#ff7f0e', '#2ca02c'
ax.plot(series['year'], series['count_gif'], marker='o', linewidth=2.2, color=obs_c, label='Observed annual total (final through 2025)')
for model, label, ls, marker, color in [
    ('ARIMA(1,0,0)', 'ARIMA(1,0,0)', '--', 's', ar_c),
    ('ARIMAX calor + prevención t-1', 'ARIMAX: heat + prevention (t-1)', '-.', 'D', ax_c),
]:
    dd = fit_ext[fit_ext['model'] == model].sort_values('year')
    ax.plot(dd['year'], dd['predicted'], linestyle=ls, marker=marker, linewidth=1.8, markersize=4.5, color=color, label=label)
    f = dd[dd['phase'] == 'forecast']
    ax.fill_between(f['year'].to_numpy(float), f['pi90_low'].to_numpy(float), f['pi90_high'].to_numpy(float), alpha=0.11, color=color)

# Frozen one-step-ahead ARIMA forecast for 2026, fitted only through 2025.
r = pros_arima.iloc[0]
ax.vlines(2026, r['pi90_low'], r['pi90_high'], linewidth=3.0, alpha=0.35, color=ar_c)
ax.scatter([2026], [r['predicted_count']], marker='s', s=46, color=ar_c, zorder=6)

# Latest provisional 2026 snapshot: not an annual outcome.
s_latest = pros_snapshot.sort_values('date').iloc[-1]
ax.scatter([2026.04], [s_latest['large_wildfires']], s=82, marker='o', facecolors='white', edgecolors='black', linewidths=1.6, zorder=7, label='2026 provisional: 44 by 31 Aug')

ax.axvline(2023.5, linewidth=1.0, color='0.45')
ax.axvline(2025.5, linewidth=1.0, color='0.45')
ax.axvspan(2023.5, 2025.5, alpha=0.07, color='0.6')
ax.axvspan(2025.5, 2026.35, alpha=0.12, color='0.75')
ax.text(2017.0, 91, 'Development / rolling validation', ha='center', va='center', fontsize=9.2)
ax.text(2024.5, 91, 'External 2024-2025', ha='center', va='center', fontsize=9.0)
ax.text(2026.18, 87, '2026 prospective', ha='center', va='center', rotation=90, fontsize=8.6)
ax.annotate('2025: 63 observed\n46.7 conditional ARIMAX nowcast',
            xy=(2025, 63), xytext=(2021.15, 77),
            arrowprops={'arrowstyle': '->', 'lw': 1.0}, ha='left', va='center', fontsize=8.8)
ax.annotate('Frozen ARIMA: 17.4\n90% PI 4.6-59.5',
            xy=(2026, r['predicted_count']), xytext=(2022.25, 9),
            arrowprops={'arrowstyle': '->', 'lw': 1.0}, ha='left', va='center', fontsize=8.8)
ax.set_title('(A) Annual series and external predictions')
ax.set_xlabel('Year')
ax.set_ylabel('Number of large wildfires')
ax.set_xlim(2005, 2026.35)
ax.set_ylim(0, 98)
ax.grid(axis='y', alpha=0.25)
ax.legend(loc='upper left', frameon=True, fontsize=8.0, ncol=1)

# Panel B: dated 2026 lower-bound sequence vs frozen annual distribution.
ps = pros_snapshot.copy()
ps['date'] = pd.to_datetime(ps['date'])
axb.axhspan(float(r['pi90_low']), float(r['pi90_high']), alpha=0.12, color=ar_c, label='Frozen ARIMA 90% PI')
axb.axhline(float(r['predicted_count']), linestyle='--', linewidth=1.6, color=ar_c, label='Frozen ARIMA point forecast')
axb.plot(ps['date'], ps['large_wildfires'], marker='o', linewidth=2.0, color='black', label='MITECO provisional count')
for _, rr in ps.iterrows():
    axb.annotate(f"{int(rr['large_wildfires'])}", (rr['date'], rr['large_wildfires']), xytext=(0,7), textcoords='offset points', ha='center', fontsize=8.5)
# Same-date 2025 comparator from the official 31-Aug-2025 MITECO report.
aug31 = pd.Timestamp('2026-08-31')
axb.scatter([aug31], [60], marker='x', s=62, linewidths=1.7, color='#8c2d04', label='2025 count by 31 Aug: 60')
axb.annotate('44 = 2.52x point forecast\n(still inside PI90)', xy=(aug31, 44), xytext=(-165, 18), textcoords='offset points',
             arrowprops={'arrowstyle':'->','lw':0.9}, fontsize=8.6, ha='left')
axb.set_title('(B) Prospective 2026 snapshots')
axb.set_xlabel('2026 snapshot date')
axb.set_ylabel('Large wildfires recorded')
axb.set_ylim(0, 70)
axb.grid(axis='y', alpha=0.25)
axb.tick_params(axis='x', rotation=35)
axb.legend(loc='upper left', fontsize=8.1, frameon=True, ncol=2)

fig.suptitle('Observed large-wildfire counts and the frozen 2026 prospective stress test', y=0.995, fontsize=13)
fig.tight_layout()
fig.savefig(FIG / 'fig1_evolution_arima_arimax.pdf', bbox_inches='tight')
fig.savefig(FIG / 'fig1_evolution_arima_arimax.png', dpi=250, bbox_inches='tight')
plt.close(fig)

# 2. CRPS bar chart
labels_map = {
    'NB-AR1+HE+I2': 'Heat events + total investment (t-2)',
    'NB-AR1+HE+P2': 'Heat events + prevention (t-2)',
    'NB-AR1': 'Endogenous count model',
    'NB-AR1+I2': 'Total investment (t-2)',
    'NB-AR1+P2': 'Prevention (t-2)',
    'NB-AR1+HE': 'Heat events',
    'NB-AR1+HD+P2': 'Heatwave days + prevention (t-2)',
}
d = metrics_count.sort_values('CRPS', ascending=True).copy()
d['label'] = d['model_id'].map(labels_map).fillna(d['model_id'])
fig, ax = plt.subplots(figsize=(8.6, 4.9))
ax.barh(d['label'], d['CRPS'])
ax.set_xlabel('Mean CRPS (lower is better)')
ax.set_title('Probabilistic accuracy in rolling-origin validation, 2014-2023')
ax.grid(axis='x', alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / 'fig2_crps_models.pdf', bbox_inches='tight')
fig.savefig(FIG / 'fig2_crps_models.png', dpi=250, bbox_inches='tight')
plt.close(fig)

# 3. Posterior effects
sel = post[post['parameter'].isin(['heatwave_days', 'prevention_lag2'])].copy()
sel['label'] = sel['parameter'].map({
    'heatwave_days': '+10 heatwave days',
    'prevention_lag2': '+1 real EUR ha$^{-1}$ prevention (t-2)',
})
# Reverse so heat is on top
sel = sel.iloc[::-1]
fig, ax = plt.subplots(figsize=(8.3, 3.6))
y = np.arange(len(sel))
ax.errorbar(sel['irr_mean'], y,
            xerr=[sel['irr_mean'] - sel['irr_q025'], sel['irr_q975'] - sel['irr_mean']],
            fmt='o', capsize=4, linewidth=1.5)
ax.axvline(1.0, linestyle='--', linewidth=1.0)
ax.set_yticks(y, sel['label'])
ax.set_xlabel('Incidence-rate ratio (posterior mean and 95% credible interval)')
ax.set_title('Posterior effects from the negative-binomial AR(1) model')
for yi, (_, r) in enumerate(sel.iterrows()):
    prob = 1-r['prob_negative'] if r['parameter']=='heatwave_days' else r['prob_negative']
    sign = '>0' if r['parameter']=='heatwave_days' else '<0'
    ax.text(r['irr_q975'] + 0.02, yi, f'P(effect {sign})={prob:.3f}', va='center', fontsize=9)
ax.grid(axis='x', alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / 'fig3_posterior_effects.pdf', bbox_inches='tight')
fig.savefig(FIG / 'fig3_posterior_effects.png', dpi=250, bbox_inches='tight')
plt.close(fig)

# 4. PCA biplot
var_labels = {
    'count_gif': 'Large-wildfire count',
    'heatwave_days': 'Heatwave days',
    'heatwave_events': 'Heatwave events',
    'prevention_lag2': 'Prevention (t-2)',
    'total_investment_lag2': 'Total investment (t-2)',
}
fig, ax = plt.subplots(figsize=(7.2, 6.2))
ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)
for _, r in loadings.iterrows():
    ax.arrow(0, 0, r['PC1'], r['PC2'], head_width=0.025, length_includes_head=True)
    ax.text(r['PC1']*1.08, r['PC2']*1.08, var_labels.get(r['variable'], r['variable']), ha='center', va='center', fontsize=9)
ax.set_xlim(-1.05, 1.1)
ax.set_ylim(-0.15, 1.02)
ax.set_xlabel('PC1 (54.0%)')
ax.set_ylabel('PC2 (32.5%)')
ax.set_title('Principal-component structure of annual counts and exogenous predictors')
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(FIG / 'fig4_pca_biplot.pdf', bbox_inches='tight')
fig.savefig(FIG / 'fig4_pca_biplot.png', dpi=250, bbox_inches='tight')
plt.close(fig)

# S1. ARIMA grid heatmap, excluding constant-only (0,0,0), already absent
ar = grid[grid['family'] == 'ARIMA'].copy()
orders = ar['order'].apply(ast.literal_eval)
ar[['p','d','q']] = pd.DataFrame(orders.tolist(), index=ar.index)
# panels by differencing order; reserve a dedicated colour-bar axis
fig = plt.figure(figsize=(10.6, 4.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.055], wspace=0.24)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
cax = fig.add_subplot(gs[0, 2])
vmin, vmax = ar['MAE'].min(), ar['MAE'].max()
for ax, dval in zip(axes, [0,1]):
    sub = ar[ar['d'] == dval]
    mat = np.full((4,4), np.nan)
    for _, r in sub.iterrows():
        mat[int(r['p']), int(r['q'])] = r['MAE']
    im = ax.imshow(mat, aspect='equal', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(4), [f'q={q}' for q in range(4)])
    ax.set_yticks(range(4), [f'p={p}' for p in range(4)])
    ax.set_title(f'd={dval}')
    for p in range(4):
        for q in range(4):
            if np.isfinite(mat[p,q]):
                ax.text(q, p, f'{mat[p,q]:.1f}', ha='center', va='center', fontsize=8)
    # identify ARIMA(1,0,0)
    if dval == 0:
        ax.add_patch(plt.Rectangle((-0.5,0.5),1,1,fill=False,linewidth=2.0))
cb = fig.colorbar(im, cax=cax)
cb.set_label('Rolling-origin MAE')
fig.suptitle('Rolling-origin MAE across the prespecified ARIMA grid', y=0.98)
fig.supxlabel('Moving-average order', y=0.04)
fig.supylabel('Autoregressive order', x=0.02)
fig.subplots_adjust(left=0.09, right=0.95, top=0.84, bottom=0.17)
fig.savefig(FIG / 'figS1_arima_grid.pdf', bbox_inches='tight')
fig.savefig(FIG / 'figS1_arima_grid.png', dpi=250, bbox_inches='tight')
plt.close(fig)

# S2. 2025 lag sensitivity
s = ext_seq[(ext_seq['year'] == 2025) & ext_seq['model'].str.contains('ARIMAX calor\+prevención', regex=True)].copy()
s['lag'] = s['model'].str.extract(r't-(\d)').astype(int)
s = s.sort_values('lag')
fig, ax = plt.subplots(figsize=(7.8, 4.2))
y = np.arange(len(s))
ax.errorbar(s['predicted'], y,
            xerr=[s['predicted'] - s['pi90_low'], s['pi90_high'] - s['predicted']],
            fmt='o', capsize=4)
ax.axvline(63, linestyle='--', linewidth=1.2, label='Observed in 2025: 63')
ax.set_yticks(y, [f'Prevention lag t-{i}' for i in s['lag']])
ax.set_xlabel('Predicted number of large wildfires (90% prediction interval)')
ax.set_title('Sensitivity of the 2025 external prediction to prevention lag')
ax.grid(axis='x', alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(FIG / 'figS2_2025_lag_sensitivity.pdf', bbox_inches='tight')
fig.savefig(FIG / 'figS2_2025_lag_sensitivity.png', dpi=250, bbox_inches='tight')
plt.close(fig)

print('English figures written to', FIG)
