from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'
RES = BASE / 'results' / 'prospective_2026'
RES.mkdir(parents=True, exist_ok=True)


def fit_sarimax(y, x=None, order=(1,0,0)):
    trend = 'c' if order[1] == 0 else 't'
    return SARIMAX(
        y, exog=x, order=order, trend=trend,
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False, maxiter=250)


def inv_log_median(mu):
    return max(0.0, float(np.expm1(min(15.0, mu))))


def interval_original(mu, se, level=0.90):
    a = (1-level)/2
    loz, hiz = norm.ppf(a), norm.ppf(1-a)
    lo = max(0.0, float(np.expm1(mu + loz*se)))
    hi = max(0.0, float(np.expm1(mu + hiz*se)))
    return lo, hi


d = pd.read_csv(DATA / 'serie_modelizacion_2005_2025.csv').sort_values('year').reset_index(drop=True)
for lag in (1,2,3):
    d[f'prevention_lag{lag}'] = d['prevention_eur_per_forest_ha'].shift(lag)
d['log_count1p'] = np.log1p(d['count_gif'].astype(float))

# 1) Frozen annual ARIMA forecast: all model choices fixed before seeing 2026.
tr = d[d.year <= 2025].copy()
ar = fit_sarimax(tr['log_count1p'].to_numpy(), None, (1,0,0))
fc = ar.get_forecast(1)
mu, se = float(fc.predicted_mean[0]), float(fc.se_mean[0])
lo, hi = interval_original(mu, se, 0.90)
arima_out = pd.DataFrame([{
    'forecast_origin': 2025,
    'target_year': 2026,
    'model': 'ARIMA(1,0,0)',
    'training_years': '2005-2025',
    'predicted_count': inv_log_median(mu),
    'pi90_low': lo,
    'pi90_high': hi,
    'ar1_coefficient': float(ar.params[1]) if len(ar.params) > 1 else np.nan,
    'ar1_p_value': float(ar.pvalues[1]) if len(ar.pvalues) > 1 else np.nan,
}])
arima_out.to_csv(RES / 'frozen_arima_2026_forecast.csv', index=False)

# 2) Pre-specified lag-two conditional scenario analysis. 2024 prevention is known,
# so no 2025 expenditure is imputed. The historical ARIMAX(0,0,0)+exog specification
# is algebraically a log-linear regression; OLS is used here for numerical stability.
# Heatwave days are deliberately scenario values because the final homogeneous AEMET
# annual heatwave-day total for 2026 is unavailable.
cols = ['heatwave_days','prevention_lag2']
tr2 = d[d.year <= 2025].dropna(subset=['log_count1p'] + cols).copy()
X = sm.add_constant(tr2[cols].astype(float), has_constant='add')
rr = sm.OLS(tr2['log_count1p'].astype(float), X).fit()
prevention_2024 = float(d.loc[d.year == 2024, 'prevention_eur_per_forest_ha'].iloc[0])
scenario_days = np.array([15,18,20,22,25,30,33,35,40,45], dtype=float)
rows=[]
for h in scenario_days:
    x = pd.DataFrame({'const':[1.0], 'heatwave_days':[h], 'prevention_lag2':[prevention_2024]})
    f = rr.get_prediction(x).summary_frame(alpha=0.10)
    m = float(f['mean'].iloc[0])
    llog = float(f['obs_ci_lower'].iloc[0])
    ulog = float(f['obs_ci_upper'].iloc[0])
    rows.append({
        'target_year':2026,
        'model':'log-linear heatwave_days + prevention_lag2',
        'heatwave_days_scenario':h,
        'prevention_lag2_eur_per_forest_ha':prevention_2024,
        'predicted_count':inv_log_median(m),
        'pi90_low':max(0.0, float(np.expm1(llog))),
        'pi90_high':max(0.0, float(np.expm1(ulog))),
        'heat_coefficient':float(rr.params['heatwave_days']),
        'heat_p_value':float(rr.pvalues['heatwave_days']),
        'prevention_coefficient':float(rr.params['prevention_lag2']),
        'prevention_p_value':float(rr.pvalues['prevention_lag2']),
        'r2_fit_log_scale':float(rr.rsquared),
    })
pd.DataFrame(rows).to_csv(RES / 'arimax_lag2_2026_heat_scenarios.csv', index=False)

# 3) Same-date seasonal completion benchmarks. These are descriptive checks only,
# not model forecasts. The 2-Aug series is retained for provenance; the manuscript
# now uses the cleaner 31-Aug comparison because the end-of-August 2026 MITECO
# snapshot is available.
aug2_counts = {
    2016:5, 2017:14, 2018:2, 2019:11, 2020:5,
    2021:12, 2022:42, 2023:16, 2024:12, 2025:15, 2026:32,
}
final_map = d.set_index('year')['count_gif'].to_dict()
comp=[]
for y in range(2016,2026):
    a=float(aug2_counts[y]); f=float(final_map[y])
    comp.append({'year':y,'count_by_aug2':a,'final_count':f,
                 'aug2_fraction_of_final':a/f,'final_to_aug2_ratio':f/a})
comp_df=pd.DataFrame(comp)
comp_df.to_csv(RES / 'same_date_completion_aug2_2016_2025.csv',index=False)
ratios=comp_df['final_to_aug2_ratio'].to_numpy(float)
q25, med, q75=np.quantile(ratios,[.25,.5,.75])
summary_aug2=pd.DataFrame([{
    'snapshot_date':'2026-08-02',
    'provisional_large_wildfires':32,
    'historical_years':'2016-2025',
    'median_final_to_snapshot_ratio':med,
    'iqr_ratio_low':q25,
    'iqr_ratio_high':q75,
    'descriptive_median_completion_count':32.0*med,
    'descriptive_iqr_completion_low':32.0*q25,
    'descriptive_iqr_completion_high':32.0*q75,
    'warning':'Descriptive historical completion benchmark; not a model forecast.'
}])
summary_aug2.to_csv(RES / 'same_date_completion_aug2_2026_summary.csv',index=False)

# Official 31-Aug counts for 2015-2024 are transcribed from the MITECO
# 31-Aug-2025 provisional report. The 2026 snapshot is the MITECO balance reported
# after the 31-Aug-2026 follow-up meeting.
aug31_counts = {
    2015:14, 2016:13, 2017:21, 2018:3, 2019:13,
    2020:10, 2021:16, 2022:55, 2023:17, 2024:16,
}
comp31=[]
for y in range(2015,2025):
    a=float(aug31_counts[y]); f=float(final_map[y])
    comp31.append({'year':y,'count_by_aug31':a,'final_count':f,
                   'aug31_fraction_of_final':a/f,'final_to_aug31_ratio':f/a})
comp31_df=pd.DataFrame(comp31)
comp31_df.to_csv(RES / 'same_date_completion_aug31_2015_2024.csv',index=False)
ratios31=comp31_df['final_to_aug31_ratio'].to_numpy(float)
q25_31, med31, q75_31=np.quantile(ratios31,[.25,.5,.75])
summary_aug31=pd.DataFrame([{
    'snapshot_date':'2026-08-31',
    'provisional_large_wildfires':44,
    'historical_years':'2015-2024',
    'historical_same_date_mean_count':float(np.mean(list(aug31_counts.values()))),
    'median_final_to_snapshot_ratio':med31,
    'iqr_ratio_low':q25_31,
    'iqr_ratio_high':q75_31,
    'descriptive_median_completion_count':44.0*med31,
    'descriptive_iqr_completion_low':44.0*q25_31,
    'descriptive_iqr_completion_high':44.0*q75_31,
    'warning':'Descriptive historical completion benchmark; not a model forecast.'
}])
summary_aug31.to_csv(RES / 'same_date_completion_aug31_2026_summary.csv',index=False)

# 4) Machine-readable source snapshots used in the manuscript.
snapshot=pd.DataFrame([
    {'date':'2026-08-02','source':'MITECO provisional report','large_wildfires':32,'total_burned_area_ha':144844.80,'large_wildfire_burned_area_ha':np.nan,
     'note':'Autonomous-community submissions at period close; provisional.'},
    {'date':'2026-08-07','source':'MITECO report note using EFFIS estimate','large_wildfires':39,'total_burned_area_ha':190903.41,'large_wildfire_burned_area_ha':np.nan,
     'note':'EFFIS-supported publication-date estimate; provisional.'},
    {'date':'2026-08-17','source':'MITECO mid-August balance with EFFIS supplementation','large_wildfires':42,'total_burned_area_ha':254730.00,'large_wildfire_burned_area_ha':223872.57,
     'note':'Mid-August manuscript snapshot; provisional.'},
    {'date':'2026-08-21','source':'MITECO weekly advance published 21 Aug with EFFIS supplementation','large_wildfires':43,'total_burned_area_ha':264946.64,'large_wildfire_burned_area_ha':np.nan,
     'note':'Publication-date operational estimate; underlying autonomous-community close referred to 16 Aug; provisional.'},
    {'date':'2026-08-31','source':'MITECO follow-up balance reported after 31-Aug meeting','large_wildfires':44,'total_burned_area_ha':264656.00,'large_wildfire_burned_area_ha':np.nan,
     'note':'Latest manuscript snapshot; 2025 same-date comparison: 60 large wildfires and 346443 ha total forest area. Provisional.'},
])
snapshot.to_csv(RES/'miteco_2026_snapshot_used_in_manuscript.csv',index=False)

print(arima_out.to_string(index=False))
print('\nLag-two scenario at 30 heatwave days:')
print(pd.DataFrame(rows).query('heatwave_days_scenario == 30').to_string(index=False))
print('\nEnd-August completion summary:')
print(summary_aug31.to_string(index=False))
