from pathlib import Path
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
classic=pd.read_csv(BASE/'results'/'classical'/'metricas_grid_arima_y_baselines.csv')
common=pd.read_csv(BASE/'results'/'classical'/'sensibilidad_retardos_comun_2016_2023.csv')
external=pd.read_csv(BASE/'results'/'classical'/'validacion_externa_secuencial.csv')
nb=pd.read_csv(BASE/'results'/'metricas_rodantes_conteo_v13.csv')
post=pd.read_csv(BASE/'results'/'coeficientes_posteriores_nb_ar1_HDP2.csv')
load=pd.read_csv(BASE/'results'/'cargas_pca_count.csv')

lines=[]
a=lines.append
a(r'''\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype}
\usepackage[a4paper,margin=1.65cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,tabularx,longtable,graphicx,float,array}
\usepackage[hidelinks]{hyperref}
\newcolumntype{Y}{>{\raggedleft\arraybackslash}X}
\title{Supplementary Material\\Forecasting annual large-wildfire counts in Spain}
\author{Santi García-Cremades, José Juan López Espín, José María Cecilia Canales}
\date{}
\begin{document}
\maketitle

\section{Reproducibility map}
\begin{table}[H]\centering\small
\begin{tabularx}{\textwidth}{p{4.6cm}X}
\toprule
File & Purpose \\
\midrule
\texttt{analisis\_clasico\_arima\_arimax\_v13.py} & Prespecified ARIMA grid, AutoReg models, exponential smoothing, Holt, ARIMAX lag sensitivity, fit and predictive $R^2$, relative skill and sequential external validation. \\
\texttt{modelos\_conteo\_gif\_v13.py} & NB2 observation-driven models, Bayesian negative-binomial AR(1), rolling-origin validation, probabilistic metrics and external validation. \\
\texttt{descargar\_agregar\_fwi\_era5.py} & Copernicus/ERA5 historical FWI request and spatial aggregation. \\
\texttt{validar\_panel.py} & Regional-panel coverage, type, range, real-euro and adjacency checks. \\
\texttt{panel\_ccaa\_gif.py} & Hierarchical negative-binomial panel with offset, Mundlak decomposition, annual random walk and optional BYM2 component. \\
\bottomrule
\end{tabularx}
\end{table}

\section{Response overdispersion}
For 2005--2023, $n=19$, mean $=25.684$, variance $=319.006$, variance-to-mean ratio $=12.420$, minimum $=3$ and maximum $=58$. The Bayesian parameterisation uses $\operatorname{Var}(N\mid\mu,\alpha)=\mu+\mu^2/\alpha$, whereas the frequentist NB2 parameterisation uses $\mu+\phi\mu^2$.

\section{Complete classical benchmark}
The constant-only ARIMA$(0,0,0)$ and the unstable highest-order specification were excluded a priori. Rolling-origin errors, rather than in-sample AICc alone, determined the benchmark.

\subsection{ARIMA grid}
\begin{longtable}{lrrrrrr}
\toprule
Model & $n$ & MAE & RMSE & NMAE & MASE & $R^2_{\mathrm{pred}}$ \\
\midrule
\endfirsthead
\toprule
Model & $n$ & MAE & RMSE & NMAE & MASE & $R^2_{\mathrm{pred}}$ \\
\midrule
\endhead
''')
for _,r in classic[classic.family=='ARIMA'].sort_values('MAE').iterrows():
    order=str(r['order']).replace(' ','')
    a(f"ARIMA${order}$ & {int(r['n'])} & {r['MAE']:.2f} & {r['RMSE']:.2f} & {r['NMAE']:.3f} & {r['MASE']:.3f} & {r['R2_pred']:.3f} \\\\")
a(r'''\bottomrule
\end{longtable}

\subsection{Non-ARIMA references}
\begin{table}[H]\centering\small
\begin{tabular}{lrrrrrr}
\toprule
Model & $n$ & MAE & RMSE & NMAE & MASE & $R^2_{\mathrm{pred}}$ \\
\midrule''')
name_map={'Holt_damped':'Damped Holt','Holt':'Additive Holt','AutoReg1':'AutoReg(1)','AutoReg2':'AutoReg(2)','AutoReg3':'AutoReg(3)','SES':'Simple exponential smoothing','mean':'Historical mean','naive':'Naive'}
for _,r in classic[classic.family!='ARIMA'].sort_values('MAE').iterrows():
    name=name_map.get(r['spec'],r['spec'])
    a(f"{name} & {int(r['n'])} & {r['MAE']:.2f} & {r['RMSE']:.2f} & {r['NMAE']:.3f} & {r['MASE']:.3f} & {r['R2_pred']:.3f} \\\\")
a(r'''\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=0.86\textwidth]{../figures/figS1_arima_grid.pdf}
\caption{Rolling-origin MAE across the prespecified ARIMA grid. ARIMA$(1,0,0)$ is outlined.}
\end{figure}

Negative predictive $R^2$ values are valid: they mean that the squared forecast error exceeded the error of using the validation-window mean. They are not retrospective coefficients of determination.

\section{ARIMAX lag sensitivity and coefficients}
\begin{table}[H]\centering\small
\begin{tabular}{lrrrrrr}
\toprule
Specification & MAE & RMSE & $R^2_{\mathrm{pred}}$ & MSE skill & $p_H$ & $p_P$ \\
\midrule''')
label_map={'calor+prevención t-1':'Heat + prevention $(t-1)$','calor+prevención t-2':'Heat + prevention $(t-2)$','calor+prevención t-3':'Heat + prevention $(t-3)$','calor solo':'Heat only','ARIMA(1,0,0)':'ARIMA$(1,0,0)$'}
for _,r in common.iterrows():
    ph='---' if pd.isna(r['heat_p_value']) else f"{r['heat_p_value']:.3f}"
    pp='---' if pd.isna(r['prevention_p_value']) else f"{r['prevention_p_value']:.3f}"
    a(f"{label_map.get(r['spec'],r['spec'])} & {r['MAE']:.2f} & {r['RMSE']:.2f} & {r['R2_pred']:.3f} & {r['Skill_MSE_vs_ARIMA100']:.3f} & {ph} & {pp} \\\\")
a(r'''\bottomrule
\end{tabular}
\end{table}

The lag-one model was retained for conditional prediction because it had the lowest RMSE among specifications containing prevention. The lag-two model was retained for coefficient interpretation because it was the structural lag and had the most precise coefficients. No lag is claimed to be causally identified from the national series.

\section{Sequential external validation}
\begin{longtable}{llrrrrrr}
\toprule
Model & Year & Observed & Predicted & PI90 low & PI90 high & Std. residual & Upper-tail $p$ \\
\midrule
\endfirsthead
\toprule
Model & Year & Observed & Predicted & PI90 low & PI90 high & Std. residual & Upper-tail $p$ \\
\midrule
\endhead
''')
model_map={'ARIMAX calor+prevención t-1':'ARIMAX heat + prevention $(t-1)$','ARIMAX calor+prevención t-2':'ARIMAX heat + prevention $(t-2)$','ARIMAX calor+prevención t-3':'ARIMAX heat + prevention $(t-3)$','ARIMAX episodios+inversión t-2':'ARIMAX events + investment $(t-2)$'}
for _,r in external.iterrows():
    name=model_map.get(r['model'],r['model'])
    a(f"{name} & {int(r['year'])} & {int(r['observed'])} & {r['predicted']:.2f} & {r['pi90_low']:.2f} & {r['pi90_high']:.2f} & {r['standardized_log_residual']:.3f} & {r['one_sided_upper_tail_p']:.4f} \\\\")
a(r'''\bottomrule
\end{longtable}

\begin{figure}[H]\centering
\includegraphics[width=0.82\textwidth]{../figures/figS2_2025_lag_sensitivity.pdf}
\caption{Sensitivity of the 2025 conditional prediction to the prevention lag.}
\end{figure}

\section{Complete negative-binomial rolling validation}
\begin{longtable}{lrrrrrrrrr}
\toprule
Model & MAE & RMSE & NMAE & NRMSE & MASE & $R^2_{\mathrm{pred}}$ & MSE skill & CRPS & Coverage \\
\midrule
\endfirsthead
\toprule
Model & MAE & RMSE & NMAE & NRMSE & MASE & $R^2_{\mathrm{pred}}$ & MSE skill & CRPS & Coverage \\
\midrule
\endhead
''')
nb_map={'NB-AR1+HE+I2':'Events + investment $(t-2)$','NB-AR1+HE+P2':'Events + prevention $(t-2)$','NB-AR1':'Endogenous','NB-AR1+I2':'Investment $(t-2)$','NB-AR1+P2':'Prevention $(t-2)$','NB-AR1+HE':'Heat events','NB-AR1+HD+P2':'Heatwave days + prevention $(t-2)$'}
for _,r in nb.sort_values('CRPS').iterrows():
    a(f"{nb_map.get(r['model_id'],r['model_id'])} & {r['MAE']:.2f} & {r['RMSE']:.2f} & {r['NMAE']:.3f} & {r['NRMSE']:.3f} & {r['MASE']:.3f} & {r['R2_pred']:.3f} & {r['Skill_MSE_vs_endogenous']:.3f} & {r['CRPS']:.2f} & {r['coverage_90']:.2f} \\\\")
a(r'''\bottomrule
\end{longtable}

\section{Bayesian posterior summary}
\begin{table}[H]\centering\small
\begin{tabular}{lrrrrrr}
\toprule
Parameter & Mean & 2.5\% & 97.5\% & IRR & $P(<0)$ & $\widehat R$ \\
\midrule''')
param_map={'heatwave_days':'Heatwave days (standardised)','prevention_lag2':'Prevention $(t-2)$ (standardised)','beta0':'Intercept','rho':'AR(1) coefficient','sigma_u':'State SD','alpha_disp':'Dispersion'}
for _,r in post.iterrows():
    irr='---' if pd.isna(r['irr_mean']) else f"{r['irr_mean']:.3f}"
    a(f"{param_map.get(r['parameter'],r['parameter'])} & {r['mean_beta']:.3f} & {r['q025_beta']:.3f} & {r['q975_beta']:.3f} & {irr} & {r['prob_negative']:.3f} & {r['rhat']:.3f} \\\\")
a(r'''\bottomrule
\end{tabular}
\end{table}

On the natural scale, ten additional heatwave days had posterior IRR 1.587 (95% credible interval 1.169--2.081). One additional real EUR ha$^{-1}$ of lag-two prevention had posterior IRR 1.095 (0.966--1.227).

\section{PCA loadings and regional extension}
\begin{table}[H]\centering\small
\begin{tabular}{lrr}
\toprule
Variable & PC1 & PC2 \\
\midrule''')
load_map={'count_gif':'Large-wildfire count','heatwave_days':'Heatwave days','heatwave_events':'Heatwave events','prevention_lag2':'Prevention $(t-2)$','total_investment_lag2':'Total investment $(t-2)$'}
for _,r in load.iterrows():
    a(f"{load_map.get(r['variable'],r['variable'])} & {r['PC1']:.3f} & {r['PC2']:.3f} \\\\")
a(r'''\bottomrule
\end{tabular}
\end{table}

PC1 and PC2 explained 54.0% and 32.5% of variance, respectively. The planned regional model is
\[
N_{it}\sim\operatorname{NB}(\mu_{it},\alpha),\qquad
\log\mu_{it}=\log(F_{it}/10^6)+\beta_0+a_i+\tau_t+\beta_HH_{it}
+\beta_W(P_{i,t-2}-\overline P_i)+\gamma\overline P_i.
\]
The panel template covers 17 autonomous communities and 2003--2025. Years 2003--2004 create the lagged expenditure; 2005--2023 form development and 2024--2025 are external. The panel remains intentionally unestimated until validated regional count, heat, expenditure and FWI data are complete.

\section{FWI integration rule}
The observational FWI covariate must be reconstructed from ERA5/Copernicus for the analytical period, using regional masks and a preregistered threshold/window. Climate-projection values are used only after estimating the observational coefficient and are never used to impute the historical series.

\end{document}
''')

out=BASE/'supplementary'/'supplementary_material_jem.tex'
out.write_text('\n'.join(lines), encoding='utf-8')
print(out)
