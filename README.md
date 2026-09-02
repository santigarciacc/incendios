# Data and code for annual large-wildfire forecasting in Spain

Version 1.3.0 — 1 September 2026

This repository contains the annual national dataset, model code, derived results and figure-generation files supporting the manuscript:

**Forecasting annual counts of large wildfires in Spain: separating extreme-heat signals, prevention expenditure and time-series memory**

Authors: Santi García-Cremades, José Juan López Espín and José María Cecilia Canales.

## Scope

The repository reproduces the analyses actually reported in the national manuscript:

- classical ARIMA, autoregressive, exponential-smoothing and Holt benchmarks;
- ARIMAX lag-sensitivity analyses;
- negative-binomial observation-driven and Bayesian count models;
- rolling-origin and external validation;
- probabilistic metrics and prediction intervals;
- PCA and predictor-redundancy analyses;
- English figures and supplementary tables;
- a prospectively frozen 2026 ARIMA forecast and a date-stamped sequence of provisional MITECO/EFFIS snapshots;
- descriptive same-date completion benchmarking and conditional 2026 heat-exposure scenarios, kept separate from the fitted historical models.

The planned autonomous-community panel and the future FWI extension are **not** included as completed analyses in this record. They form a separate future research line.

## Directory structure

- `data/`: annual national modelling series, 2005–2025.
- `code/`: scripts for classical, count-model and figure analyses.
- `results/`: derived tables and stored model outputs used in the manuscript.
- `figures/`: main and supplementary figures.
- `documentation/`: supplementary material.

## Reproduction

Create a clean environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the classical benchmark:

```bash
python code/analisis_clasico_arima_arimax_v13.py \
  --input data/serie_modelizacion_2005_2025.csv \
  --output reproduced/classical
```

Run the frequentist negative-binomial analysis:

```bash
python code/modelos_conteo_gif_v13.py \
  --input data/serie_modelizacion_2005_2025.csv \
  --out reproduced/count \
  --backend glarma
```

Run the Bayesian analysis after installing compatible PyMC and ArviZ versions:

```bash
python code/modelos_conteo_gif_v13.py \
  --input data/serie_modelizacion_2005_2025.csv \
  --out reproduced/count_bayesian \
  --backend pymc \
  --draws 1500 --tune 1500 --chains 4
```

Run the prospective 2026 stress-test script:

```bash
python code/analysis_prospective_2026.py
```

This writes the frozen ARIMA(1,0,0) prediction for 2026, conditional lag-two heat-exposure scenarios, same-date historical completion benchmarks for 2 August and 31 August, and the dated provisional MITECO/EFFIS snapshot to `results/prospective_2026/`. The 2026 final outcome is intentionally absent from this repository version so that it remains a genuine prospective holdout.

## Data provenance

The annual fire series is derived from the Spanish General Wildfire Statistics and the Civio reproducible database. Heatwave variables are derived from AEMET annual heatwave reports. Forest-management expenditure variables are derived from Spanish forest-statistics annual reports. The repository contains the harmonised annual analytical series used in the manuscript, not the complete raw source databases.

## Licensing

- Original code in this repository: MIT License.
- Original derived tables, documentation and figures: CC BY 4.0.
- Source datasets retain their original terms and attribution requirements.

## DOI

This repository is prepared for the Zenodo–GitHub integration. Zenodo should be connected to this public repository and version `v1.3.0` archived from a GitHub release. Zenodo will then mint a DOI for that software release. The DOI should be inserted into the manuscript data-availability statement after the release has been processed.

The GitHub/Zenodo release is intentionally frozen at the 31 August 2026 Spanish snapshot. Wildfire developments reported after that date, including subsequent European events, belong to later prospective follow-up and do not alter the fitted models or this release.


## 31 August 2026 prospective snapshot

The model remains frozen through 2025. The prospectively archived sequence is 32 large wildfires (2 August), 39 (7 August), 42 (17 August), 43 (21 August publication-date estimate) and, following the MITECO wildfire follow-up meeting, 44 on 31 August 2026. The latest provisional national balance reports 264,656 ha of forest area affected since 1 January. At the same date in 2025, MITECO had recorded 60 large wildfires and an EFFIS-supported estimate of 346,443 ha. The 2026 count is therefore 2.52 times the frozen ARIMA point prediction (17.44) but remains below its 90% upper prediction limit (59.54). The 2015–2024 same-date mean was 17.8 large wildfires and 83,694 ha affected, placing the 2026 end-August snapshot at 2.47 and 3.16 times those historical means, respectively. These provisional values are stored separately from the annual modelling dataset and are never used to refit or select models.
