#!/usr/bin/env python3
"""Ejecuta un unico pliegue PyMC en proceso aislado.

Evita que las compilaciones repetidas de PyTensor acumulen memoria durante la
validacion rodante. Es un auxiliar del paquete V1.0 del repositorio, no un modelo distinto.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from modelos_conteo_gif_v13 import (
    prepare_data, standardize, fit_predict_nb_ar1, summarize_forecast, stable_rng,
)

ap=argparse.ArgumentParser()
ap.add_argument('--input',type=Path,required=True)
ap.add_argument('--model-id',required=True)
ap.add_argument('--year',type=int,required=True)
ap.add_argument('--mode',choices=['rolling','external'],default='rolling')
ap.add_argument('--draws',type=int,default=300)
ap.add_argument('--tune',type=int,default=300)
ap.add_argument('--chains',type=int,default=2)
ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args()
d,specs=prepare_data(a.input)
cols=specs[a.model_id]
needed=['count_gif']+cols
if a.mode=='rolling':
    train=d[d.year<a.year].dropna(subset=needed)
else:
    train=d[d.year<=2023].dropna(subset=needed)
test=d[d.year==a.year].dropna(subset=needed)
if test.empty:
    raise SystemExit(f'Sin datos para {a.year}')
Xtr,Xte,_,_=standardize(train,test,cols)
obs=float(test.count_gif.iloc[0])
sims=fit_predict_nb_ar1(
    train.count_gif.to_numpy(int),Xtr,Xte,a.draws,a.tune,a.chains,
    stable_rng('pymc-isolated',a.model_id,a.year,a.mode),
)
row={'backend':'NB-AR1 (PyMC)','model_id':a.model_id,'year':a.year,
     'observed':obs,'mode':a.mode,**summarize_forecast(sims,obs)}
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(row,ensure_ascii=False),encoding='utf-8')
