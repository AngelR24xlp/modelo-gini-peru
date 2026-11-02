"""Modelos econométricos del coeficiente de Gini para Perú."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

from .gini import gini, sigma_desde_gini


def preparar_variables(df):
    d = df.copy()
    d["ln_pib_pc"] = np.log(d["pib_pc_ppp"])
    d["ln_pib_pc2"] = d["ln_pib_pc"] ** 2
    d["gini_lag"] = d["gini"].shift(1)
    d["covid"] = (d.index == 2020).astype(float)
    return d


ESPECIFICACIONES = {
    # Curva de Kuznets ampliada con variables del ciclo económico
    "kuznets_macro": ["ln_pib_pc", "ln_pib_pc2", "desempleo", "inflacion"],
    # Modelo dinámico de ajuste parcial
    "dinamico": ["gini_lag", "crec_pib", "desempleo", "inflacion"],
}


def estimar_ols(df, regresores, hasta=2024):
    d = df.loc[:hasta, ["gini"] + regresores].dropna()
    X = sm.add_constant(d[regresores])
    return sm.OLS(d["gini"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})


def predecir(modelo, df, regresores, anios):
    X = sm.add_constant(df.loc[anios, regresores], has_constant="add")
    pred = modelo.get_prediction(X).summary_frame(alpha=0.05)
    pred.index = anios
    return pred[["mean", "obs_ci_lower", "obs_ci_upper"]].rename(
        columns={"mean": "gini_estimado", "obs_ci_lower": "ic95_inf", "obs_ci_upper": "ic95_sup"})


def pronostico_arima(serie, orden=(1, 1, 0), pasos=1):
    serie = serie.dropna()
    serie.index = pd.period_range(start=str(serie.index[0]), periods=len(serie), freq="Y")
    modelo = ARIMA(serie, order=orden).fit()
    f = modelo.get_forecast(pasos).summary_frame(alpha=0.05)
    f.index = f.index.year
    return modelo, f[["mean", "mean_ci_lower", "mean_ci_upper"]].rename(
        columns={"mean": "gini_estimado", "mean_ci_lower": "ic95_inf", "mean_ci_upper": "ic95_sup"})


def microsimulacion(gini_obj, ingreso_medio, n=200_000, semilla=2025):
    """Simula ingresos log-normales consistentes con un Gini y un ingreso medio dados."""
    sigma = float(sigma_desde_gini(gini_obj))
    mu = np.log(ingreso_medio) - sigma ** 2 / 2
    ingresos = np.random.default_rng(semilla).lognormal(mu, sigma, n)
    return ingresos, gini(ingresos)


def tabla_coeficientes(modelos):
    filas = []
    for nombre, m in modelos.items():
        for var in m.params.index:
            filas.append({"modelo": nombre, "variable": var, "coef": m.params[var],
                          "error_std": m.bse[var], "p_valor": m.pvalues[var]})
        filas.append({"modelo": nombre, "variable": "R2_ajustado", "coef": m.rsquared_adj})
        filas.append({"modelo": nombre, "variable": "AIC", "coef": m.aic})
        filas.append({"modelo": nombre, "variable": "N", "coef": m.nobs})
    return pd.DataFrame(filas)
