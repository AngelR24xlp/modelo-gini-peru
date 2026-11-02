"""Carga y actualización de datos de Perú (Gini y variables macroeconómicas)."""
from pathlib import Path

import pandas as pd
import requests

RUTA_DATOS = Path(__file__).resolve().parents[1] / "data" / "peru_gini_macro_2000_2025.csv"

OWID = {
    "gini": "economic-inequality-gini-index",
    "pib_pc_ppp": "gdp-per-capita-worldbank",
    "desempleo": "unemployment-rate",
}
BANCO_MUNDIAL = {
    "crec_pib": "NY.GDP.MKTP.KD.ZG",
    "inflacion": "FP.CPI.TOTL.ZG",
}


def cargar_datos(ruta=RUTA_DATOS):
    return pd.read_csv(ruta, index_col="anio")


def _serie_owid(slug, nombre):
    url = f"https://ourworldindata.org/grapher/{slug}.csv?country=~PER"
    df = pd.read_csv(url, storage_options={"User-Agent": "modelo-gini-peru"})
    df = df[df["Entity"] == "Peru"].set_index("Year")
    return df.iloc[:, 2].rename(nombre)


def _serie_banco_mundial(indicador, nombre):
    url = f"https://api.worldbank.org/v2/country/PER/indicator/{indicador}"
    r = requests.get(url, params={"format": "json", "date": "2000:2025", "per_page": 100}, timeout=60)
    r.raise_for_status()
    return pd.Series({int(x["date"]): x["value"] for x in r.json()[1]}, name=nombre)


def actualizar_datos(ruta=RUTA_DATOS):
    """Descarga las series desde Our World in Data (PIP/Banco Mundial) y la API del Banco Mundial."""
    series = [_serie_owid(s, n) for n, s in OWID.items()]
    series += [_serie_banco_mundial(i, n) for n, i in BANCO_MUNDIAL.items()]
    df = pd.concat(series, axis=1).sort_index().loc[2000:2025]
    df.index.name = "anio"
    df.round(6).to_csv(ruta)
    return df
