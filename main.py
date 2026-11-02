"""Modelo económico del coeficiente de Gini para Perú, 2020-2025.

Uso:
    python main.py              # usa los datos guardados en data/
    python main.py --actualizar # descarga de nuevo los datos
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.datos import actualizar_datos, cargar_datos
from src.gini import (atkinson_lognormal, lorenz_lognormal, participaciones_quintiles,
                      ratio_palma, sigma_desde_gini, theil_lognormal)
from src.modelo import (ESPECIFICACIONES, estimar_ols, microsimulacion, predecir,
                        preparar_variables, pronostico_arima, tabla_coeficientes)

SALIDA = Path(__file__).resolve().parent / "outputs"
FIG, TAB = SALIDA / "figuras", SALIDA / "tablas"
PERIODO = list(range(2020, 2026))
ULTIMO_OBSERVADO = 2024


def main(actualizar=False):
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    df = preparar_variables(actualizar_datos() if actualizar else cargar_datos())

    # 1. Estimación econométrica (muestra 2000-2024)
    modelos = {n: estimar_ols(df, r, hasta=ULTIMO_OBSERVADO) for n, r in ESPECIFICACIONES.items()}
    tabla_coeficientes(modelos).round(5).to_csv(TAB / "coeficientes_modelos.csv", index=False)
    with open(TAB / "resumen_regresiones.txt", "w") as f:
        for n, m in modelos.items():
            f.write(f"{'=' * 78}\nModelo: {n}\n{m.summary()}\n\n")

    preds = {n: predecir(modelos[n], df, ESPECIFICACIONES[n], PERIODO) for n in modelos}
    arima, pred_arima = pronostico_arima(df["gini"].loc[:ULTIMO_OBSERVADO], pasos=1)

    m = modelos["kuznets_macro"]
    punto_giro = np.exp(-m.params["ln_pib_pc"] / (2 * m.params["ln_pib_pc2"]))

    # 2. Serie final 2020-2025: observado hasta 2024 y estimado (promedio de modelos) en 2025
    est_2025 = pd.concat([preds["kuznets_macro"].loc[[2025]], preds["dinamico"].loc[[2025]],
                          pred_arima.loc[[2025]]]).mean()
    res = df.loc[PERIODO, ["gini", "pib_pc_ppp", "crec_pib", "desempleo", "inflacion"]].copy()
    res["fuente_gini"] = np.where(res.index <= ULTIMO_OBSERVADO, "observado (PIP-Banco Mundial/ENAHO)",
                                  "estimado (promedio de modelos)")
    res.loc[2025, "gini"] = est_2025["gini_estimado"]
    for n, p in preds.items():
        res[f"gini_{n}"] = p["gini_estimado"]
    res.loc[2025, "gini_arima"] = pred_arima.loc[2025, "gini_estimado"]
    res["ic95_inf_2025"] = np.where(res.index == 2025, est_2025["ic95_inf"], np.nan)
    res["ic95_sup_2025"] = np.where(res.index == 2025, est_2025["ic95_sup"], np.nan)

    # 3. Indicadores derivados (supuesto de distribución log-normal del ingreso)
    sigma = sigma_desde_gini(res["gini"])
    res["sigma_lognormal"] = sigma
    res["theil"] = theil_lognormal(sigma)
    res["atkinson_e1"] = atkinson_lognormal(sigma, 1.0)
    res["ratio_palma"] = ratio_palma(sigma)
    quint = pd.DataFrame([participaciones_quintiles(s) for s in sigma], index=res.index,
                         columns=[f"Q{i}" for i in range(1, 6)])
    res["ratio_Q5_Q1"] = quint["Q5"] / quint["Q1"]
    res["var_gini_pp"] = res["gini"].diff() * 100
    res.round(4).to_csv(TAB / "gini_peru_2020_2025.csv")
    (quint * 100).round(2).to_csv(TAB / "participacion_quintiles_2020_2025.csv")

    # 4. Microsimulación: valida que el Gini simulado reproduce el objetivo
    micro = []
    for a in PERIODO:
        _, g_sim = microsimulacion(res.loc[a, "gini"], res.loc[a, "pib_pc_ppp"])
        micro.append({"anio": a, "gini_objetivo": res.loc[a, "gini"], "gini_simulado": g_sim})
    pd.DataFrame(micro).round(4).to_csv(TAB / "microsimulacion.csv", index=False)

    anios_ajuste = list(range(2001, 2026))
    ajustes = {n: predecir(modelos[n], df, ESPECIFICACIONES[n], anios_ajuste) for n in modelos}
    graficos(df, res, ajustes, quint, m)

    print("\nCOEFICIENTE DE GINI - PERU 2020-2025")
    print(res[["gini", "fuente_gini", "var_gini_pp", "theil", "ratio_palma"]].round(4).to_string())
    print(f"\nGini 2025 estimado: {est_2025['gini_estimado']:.4f} "
          f"(IC95%: {est_2025['ic95_inf']:.4f} - {est_2025['ic95_sup']:.4f})")
    print(f"Punto de giro de la curva de Kuznets: US$ {punto_giro:,.0f} PPA per cápita")
    print(f"R2 ajustado - Kuznets: {modelos['kuznets_macro'].rsquared_adj:.3f} | "
          f"Dinámico: {modelos['dinamico'].rsquared_adj:.3f}")
    print(f"\nResultados guardados en {SALIDA}")


def graficos(df, res, ajustes, quint, m):
    plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})

    # Evolución del Gini
    fig, ax = plt.subplots(figsize=(10, 5.5))
    obs = df["gini"].loc[2000:ULTIMO_OBSERVADO]
    ax.plot(obs.index, obs, "o-", color="#1f4e79", label="Gini observado")
    for n, c in [("kuznets_macro", "#c0504d"), ("dinamico", "#9bbb59")]:
        ax.plot(ajustes[n].index, ajustes[n]["gini_estimado"], "--", color=c, label=f"Modelo {n}")
    ax.errorbar(2025, res.loc[2025, "gini"],
                yerr=[[res.loc[2025, "gini"] - res.loc[2025, "ic95_inf_2025"]],
                      [res.loc[2025, "ic95_sup_2025"] - res.loc[2025, "gini"]]],
                fmt="D", color="#ff7f0e", capsize=6, ms=9, label="Estimación 2025 (IC 95%)")
    ax.axvspan(2019.5, 2025.5, color="grey", alpha=0.08)
    ax.set(title="Perú: coeficiente de Gini del ingreso per cápita, 2000-2025",
           xlabel="Año", ylabel="Gini")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "01_evolucion_gini.png")
    plt.close(fig)

    # Curvas de Lorenz 2020-2025
    fig, ax = plt.subplots(figsize=(7, 7))
    p = np.linspace(0, 1, 201)
    ax.plot(p, p, "k-", lw=1, label="Igualdad perfecta")
    for a, c in zip(PERIODO, plt.cm.viridis(np.linspace(0, 0.9, len(PERIODO)))):
        ax.plot(p, lorenz_lognormal(p, res.loc[a, "sigma_lognormal"]), color=c,
                label=f"{a} (G={res.loc[a, 'gini']:.3f})")
    ax.set(title="Curvas de Lorenz (log-normal), Perú 2020-2025",
           xlabel="Proporción acumulada de población", ylabel="Proporción acumulada del ingreso")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "02_curvas_lorenz.png")
    plt.close(fig)

    # Curva de Kuznets
    fig, ax = plt.subplots(figsize=(9, 5.5))
    d = df.loc[:ULTIMO_OBSERVADO].dropna(subset=["gini"])
    ax.scatter(d["pib_pc_ppp"], d["gini"], c=d.index, cmap="viridis")
    for a in PERIODO[:-1]:
        ax.annotate(str(a), (d.loc[a, "pib_pc_ppp"], d.loc[a, "gini"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    y = np.linspace(d["pib_pc_ppp"].min(), d["pib_pc_ppp"].max(), 100)
    ly = np.log(y)
    ajuste = (m.params["const"] + m.params["ln_pib_pc"] * ly + m.params["ln_pib_pc2"] * ly ** 2
              + m.params["desempleo"] * df["desempleo"].mean() + m.params["inflacion"] * df["inflacion"].mean())
    ax.plot(y, ajuste, "r--", label="Curva de Kuznets estimada")
    ax.set(title="Perú: Gini vs. PIB per cápita (US$ PPA 2021), 2000-2024",
           xlabel="PIB per cápita (US$ PPA)", ylabel="Gini")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "03_curva_kuznets.png")
    plt.close(fig)

    # Participación por quintiles
    fig, ax = plt.subplots(figsize=(10, 5.5))
    base = np.zeros(len(quint))
    for q, c in zip(quint.columns, plt.cm.Blues(np.linspace(0.3, 0.95, 5))):
        ax.bar(quint.index, quint[q] * 100, bottom=base, color=c, label=q)
        base += quint[q].values * 100
    ax.set(title="Participación en el ingreso por quintiles (log-normal), 2020-2025",
           xlabel="Año", ylabel="% del ingreso total")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(FIG / "04_quintiles.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actualizar", action="store_true", help="Descargar datos actualizados")
    main(parser.parse_args().actualizar)
