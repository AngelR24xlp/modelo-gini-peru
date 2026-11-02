"""Medidas de desigualdad: Gini, curva de Lorenz y modelo log-normal."""
import numpy as np
from scipy.stats import norm


def gini(ingresos, pesos=None):
    """Coeficiente de Gini (ponderado) a partir de microdatos, p. ej. ENAHO con factor de expansión."""
    x = np.asarray(ingresos, dtype=float)
    w = np.ones_like(x) if pesos is None else np.asarray(pesos, dtype=float)
    orden = np.argsort(x)
    x, w = x[orden], w[orden]
    p = np.cumsum(w) / w.sum()
    l = np.cumsum(w * x) / np.sum(w * x)
    p, l = np.insert(p, 0, 0), np.insert(l, 0, 0)
    return 1 - np.sum((p[1:] - p[:-1]) * (l[1:] + l[:-1]))


def curva_lorenz(ingresos, pesos=None):
    x = np.asarray(ingresos, dtype=float)
    w = np.ones_like(x) if pesos is None else np.asarray(pesos, dtype=float)
    orden = np.argsort(x)
    x, w = x[orden], w[orden]
    p = np.insert(np.cumsum(w) / w.sum(), 0, 0)
    l = np.insert(np.cumsum(w * x) / np.sum(w * x), 0, 0)
    return p, l


# --- Distribución log-normal: G = 2*Phi(sigma/sqrt(2)) - 1 ---

def sigma_desde_gini(g):
    return np.sqrt(2) * norm.ppf((np.asarray(g) + 1) / 2)


def gini_desde_sigma(sigma):
    return 2 * norm.cdf(np.asarray(sigma) / np.sqrt(2)) - 1


def lorenz_lognormal(p, sigma):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return norm.cdf(norm.ppf(p) - sigma)


def theil_lognormal(sigma):
    return sigma ** 2 / 2


def atkinson_lognormal(sigma, epsilon=1.0):
    return 1 - np.exp(-epsilon * sigma ** 2 / 2)


def participaciones_quintiles(sigma):
    cortes = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    l = lorenz_lognormal(cortes, sigma)
    l[0], l[-1] = 0.0, 1.0
    return np.diff(l)


def ratio_palma(sigma):
    """Participación del 10% más rico / participación del 40% más pobre."""
    return (1 - lorenz_lognormal(0.9, sigma)) / lorenz_lognormal(0.4, sigma)
