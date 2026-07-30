"""Convertir fase-de-TIEMPO en fase-de-DISTANCIA sobre el trazado.

El problema: la etiqueta de un landmark es la fraccion de TIEMPO de la vuelta en que
fue recortado, pero el punto se dibuja a esa fraccion de DISTANCIA sobre el trazado.
Eso equivale a suponer velocidad constante, y no lo es: en la recta principal la moto
va a ~300 km/h y en la horquilla del Musee a ~70.

Consecuencia medida: en el arranque de vuelta, que es rapido, la moto recorre mas
distancia que fraccion de tiempo, asi que el punto se dibuja ATRASADO hasta 172 m.
Es exactamente el tramo hasta la primera curva grande.

Como se estima el perfil de velocidad: de la geometria del propio trazado. Donde el
circuito es recto la moto va rapido y donde cierra, lento. No es telemetria real, asi
que los valores extremos son una aproximacion; por eso la correccion tiene un
parametro de FUERZA para poder graduarla mirando el resultado.

Esta correccion es puramente de POSICIONAMIENTO: cambia donde se dibuja el punto, no
que landmark se detecta. Las metricas de deteccion se miden en fase-tiempo y no se
ven afectadas.
"""
import numpy as np

import trazado_svg as T

V_MAX_KMH = 300.0        # recta principal
V_MIN_KMH = 70.0         # horquilla mas cerrada
EXP = 0.45               # cuan brusca es la caida de velocidad con la curvatura
SUAVIZADO = 41           # puntos de ventana para suavizar la curvatura


def curvatura(p, k=12):
    """Cuanto gira la direccion cada k puntos, normalizado por la longitud."""
    a = np.roll(p, k, 0) - p
    b = p - np.roll(p, -k, 0)
    ang = np.arctan2(a[:, 1], a[:, 0]) - np.arctan2(b[:, 1], b[:, 0])
    ang = (ang + np.pi) % (2 * np.pi) - np.pi
    paso = np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1).mean()
    c = np.abs(ang) / (k * paso + 1e-9)
    # Suavizado circular: la vuelta cierra, asi que los extremos se tocan.
    n = SUAVIZADO
    return np.convolve(np.r_[c[-n:], c, c[:n]], np.ones(n) / n, "same")[n:-n]


def _perfil(path):
    """Devuelve (fase_distancia, fase_tiempo) a lo largo de la vuelta."""
    c = curvatura(path)
    v = V_MAX_KMH - (V_MAX_KMH - V_MIN_KMH) * (c / (c.max() + 1e-9)) ** EXP
    dt = 1.0 / v                       # tiempo por paso; la escala no importa
    t = np.r_[0.0, np.cumsum(dt)[:-1]]
    t /= t[-1] + dt[-1]                # normalizar a [0,1)
    s = np.arange(len(path)) / len(path)
    return s, t


_cache = {}


def a_distancia(fase_tiempo, fuerza=1.0):
    """Fase-tiempo -> fase-distancia. `fuerza` 0 no corrige nada, 1 aplica el modelo."""
    if fuerza == 0:
        return np.asarray(fase_tiempo, float) % 1.0
    if "p" not in _cache:
        path, _ = T.construir()
        _cache["p"] = _perfil(path)
    s, t = _cache["p"]
    f = np.asarray(fase_tiempo, float) % 1.0
    # Invertir t(s): para cada fase de tiempo, que fraccion de distancia le toca.
    s_corr = np.interp(f, t, s, period=1.0)
    d = (s_corr - f + 0.5) % 1.0 - 0.5          # correccion por el camino corto
    return (f + fuerza * d) % 1.0


if __name__ == "__main__":
    path, _ = T.construir()
    s, t = _perfil(path)
    c = curvatura(path)
    recto = c < np.percentile(c, 25)
    dt = np.diff(np.r_[t, 1.0])
    print(f"tiempo en el 25% mas recto del trazado: {100*dt[recto].sum():.0f}% "
          f"de la vuelta  (con velocidad constante seria 25%)")

    print(f"\n{'fase':>6} {'corregida':>11} {'avance':>9} {'avance':>9}")
    print(f"{'':>6} {'':>11} {'(fase)':>9} {'(metros)':>9}")
    for f in np.arange(0.0, 1.0, 0.1):
        g = float(a_distancia(f))
        d = (g - f + 0.5) % 1.0 - 0.5
        print(f"{f:6.2f} {g:11.3f} {d:+9.3f} {d*4185:+8.0f}m")
