"""Recorrido del circuito a partir del SVG, con origen en la linea de meta.

Reemplaza a trazado_manual.py. El SVG trae la pista como un unico path cerrado de
coordenadas exactas, asi que no hay que extraer nada de pixeles ni poner waypoints a
mano, y ademas incluye la posicion de la linea de meta, que antes no habia forma de
saber mirando los videos.

La fase 0 queda en la meta: la fase deja de ser "que fraccion del clip va" y pasa a
ser "que fraccion de la vuelta va", que es lo que el punto necesita para caer en el
lugar correcto del plano.
"""
import numpy as np
from pathlib import Path

import svg_track

OUT = Path(__file__).parent / "out"
N_PUNTOS = 1000
LARGO_REAL_M = 4185.0

# Sentido de marcha sobre el path del SVG. Verificado visualmente con los marcadores
# numerados de out/trazado_svg.jpg contra los nombres de curva del plano.
INVERTIR = False


def construir(n=N_PUNTOS):
    """Devuelve (puntos, largo_en_unidades_svg), empezando en la linea de meta."""
    paths, meta = svg_track.cargar()
    p = max(paths, key=lambda q: np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
    if np.linalg.norm(p[0] - p[-1]) < 2:      # quitar el punto repetido del cierre
        p = p[:-1]
    if INVERTIR:
        p = p[::-1]

    # Rotar la secuencia para que arranque en el punto mas cercano a la meta.
    if meta is not None:
        p = np.roll(p, -int(np.argmin(np.linalg.norm(p - meta, axis=1))), axis=0)

    # Remuestrear por longitud de arco: un avance constante de fase equivale a un
    # avance constante de distancia sobre la pista.
    c = np.vstack([p, p[:1]])
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    s = np.linspace(0, d[-1], n, endpoint=False)
    return np.stack([np.interp(s, d, c[:, 0]), np.interp(s, d, c[:, 1])], 1), float(d[-1])


def a_pixeles(path, ancho, alto, margen=18):
    """Escalar el recorrido a un lienzo, devolviendo (puntos_px, escala, offset)."""
    lo, hi = path.min(0), path.max(0)
    esc = min((ancho - 2 * margen) / (hi[0] - lo[0]),
              (alto - 2 * margen) / (hi[1] - lo[1]))
    off = np.array([(ancho - (hi[0] - lo[0]) * esc) / 2 - lo[0] * esc,
                    (alto - (hi[1] - lo[1]) * esc) / 2 - lo[1] * esc])
    return path * esc + off, esc, off


def main():
    import cv2
    path, largo = construir()
    print(f"{len(path)} puntos | largo {largo:.1f} unidades SVG")
    print(f"escala: {LARGO_REAL_M/largo:.2f} m por unidad")

    W, H = 760, 800
    img = np.full((H, W, 3), 255, np.uint8)
    pts, _, _ = a_pixeles(path, W, H)
    pts = pts.astype(int)
    for i in range(len(pts)):
        cv2.line(img, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]),
                 (30, 30, 30), 5, cv2.LINE_AA)
    for k in range(12):
        p = pts[k * len(pts) // 12]
        cv2.circle(img, tuple(p), 13, (255, 255, 255), -1)
        cv2.circle(img, tuple(p), 13, (40, 40, 220), 2, cv2.LINE_AA)
        cv2.putText(img, str(k), (p[0] - 5 - (4 if k > 9 else 0), p[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "0 = linea de meta   (0..11 en sentido de marcha)", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "trazado_svg.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(f"-> {OUT / 'trazado_svg.jpg'}")


if __name__ == "__main__":
    main()
