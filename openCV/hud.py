"""Detectar los graficos fijos de la transmision para excluirlos del fondo.

El problema que resuelve: un candidato recortado sobre el cartel "MotoGP | Le Mans"
ganaba en TODOS los frames de consulta con ~40 inliers, porque ese grafico aparece
identico en todos los videos del paquete de transmision. Tambien matcheaba contra el
recuadro "Aprilia OnBoard": dos cajas claras con texto oscuro que SIFT empareja sin
problema. Nada de eso es circuito.

El filtro de unicidad no lo atrapaba porque el cartel de MotoGP solo se muestra en
los primeros ~20 s de la vuelta de referencia, o sea que daba un pico unico y limpio.

Como se detectan solos: un grafico superpuesto es ESTATICO pixel a pixel, mientras
que el paisaje visto a 300 km/h cambia todo el tiempo. La desviacion temporal de cada
pixel separa las dos cosas sin ambiguedad.
"""
import cv2
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent / "data"

N_MUESTRAS = 90
UMBRAL_STD = 6.0     # por debajo de esto el pixel practicamente no cambia
DILATA = 9           # margen alrededor, para cubrir bordes y antialiasing


def mascara(video, t0, t1, cache={}):
    """True donde hay grafico fijo. Se cachea por video."""
    clave = (str(video), round(t0, 1), round(t1, 1))
    if clave in cache:
        return cache[clave]

    cap = cv2.VideoCapture(str(DATA / video))
    acc = []
    for t in np.linspace(t0, t1, N_MUESTRAS):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * 25))
        ok, f = cap.read()
        if ok:
            acc.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if len(acc) < 10:
        raise RuntimeError(f"pocas muestras en {video}")

    std = np.stack(acc).std(0)
    m = (std < UMBRAL_STD).astype(np.uint8)
    # Quitar motas sueltas y despues ensanchar: interesa cubrir el grafico entero,
    # no marcar pixeles aislados que casualmente no variaron.
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.dilate(m, np.ones((DILATA, DILATA), np.uint8))
    cache[clave] = m
    return m


def filtrar_kp(kps, descs, hud, offset_y):
    """Descartar keypoints que caen sobre un grafico.

    `offset_y` es el desplazamiento del recorte de fondo dentro del frame completo,
    porque los keypoints vienen en coordenadas del recorte y la mascara esta en
    coordenadas del frame.
    """
    if descs is None or len(descs) == 0:
        return kps, descs
    h, w = hud.shape
    ok = []
    for i, k in enumerate(kps):
        x, y = int(round(k.pt[0])), int(round(k.pt[1] + offset_y))
        if 0 <= y < h and 0 <= x < w and hud[y, x]:
            continue
        ok.append(i)
    if not ok:
        return [], np.zeros((0, descs.shape[1]), np.float32)
    return [kps[i] for i in ok], descs[ok]


def fraccion_hud(hud, box):
    """Que fraccion de una caja candidata cae sobre grafico fijo."""
    h, w = hud.shape
    x0, y0, x1, y1 = box
    sub = hud[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
    return float(sub.mean()) if sub.size else 1.0
