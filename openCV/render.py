"""Video de salida: el onboard a la izquierda, el plano con el punto a la derecha.

La posicion sale de localizar.py, que decide solo por comparacion de imagenes contra
la base de landmarks. Cuando ningun landmark aparece en el frame, el sistema no sabe
donde esta: el punto se dibuja hueco y en gris, con la ultima posicion conocida, y el
cartel lo dice. Preferimos mostrar la incertidumbre antes que inventar una posicion.

Uso: python render.py [aprilia|ktm|...] [peso_min] [n_lm_min]

Los dos ultimos son el criterio de aceptacion del consenso. Con 0 1 el punto esta
encendido casi siempre pero salta mas seguido a un lugar equivocado; con 40 2 se
apaga mas pero casi no se equivoca. El archivo de salida lleva el criterio en el
nombre, para poder comparar las dos versiones.
"""
import sys
import cv2
import numpy as np
from pathlib import Path

import trazado_svg as T
import velocidad

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "out"

VIDEOS = {
    "yamaha":  "lemans2018_yamaha.mp4",
    "ktm":     "lemans2018_ktm.mp4",
    "aprilia": "lemans2018_aprilia.mp4",
}

# Desfasaje entre el inicio de la vuelta de referencia y la linea de meta del plano.
# La referencia arranca en 6.4s, que no es exactamente el cruce de meta.
FASE_0 = 0.0
# Correccion de velocidad no constante. Las etiquetas son fraccion de TIEMPO pero el
# punto se dibuja por DISTANCIA, asi que en los tramos rapidos el punto queda
# atrasado (hasta 195 m en la recta de arranque). 0 = sin corregir, 1 = modelo pleno.
FUERZA_VEL = 0.6
ALTO = 460
# Umbral de inliers para aceptar una deteccion. Elegido con el barrido de
# localizar2.py: 83.4% de cobertura con 81% de aciertos sobre el video reservado.
# Se prefiere 14 sobre 18 porque el atraso visible no viene de las detecciones (van
# +0.5 s ADELANTADAS) sino de los huecos sin deteccion, donde el punto se congela
# mientras la moto sigue avanzando. Menos huecos, menos atraso aparente.
UMBRAL = 14
# Criterio para la salida por consenso (localizar4.py): peso acumulado minimo y
# cuantos landmarks distintos deben coincidir en la misma posicion.
PESO_MIN, N_LM_MIN = 40, 2
INLIERS_FUERTE = 45     # a partir de aca el punto se dibuja a full
# Entre dos detecciones confirmadas se interpola la posicion en vez de congelar el
# punto. No inventa una posicion de la nada: une dos puntos MEDIDOS, y con el
# criterio estricto ambos extremos son correctos por construccion. Se dibuja hueco
# para distinguir lo interpolado de lo medido.
INTERPOLAR = False
MAX_INTERP_S = 5.0      # huecos mas largos que esto no se rellenan
MAX_AVANCE = 0.12       # avance de fase maximo admisible en un hueco (~11 s de vuelta)
# A los pocos segundos sin confirmacion el punto se desvanece: si no, una posicion
# vieja se sigue leyendo como si fuera actual.
VIDA_S = 3.0


def panel_mapa(path, ancho):
    """Plano dibujado desde la geometria vectorial del SVG.

    Se dibuja el trazado en vez de cargar una imagen: las coordenadas son las mismas
    que las del punto, asi que no hay que alinear nada y no hay error de registro.
    """
    lienzo = np.full((ALTO, ancho, 3), 255, np.uint8)
    pts, _, _ = T.a_pixeles(path, ancho, ALTO, margen=26)
    pts = pts.astype(int)
    for i in range(len(pts)):
        cv2.line(lienzo, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]),
                 (55, 55, 55), 6, cv2.LINE_AA)
    # Linea de meta: la fase 0 del recorrido, que el SVG trae marcada.
    p0, p1 = pts[0], pts[3]
    n = np.array([-(p1[1] - p0[1]), p1[0] - p0[0]], float)
    n = n / (np.linalg.norm(n) + 1e-9) * 11
    cv2.line(lienzo, tuple((p0 - n).astype(int)), tuple((p0 + n).astype(int)),
             (255, 255, 255), 5, cv2.LINE_AA)
    cv2.line(lienzo, tuple((p0 - n).astype(int)), tuple((p0 + n).astype(int)),
             (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(lienzo, "META", (p0[0] - 46, p0[1] + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (90, 90, 90), 1, cv2.LINE_AA)
    return lienzo, pts


def interpolar(tq, frac):
    """Rellenar los huecos entre detecciones confirmadas avanzando por la pista.

    Devuelve (fase, medido). `medido` distingue lo que se detecto de lo que se
    interpolo, para poder dibujarlos distinto y no presentar una cosa como la otra.
    Solo se rellena si el hueco es corto y el avance de fase es fisicamente plausible;
    si no, se deja vacio y el punto se apaga.
    """
    out = np.array(frac, float).copy()
    medido = ~np.isnan(out)
    idx = np.flatnonzero(medido)
    for a, b in zip(idx, idx[1:]):
        dt = tq[b] - tq[a]
        if dt > MAX_INTERP_S:
            continue
        # Avance por el camino corto hacia adelante, respetando el cierre de vuelta.
        d = (out[b] - out[a]) % 1.0
        if d > MAX_AVANCE:
            continue
        for k in range(a + 1, b):
            out[k] = (out[a] + d * (tq[k] - tq[a]) / dt) % 1.0
    return out, medido


def main():
    q = sys.argv[1] if len(sys.argv) > 1 else "aprilia"
    peso_min = int(sys.argv[2]) if len(sys.argv) > 2 else PESO_MIN
    n_lm_min = int(sys.argv[3]) if len(sys.argv) > 3 else N_LM_MIN
    sufijo = f"_p{peso_min}n{n_lm_min}"
    # Preferir la salida por consenso si existe; si no, la de landmark unico.
    p4 = OUT / f"pos4_{q}.npz"
    if p4.exists():
        z = np.load(p4)
        tq, todos = z["tq"], z["frac_todos"]
        inl = z["peso"]
        # Criterio elegido con el barrido de localizar4.py: 90.6% de cobertura con
        # 81% de aciertos. Se privilegia cobertura porque el atraso que se percibe
        # viene de los huecos, no de las detecciones.
        frac = np.where((z["peso"] >= peso_min) & (z["n_lm"] >= n_lm_min),
                        todos, np.nan)
    else:
        z = np.load(OUT / f"pos_{q}.npz")
        tq, inl = z["tq"], z["inliers"]
        todos = z["frac_todos"] if "frac_todos" in z else z["frac"]
        frac = np.where(inl >= UMBRAL, todos, np.nan)

    # Pasar de fase-tiempo (como estan etiquetados los landmarks) a fase-distancia
    # (que es lo que el trazado necesita). Sin esto el punto queda atrasado en los
    # tramos rapidos, sobre todo en la recta hasta la primera curva grande.
    if FUERZA_VEL:
        ok = ~np.isnan(frac)
        frac = frac.astype(float).copy()
        frac[ok] = velocidad.a_distancia(frac[ok], FUERZA_VEL)

    medido = ~np.isnan(frac)
    if INTERPOLAR:
        frac, medido = interpolar(tq, frac)
        print(f"medidos {medido.sum()}, interpolados "
              f"{(~np.isnan(frac)).sum() - medido.sum()} de {len(tq)}")

    path, largo = T.construir()
    mapa, pts = panel_mapa(path, 420)

    cap = cv2.VideoCapture(str(DATA / VIDEOS[q]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W_VID = int(ALTO * 16 / 9)
    salida = cv2.VideoWriter(str(OUT / f"salida_{q}{sufijo}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (W_VID + mapa.shape[1], ALTO))

    # Se recorren TODOS los frames del onboard, no solo los muestreados: si no, el
    # video sale con un frame cada 0.4s escrito a 25 fps y se reproduce 10x rapido.
    # La posicion de cada frame es la de la muestra mas cercana en el tiempo.
    ult, ult_t = None, -1e9
    n_conocidos = n_frames = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(tq[0] * 25))
    t = tq[0]
    while t <= tq[-1]:
        ok, f = cap.read()
        if not ok:
            break
        k = int(np.argmin(np.abs(tq - t)))
        n_frames += 1
        vid = cv2.resize(f, (W_VID, ALTO))

        panel = mapa.copy()
        hay = not np.isnan(frac[k])
        conocido = hay and medido[k]
        if hay:
            ult, ult_t = float(frac[k]), t
            n_conocidos += 1
        edad = t - ult_t if ult is not None else None
        if ult is not None and edad <= VIDA_S:
            i = int(((ult + FASE_0) % 1.0) * len(pts))
            p = tuple(pts[i])
            if conocido:
                # El tamano y la opacidad siguen a los inliers: un match debil se
                # dibuja debil. Es informacion real, no adorno.
                fuerza = min(1.0, (inl[k] - UMBRAL) / (INLIERS_FUERTE - UMBRAL))
                r = int(7 + 5 * fuerza)
                capa = panel.copy()
                cv2.circle(capa, p, r + 2, (255, 255, 255), -1)
                cv2.circle(capa, p, r, (40, 40, 235), -1)
                a = 0.45 + 0.55 * fuerza
                panel[:] = cv2.addWeighted(capa, a, panel, 1 - a, 0)
            elif hay:
                # Interpolado entre dos detecciones: anillo hueco, no disco lleno.
                cv2.circle(panel, p, 10, (255, 255, 255), -1)
                cv2.circle(panel, p, 10, (70, 140, 230), 2, cv2.LINE_AA)
            else:
                # Posicion vieja: se desvanece con la edad en vez de quedar fija,
                # para que no se lea como si fuera actual.
                a = 1.0 - edad / VIDA_S
                capa = panel.copy()
                cv2.circle(capa, p, 9, (150, 150, 150), 2, cv2.LINE_AA)
                panel[:] = cv2.addWeighted(capa, a, panel, 1 - a, 0)

        if conocido:
            txt, col = f"landmark detectado  (peso {int(inl[k])})", (40, 130, 40)
        elif hay:
            txt, col = "interpolado entre dos detecciones", (70, 140, 230)
        elif ult is not None and edad <= VIDA_S:
            txt, col = f"sin landmark  (posicion de hace {edad:.1f}s)", (120, 120, 120)
        else:
            txt, col = "sin landmark a la vista", (120, 120, 120)
        cv2.rectangle(panel, (0, ALTO - 34), (panel.shape[1], ALTO), (245, 245, 245), -1)
        cv2.putText(panel, txt, (12, ALTO - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, col, 1, cv2.LINE_AA)
        cv2.putText(vid, f"{q}  t={t:.1f}s", (14, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        salida.write(np.hstack([vid, panel]))
        t += 1.0 / fps

    cap.release()
    salida.release()
    print(f"{n_frames} frames escritos ({n_frames/fps:.0f}s a {fps:.0f} fps)")
    print(f"posicion conocida en {100*n_conocidos/max(n_frames,1):.0f}% del tiempo")
    print(f"-> {OUT / f'salida_{q}{sufijo}.mp4'}")


if __name__ == "__main__":
    main()
