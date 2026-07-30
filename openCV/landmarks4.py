"""Base que ademas conserva los landmarks ambiguos, con todas sus posiciones.

Hasta ahora un candidato que se encendia en dos puntos del circuito se descartaba:
una tribuna o un cartel repetido no dice donde estas. Por eso los sectores pegados a
la linea de meta quedaban mal servidos, que es justo donde el fondo se repite mas.

Pero un cartel repetido SI aporta si se lo mira junto a otro: la combinacion de dos
estructuras ambiguas suele darse en un unico punto de la vuelta. Asi que en vez de
tirarlos, se guardan con la LISTA de posiciones donde aparecen, y la desambiguacion
se hace en la consulta, buscando la posicion donde varios coinciden.

Un candidato que se enciende en demasiados lugares (mas de MAX_POS) si se descarta:
es textura generica y no aporta ni en combinacion.

Los corrimientos por fuente se aplican aca (medidos con realinear.py sobre la base
anterior): las cinco vueltas no arrancan exactamente en el mismo punto de la pista.

NOTA sobre la ventana de vuelta. Una version anterior usaba el tramo continuo sin
cortes de cada clip (6.4 s hasta ~112 s, distinto en cada video). Eso traia dos
problemas:

  - La fase quedaba estirada, porque se repartia una vuelta de 94 s sobre ~106 s de
    metraje. Corria el punto sobre el plano hasta 11 s.
  - El chequeo de unicidad describia mas de una vuelta de metraje, asi que un mismo
    lugar aparecia dos veces y se descartaba como ambiguo sin serlo.

Usar exactamente una vuelta (VUELTA_T0..VUELTA_T1, igual para todos) arregla las dos.
El costo es que ya no se toman candidatos del tramo de aproximacion ni del posterior
a la meta, que en la base vieja aportaban 58 de 476 landmarks; pero esos cubren el
mismo pedazo de pista que el final y el arranque de la vuelta, asi que son redundantes.

El archivo out/landmarks_multi.npz que hay hoy se construyo con la version vieja y se
corrigio despues con recalibrar.py. Reconstruir con este script da una base
equivalente sin ese segundo paso.
"""
import cv2
import numpy as np
from pathlib import Path

import hud
import landmarks2 as L

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "out"

# Ventana de la VUELTA, igual para los seis clips: los seis comparten el mismo
# formato de produccion (placa de titulos, aproximacion, vuelta lanzada). No es el
# tramo continuo sin cortes del clip, que arranca en 6.4 s e incluye la aproximacion
# antes de cruzar la linea y un resto despues; usar ese tramo estiraba la fase y
# corria el punto sobre el plano hasta 11 s.
VUELTA_T0, VUELTA_T1 = 10.4, 104.4
LARGO = VUELTA_T1 - VUELTA_T0

FUENTES_DEFECTO = ["yamaha", "ktm", "suzuki", "ducati", "honda"]
FUENTES = [(f"lemans2018_{n}.mp4", VUELTA_T0, VUELTA_T1) for n in FUENTES_DEFECTO]


def usar_fuentes(nombres):
    """Elegir que videos aportan landmarks. `nombres` son cortos: 'yamaha', 'ktm'...

    Hacen falta al menos dos: un candidato tiene que poder confirmarse desde otra
    moto, que es el filtro que descarta los recortes del piloto y del carenado.
    """
    global FUENTES
    if len(nombres) < 2:
        raise ValueError("hacen falta al menos 2 fuentes para el filtro cruzado")
    FUENTES = [(f"lemans2018_{n}.mp4", VUELTA_T0, VUELTA_T1) for n in nombres]

# Corrimiento residual de cada fuente respecto del marco comun (Yamaha), en segundos,
# medido por deteccion directa con realinear.py. Es chico (< 0.7 s) porque el grueso
# de la discrepancia entre fuentes venia de que cada clip usaba una ventana distinta,
# y eso ya no pasa al ser VUELTA_T0/VUELTA_T1 comunes.
OFFSET_S = {"lemans2018_yamaha.mp4": +0.10, "lemans2018_ktm.mp4": -0.14,
            "lemans2018_suzuki.mp4": +0.33, "lemans2018_ducati.mp4": +0.70,
            "lemans2018_honda.mp4": -0.05}

CAND_EVERY_S = 1.0
MAX_POS = 3          # mas de esto es textura generica, no sirve ni combinada
PICO_FRAC = 0.55     # un pico cuenta como posicion si llega a esta fraccion del maximo


def picos(v, ts, frac_min=PICO_FRAC, sep=L.FAR_S):
    """Posiciones donde el candidato se enciende, separadas entre si."""
    top = v.max()
    if top <= 0:
        return []
    out = []
    orden = np.argsort(v)[::-1]
    for i in orden:
        if v[i] < frac_min * top:
            break
        if all(abs(ts[i] - ts[j]) > sep for j in out):
            out.append(int(i))
        if len(out) > MAX_POS:
            break
    return out


def main():
    det_frame, det_tpl = L.sift(L.NFEAT_FRAME), L.sift(L.NFEAT_TPL)

    vueltas = {}
    for v, t0, t1 in FUENTES:
        ts, kps, ds = L.describir_vuelta(v, t0, t1, det_frame)
        fl, mapa = L.indice(ds)
        vueltas[v] = dict(ts=ts, kps=kps, ds=ds, fl=fl, mapa=mapa, t0=t0, t1=t1,
                          hud=hud.mascara(v, t0, t1))
        print(f"{v}: {len(ts)} posiciones")

    kept, motivos, n_cand = [], {}, 0

    def rechazo(m):
        motivos[m] = motivos.get(m, 0) + 1

    for v, t0, t1 in FUENTES:
        # .get: un video sin corrimiento medido entra con 0, que es lo correcto por
        # defecto ahora que todas las fuentes comparten la misma ventana de vuelta.
        propia, off = vueltas[v], OFFSET_S.get(v, 0.0) / LARGO
        otras = [vueltas[o] for o, _, _ in FUENTES if o != v]
        cap = cv2.VideoCapture(str(DATA / v))
        for t in np.arange(t0 + 0.5, t1 - 0.5, CAND_EVERY_S):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * 25))
            ok, f = cap.read()
            if not ok:
                continue
            for box in L.proponer(f, det_tpl):
                n_cand += 1
                if hud.fraccion_hud(propia["hud"], box) > L.MAX_HUD:
                    rechazo("grafico de transmision"); continue
                tpl = L.bg_crop(f, box)
                if min(tpl.shape[:2]) < 36:
                    rechazo("recorte chico"); continue
                tpl = cv2.resize(tpl, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                ka, da = det_tpl.detectAndCompute(
                    cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY), None)
                if da is None or len(ka) < L.MIN_KP_TPL:
                    rechazo("sin textura"); continue

                vv = L.votos(propia["fl"], propia["mapa"], len(propia["ts"]), da)
                if vv.max() < L.MIN_VOTOS:
                    rechazo("no se detecta"); continue
                ps = picos(vv, propia["ts"])
                if not ps or len(ps) > MAX_POS:
                    rechazo("se enciende en demasiados lugares"); continue
                if L.inliers_ransac(ka, da, propia["kps"][ps[0]],
                                    propia["ds"][ps[0]]) < L.MIN_INLIERS:
                    rechazo("no verifica geometria"); continue

                confirmado = False
                for o in otras:
                    vc = L.votos(o["fl"], o["mapa"], len(o["ts"]), da)
                    if vc.max() < L.MIN_VOTOS_CHK:
                        continue
                    j = int(np.argmax(vc))
                    if L.inliers_ransac(ka, da, o["kps"][j], o["ds"][j]) >= L.MIN_INLIERS_CHK:
                        confirmado = True
                        break
                if not confirmado:
                    rechazo("no se confirma en la otra moto"); continue

                fr = [((propia["ts"][p] - t0) / (t1 - t0) + off) % 1.0 for p in ps]
                kept.append(dict(video=v, t=float(t), box=box, fracs=fr))
        cap.release()
        print(f"  tras {v}: {len(kept)} acumulados")

    n_pos = np.array([len(k["fracs"]) for k in kept])
    print(f"\n{len(kept)} landmarks de {n_cand} candidatos")
    print(f"  con 1 posicion (no ambiguos): {(n_pos == 1).sum()}")
    print(f"  con 2 o 3 posiciones (ambiguos, utiles en combinacion): "
          f"{(n_pos > 1).sum()}")
    print("descartes:", motivos)

    # Matriz de posiciones con relleno NaN, para poder guardarla como array.
    P = np.full((len(kept), MAX_POS), np.nan)
    for i, k in enumerate(kept):
        P[i, :len(k["fracs"])] = k["fracs"]
    h, _ = np.histogram(P[:, 0], bins=11, range=(0, 1))
    print("reparto por decimos (posicion principal):", h.tolist())

    # Se guarda con que fuentes se construyo, para que main.py pueda avisar si la
    # base en disco no corresponde a la configuracion actual.
    np.savez_compressed(OUT / "landmarks_multi.npz",
                        t=np.array([k["t"] for k in kept]),
                        box=np.array([k["box"] for k in kept]),
                        fracs=P, frac=P[:, 0],
                        video=np.array([k["video"] for k in kept]),
                        fuentes=np.array([v for v, _, _ in FUENTES]))
    print(f"-> {OUT / 'landmarks_multi.npz'}")


if __name__ == "__main__":
    main()
