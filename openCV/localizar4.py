"""Localizacion por consenso: los landmarks ambiguos se desambiguan entre si.

Un cartel repetido no dice donde estas: aparece en tres puntos del circuito. Pero si
en el mismo frame se ven DOS estructuras repetidas, la combinacion suele darse en un
solo lugar de la vuelta, porque cada una tiene su propio juego de posiciones y solo
en un punto coinciden.

Como se implementa: cada landmark detectado reparte su peso entre sus posiciones
posibles (uno ambiguo con tres posiciones aporta un tercio a cada una, asi que pesa
menos que uno inequivoco). Los pesos se acumulan sobre el eje de la vuelta con un
poco de tolerancia, y gana el punto donde mas se acumula.

La decision sigue siendo por imagen: cada frame se resuelve solo, sin mirar el
anterior ni usar el tiempo del video.

Uso: python localizar4.py [aprilia] [t_desde] [t_hasta]

Los tiempos son opcionales y sirven para procesar un RECORTE del video en vez de la
vuelta entera. El metodo no cambia: como cada frame se resuelve solo por imagen, un
recorte se localiza igual que la vuelta completa. Lo unico que deja de tener sentido
en un recorte es el diagnostico de error, que compara contra la fase que
corresponderia si el clip fuera una vuelta entera desde la meta.
"""
import sys
import cv2
import numpy as np
from pathlib import Path

import hud
import landmarks2 as L

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "out"

QUERY_EVERY_S = 0.2
TOP_K = 15            # candidatos votados que pasan por verificacion geometrica
MIN_INLIERS = 12      # para que una deteccion entre al consenso
N_BINS = 360          # resolucion del acumulador de fase (~0.3 s por bin)
TOL_BINS = 4          # tolerancia al acumular: dos landmarks del mismo lugar no caen
                      # exactamente en la misma fase

# Ventana de la vuelta real, no del tramo continuo del clip: el tramo continuo
# incluye la aproximacion antes de cruzar la linea y un resto despues. La vuelta va
# de 10.4 s a 104.4 s, o sea 94 s, que es el tiempo de vuelta de Le Mans.
VUELTA = (10.4, 104.4)
LARGO_VUELTA = VUELTA[1] - VUELTA[0]

VIDEOS = {k: (f"lemans2018_{k}.mp4", *VUELTA)
          for k in ("yamaha", "ktm", "suzuki", "ducati", "honda", "aprilia")}


def cargar_base(det):
    z = np.load(OUT / "landmarks_multi.npz")
    ts, boxes, fracs = z["t"], z["box"], z["fracs"]
    vids = [str(v) for v in z["video"]]
    caps, base = {}, []
    for t, box, fr, v in zip(ts, boxes, fracs, vids):
        if v not in caps:
            caps[v] = cv2.VideoCapture(str(DATA / v))
        caps[v].set(cv2.CAP_PROP_POS_FRAMES, int(t * 25))
        ok, f = caps[v].read()
        if not ok:
            continue
        tpl = cv2.resize(L.bg_crop(f, box), None, fx=2.0, fy=2.0,
                         interpolation=cv2.INTER_CUBIC)
        k, d = det.detectAndCompute(cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY), None)
        if d is None or len(d) < 2:
            continue
        pos = [p for p in fr if not np.isnan(p)]
        base.append(dict(kp=k, desc=d, pos=pos, video=v))
    for c in caps.values():
        c.release()
    return base


def consenso(detecciones, base):
    """Acumular las posiciones posibles y devolver (fase, peso, n_landmarks)."""
    acc = np.zeros(N_BINS)
    quien = [[] for _ in range(N_BINS)]
    for idx, inl in detecciones:
        pos = base[idx]["pos"]
        if not pos:
            continue
        # Repartido: un landmark ambiguo aporta menos por posicion que uno inequivoco.
        peso = inl / len(pos)
        for p in pos:
            b = int(p * N_BINS) % N_BINS
            for d in range(-TOL_BINS, TOL_BINS + 1):
                bb = (b + d) % N_BINS
                acc[bb] += peso * (1.0 - abs(d) / (TOL_BINS + 1))
                if idx not in quien[bb]:
                    quien[bb].append(idx)
    if acc.max() <= 0:
        return None, 0.0, 0
    b = int(np.argmax(acc))
    return (b + 0.5) / N_BINS, float(acc[b]), len(quien[b])


def main():
    q = sys.argv[1] if len(sys.argv) > 1 else "aprilia"
    fn, t0, t1 = VIDEOS[q]
    recorte = len(sys.argv) > 3
    if recorte:
        t0, t1 = float(sys.argv[2]), float(sys.argv[3])
        print(f"RECORTE {t0:.1f}s - {t1:.1f}s del video")
    det = L.sift(L.NFEAT_FRAME)
    base = cargar_base(L.sift(L.NFEAT_TPL))
    n_amb = sum(1 for b in base if len(b["pos"]) > 1)
    print(f"base: {len(base)} landmarks ({n_amb} ambiguos) | consulta: {fn}")

    fl = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=32))
    fl.add([b["desc"] for b in base])
    fl.train()

    hud_q = hud.mascara(fn, t0, t1)
    cap = cv2.VideoCapture(str(DATA / fn))
    # Muestrear sobre indices de frame ENTEROS y derivar el tiempo de ahi. Con
    # np.arange sobre segundos, la acumulacion de punto flotante hacia que dos
    # corridas que arrancan en tiempos distintos leyeran frames distintos para el
    # mismo instante nominal, y las comparaciones entre corridas no eran limpias.
    paso = max(1, int(round(QUERY_EVERY_S * 25)))
    frames = np.arange(int(round(t0 * 25)), int(round(t1 * 25)), paso)
    tq = frames / 25.0
    fase, peso, n_lm = [], [], []
    for n, (fr, t) in enumerate(zip(frames, tq)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
        ok, f = cap.read()
        if not ok:
            fase.append(np.nan); peso.append(0); n_lm.append(0); continue
        kb, db = L.describir_frame(f, det, hud_q)

        v = np.zeros(len(base))
        if db is not None and len(db) >= 2:
            for par in fl.knnMatch(db, k=2):
                if len(par) == 2 and par[0].distance < 0.92 * par[1].distance:
                    v[par[0].imgIdx] += 1

        dets = []
        for i in np.argsort(v)[::-1][:TOP_K]:
            if v[i] == 0:
                break
            x = L.inliers_ransac(base[i]["kp"], base[i]["desc"], kb, db)
            if x >= MIN_INLIERS:
                dets.append((int(i), x))
        p, w, k = consenso(dets, base)
        fase.append(p if p is not None else np.nan)
        peso.append(w); n_lm.append(k)
        if n % 25 == 0:
            print(f"  {t:.0f}s / {t1:.0f}s", end="\r", flush=True)
    cap.release()

    fase, peso, n_lm = np.array(fase), np.array(peso), np.array(n_lm)
    nombre = f"pos4_{q}{'_clip' if recorte else ''}.npz"
    np.savez(OUT / nombre, tq=tq, frac_todos=fase, peso=peso, n_lm=n_lm)

    if recorte:
        # En un recorte no hay verdad de referencia: la fase esperada solo se conoce
        # si el clip es una vuelta entera desde la meta.
        m = (peso >= 40) & (n_lm >= 2) & ~np.isnan(fase)
        print(f"\ncobertura {100*m.mean():.1f}%  |  "
              f"fase de {np.nanmin(fase[m]):.3f} a {np.nanmax(fase[m]):.3f}")
        print(f"-> {OUT / nombre}")
        return

    esperado = (tq - t0) / (t1 - t0)
    d = np.abs(fase - esperado)
    err = np.minimum(d, 1.0 - d) * LARGO_VUELTA
    print(f"\n{'peso min':>9} {'n_lm min':>9} {'cobertura':>10} {'<5s':>7} "
          f"{'err mediana':>12}")
    for wmin in (0, 20, 40, 60):
        for kmin in (1, 2):
            m = (peso >= wmin) & (n_lm >= kmin) & ~np.isnan(fase)
            if m.sum() < 5:
                continue
            print(f"{wmin:9d} {kmin:9d} {100*m.mean():9.1f}% "
                  f"{100*np.mean(err[m] < 5):6.0f}% {np.median(err[m]):11.1f}s")
    print(f"-> {OUT / nombre}")


if __name__ == "__main__":
    main()
