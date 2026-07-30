"""Base de landmarks densa: muchos mas candidatos, filtros calibrados, busqueda rapida.

La primera version daba 26 landmarks y solo 21% de cobertura. Dos causas:

  - Pocos candidatos (324 para una vuelta entera).
  - El filtro cruzado exigia 18 inliers desde la otra moto. Pero los tres videos son
    del MISMO dia y el MISMO circuito: el fondo es identico, solo cambian el punto de
    vista y la linea de carrera. Pedir tanto descartaba estructuras validas (100
    candidatos perdidos ahi).

El cuello de botella para escalar era el costo: comparar cada candidato contra cada
posicion de dos vueltas es cuadratico. Se reemplaza por un unico indice FLANN con los
descriptores de toda la vuelta; cada candidato lo consulta UNA vez y cada match vota
por el frame del que salio. El perfil de votos hace de perfil de deteccion, y la
verificacion geometrica con RANSAC queda solo para el pico.
"""
import cv2
import numpy as np
from pathlib import Path

import hud

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "out"

REF_VIDEO, REF_T0, REF_T1 = "lemans2018_yamaha.mp4", 6.4, 115.6
CHK_VIDEO, CHK_T0, CHK_T1 = "lemans2018_ktm.mp4", 6.4, 111.5

CAND_EVERY_S = 0.5      # antes 1.0
BOXES_PER_FRAME = 6     # antes 3
VAL_EVERY_S = 0.6
BG_TOP, BG_BOT = 0.02, 0.58
GRID_X, GRID_Y = 10, 5
PAD = 0.02
NFEAT_FRAME = 1200
NFEAT_TPL = 800

RATIO = 0.80
MIN_KP_TPL = 25
MIN_VOTOS = 12          # votos en el pico para considerar que se detecta
MIN_INLIERS = 12        # inliers tras RANSAC en el pico, en la vuelta de referencia
# Mismo dia y mismo circuito, asi que el fondo es identico y basta con que aparezca;
# pero con 6 votos pasaban recortes del piloto por coincidencia, asi que ademas se
# exige verificacion geometrica en el video de control.
MIN_VOTOS_CHK = 10
MIN_INLIERS_CHK = 8
FAR_S = 6.0
AMBIG = 0.60            # un pico lejano por encima de esto vuelve ambiguo al candidato
MAX_HUD = 0.03          # fraccion maxima del recorte que puede caer sobre grafico fijo


def sift(n):
    return cv2.SIFT_create(nfeatures=n)


def bg_crop(frame, box):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    return frame[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]


def describir_frame(f, det, hud_mask):
    """Keypoints y descriptores del fondo, sin los que caen sobre graficos fijos."""
    banda = bg_crop(f, (0.0, BG_TOP, 1.0, BG_BOT))
    k, d = det.detectAndCompute(cv2.cvtColor(banda, cv2.COLOR_BGR2GRAY), None)
    if d is None:
        return [], np.zeros((0, 128), np.float32)
    return hud.filtrar_kp(k, d, hud_mask, int(f.shape[0] * BG_TOP))


def describir_vuelta(video, t0, t1, det):
    """Keypoints y descriptores de la franja de fondo a lo largo de una vuelta."""
    hud_mask = hud.mascara(video, t0, t1)
    cap = cv2.VideoCapture(str(DATA / video))
    ts, kps, descs = np.arange(t0, t1, VAL_EVERY_S), [], []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * 25))
        ok, f = cap.read()
        if not ok:
            kps.append([]); descs.append(np.zeros((0, 128), np.float32)); continue
        k, d = describir_frame(f, det, hud_mask)
        kps.append(k); descs.append(d)
    cap.release()
    return ts, kps, descs


def indice(descs):
    """Indice FLANN con TODAS las posiciones de la vuelta a la vez.

    Cada match devuelve imgIdx, o sea de que frame salio, asi que una sola consulta
    por candidato produce el perfil de votos completo. Antes esto eran cientos de
    comparaciones par a par por candidato.
    """
    fl = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=32))
    fl.add([d for d in descs if len(d)])
    fl.train()
    # Mapear el indice interno de FLANN de vuelta al indice de frame real.
    mapa = [i for i, d in enumerate(descs) if len(d)]
    return fl, np.array(mapa)


def votos(fl, mapa, n_frames, desc):
    """Perfil de votos por frame, con test de Lowe."""
    v = np.zeros(n_frames)
    if desc is None or len(desc) < 2:
        return v
    for par in fl.knnMatch(desc, k=2):
        if len(par) == 2 and par[0].distance < RATIO * par[1].distance:
            v[mapa[par[0].imgIdx]] += 1
    return v


def inliers_ransac(ka, da, kb, db):
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return 0
    bf = cv2.BFMatcher()
    good = [m for m, n in bf.knnMatch(da, db, k=2) if m.distance < RATIO * n.distance]
    if len(good) < 6:
        return 0
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 8.0)
    if H is None or mask is None:
        return 0
    det = np.linalg.det(H[:2, :2])
    if not np.isfinite(det) or det <= 1e-6 or det > 1e4:
        return 0
    return int(mask.sum())


def proponer(frame, det, k=BOXES_PER_FRAME):
    h, w = frame.shape[:2]
    band = frame[int(h * BG_TOP):int(h * BG_BOT)]
    kp = det.detect(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY), None)
    if len(kp) < 40:
        return []
    bh, bw = band.shape[:2]
    pts = np.array([p.pt for p in kp])
    hist, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=[GRID_X, GRID_Y],
                                  range=[[0, bw], [0, bh]])
    bloques = sorted(((hist[i:i + 2, j:j + 2].sum(), i, j)
                      for i in range(GRID_X - 1) for j in range(GRID_Y - 1)),
                     reverse=True)
    span, out, tomados = BG_BOT - BG_TOP, [], []
    for s, i, j in bloques:
        if s <= 0 or len(out) >= k:
            break
        if any(abs(i - a) < 2 and abs(j - b) < 2 for a, b in tomados):
            continue
        tomados.append((i, j))
        out.append((max(0.0, xe[i] / bw - PAD), BG_TOP + ye[j] / bh * span,
                    min(1.0, xe[i + 2] / bw + PAD), BG_TOP + ye[j + 2] / bh * span))
    return out


def main():
    det_frame, det_tpl = sift(NFEAT_FRAME), sift(NFEAT_TPL)

    ref_t, ref_kp, ref_d = describir_vuelta(REF_VIDEO, REF_T0, REF_T1, det_frame)
    chk_t, chk_kp, chk_d = describir_vuelta(CHK_VIDEO, CHK_T0, CHK_T1, det_frame)
    print(f"referencia {len(ref_t)} posiciones | control {len(chk_t)} posiciones")
    fl_ref, map_ref = indice(ref_d)
    fl_chk, map_chk = indice(chk_d)
    print("indices FLANN listos")

    hud_ref = hud.mascara(REF_VIDEO, REF_T0, REF_T1)
    cap = cv2.VideoCapture(str(DATA / REF_VIDEO))
    cand_t = np.arange(REF_T0 + 0.5, REF_T1 - 0.5, CAND_EVERY_S)
    kept, motivos, n_cand = [], {}, 0

    def rechazo(m):
        motivos[m] = motivos.get(m, 0) + 1

    for t in cand_t:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * 25))
        ok, f = cap.read()
        if not ok:
            continue
        for box in proponer(f, det_tpl):
            n_cand += 1
            # Un recorte sobre el cartel de la transmision matchea contra todos los
            # frames de cualquier video: no es circuito, es grafico.
            if hud.fraccion_hud(hud_ref, box) > MAX_HUD:
                rechazo("cae sobre grafico de transmision"); continue
            tpl = bg_crop(f, box)
            if min(tpl.shape[:2]) < 36:
                rechazo("recorte chico"); continue
            tpl = cv2.resize(tpl, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            ka, da = det_tpl.detectAndCompute(cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY), None)
            if da is None or len(ka) < MIN_KP_TPL:
                rechazo("sin textura"); continue

            v = votos(fl_ref, map_ref, len(ref_t), da)
            pico = v.max()
            if pico < MIN_VOTOS:
                rechazo("no se detecta"); continue
            i = int(np.argmax(v))
            lejos = np.abs(ref_t - ref_t[i]) > FAR_S
            v_lejos = v[lejos].max() if lejos.any() else 0
            if v_lejos > AMBIG * pico:
                rechazo("aparece en otro lugar"); continue

            inl = inliers_ransac(ka, da, ref_kp[i], ref_d[i])
            if inl < MIN_INLIERS:
                rechazo("no verifica geometria"); continue

            vc = votos(fl_chk, map_chk, len(chk_t), da)
            pico_c = vc.max()
            if pico_c < MIN_VOTOS_CHK:
                rechazo("no aparece en la otra moto"); continue
            j = int(np.argmax(vc))
            lejos_c = np.abs(chk_t - chk_t[j]) > FAR_S
            vc_lejos = vc[lejos_c].max() if lejos_c.any() else 0
            if vc_lejos > AMBIG * pico_c:
                rechazo("ambiguo en la otra moto"); continue
            # Verificacion geometrica tambien alla: con solo votos se colaban
            # recortes del mono del piloto por coincidencia de textura.
            if inliers_ransac(ka, da, chk_kp[j], chk_d[j]) < MIN_INLIERS_CHK:
                rechazo("no verifica en la otra moto"); continue

            kept.append(dict(t=float(t), box=box, inl=int(inl), pico=float(pico),
                             chk_t=float(chk_t[j]), chk_pico=float(pico_c)))
    cap.release()

    print(f"\n{len(kept)} landmarks de {n_cand} candidatos")
    print("descartes:", motivos)

    # Diagnostico independiente, no interviene en ningun filtro.
    esc = (CHK_T1 - CHK_T0) / (REF_T1 - REF_T0)
    des = np.array([abs(k["chk_t"] - (CHK_T0 + (k["t"] - REF_T0) * esc)) for k in kept])
    if len(des):
        print(f"diagnostico: |desvio| mediana {np.median(des):.1f}s | "
              f"{100*np.mean(des < 5):.0f}% dentro de 5s")
    ts = np.array([k["t"] for k in kept])
    if len(ts):
        h, _ = np.histogram(ts, bins=11, range=(REF_T0, REF_T1))
        print("reparto por decimos de vuelta:", h.tolist())

    np.savez_compressed(OUT / "landmarks.npz",
                        t=ts, box=np.array([k["box"] for k in kept]),
                        peak=np.array([k["inl"] for k in kept]),
                        far=np.zeros(len(kept)),
                        chk_t=np.array([k["chk_t"] for k in kept]))
    print(f"-> {OUT / 'landmarks.npz'}")


if __name__ == "__main__":
    main()
