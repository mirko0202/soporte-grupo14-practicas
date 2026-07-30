"""Leer el trazado del circuito directamente de los datos vectoriales del SVG.

El SVG guarda la pista como curvas de Bezier con coordenadas exactas, asi que no hay
que extraer nada de pixeles: se evalua la curva y listo. Eso reemplaza por completo
los cinco intentos fallidos de sacar la linea central del PNG (contorno, esqueleto,
geodesica) y tambien los waypoints puestos a mano.

Ademas el archivo trae un grupo "Startline" con la posicion de la linea de meta, que
antes no habia forma de saber mirando el video.
"""
import re
import numpy as np
from pathlib import Path

SVG = Path(__file__).parent / "data" / "Le-Mans-Bugatti-1513.svg"
GRUPO_PISTA = "_x32_002_Bugatti_Circuit"
GRUPO_META = "Startline"


def _grupos(texto):
    """Contenido de cada <g id=...>, asumiendo que no estan anidados."""
    out = {}
    for m in re.finditer(r'<g\s+id="([^"]+)"[^>]*>(.*?)</g>', texto, re.S):
        out[m.group(1)] = m.group(2)
    return out


def _tokens(d):
    """Separar un atributo `d` en comandos y numeros."""
    for m in re.finditer(r'([MmLlHhVvCcSsQqTtAaZz])|(-?\d*\.?\d+(?:[eE][-+]?\d+)?)', d):
        yield (m.group(1), None) if m.group(1) else (None, float(m.group(2)))


def _bezier3(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def parse_d(d, n_bezier=24):
    """Convertir un atributo `d` en una polilinea. Soporta M/L/H/V/C/S/Z."""
    pts, cur, start, prev_c2, cmd = [], np.zeros(2), np.zeros(2), None, None
    nums, it = [], _tokens(d)

    def flush():
        """Ejecutar el comando pendiente con los numeros acumulados."""
        nonlocal cur, start, prev_c2, nums
        if cmd is None or not nums:
            return
        rel = cmd.islower()
        c = cmd.upper()
        i = 0
        while i < len(nums):
            if c == "M":
                p = np.array(nums[i:i + 2]); i += 2
                cur = cur + p if rel else p
                if not pts:
                    start = cur.copy()
                pts.append(cur.copy())
                prev_c2 = None
            elif c == "L":
                p = np.array(nums[i:i + 2]); i += 2
                cur = cur + p if rel else p
                pts.append(cur.copy()); prev_c2 = None
            elif c == "H":
                x = nums[i]; i += 1
                cur = np.array([cur[0] + x if rel else x, cur[1]])
                pts.append(cur.copy()); prev_c2 = None
            elif c == "V":
                y = nums[i]; i += 1
                cur = np.array([cur[0], cur[1] + y if rel else y])
                pts.append(cur.copy()); prev_c2 = None
            elif c in ("C", "S"):
                if c == "C":
                    a = np.array(nums[i:i + 2]); b = np.array(nums[i + 2:i + 4])
                    e = np.array(nums[i + 4:i + 6]); i += 6
                    c1 = cur + a if rel else a
                    c2 = cur + b if rel else b
                else:
                    b = np.array(nums[i:i + 2]); e = np.array(nums[i + 2:i + 4]); i += 4
                    # El control implicito de S es el reflejo del anterior.
                    c1 = 2 * cur - prev_c2 if prev_c2 is not None else cur
                    c2 = cur + b if rel else b
                end = cur + e if rel else e
                seg = _bezier3(cur, c1, c2, end, n_bezier)
                pts.extend(seg[1:])
                cur, prev_c2 = end, c2
            else:
                i = len(nums)
        nums = []

    for c_tok, num in it:
        if c_tok is not None:
            flush()
            if c_tok in "Zz":
                if pts:
                    pts.append(start.copy())
                cmd = None
            else:
                cmd = c_tok
        else:
            nums.append(num)
            # M/L/H/V con varios pares seguidos se resuelven en flush().
    flush()
    return np.array(pts) if pts else np.zeros((0, 2))


def cargar():
    """Devuelve (paths_de_pista, punto_de_meta). Coordenadas del SVG."""
    txt = SVG.read_text(encoding="utf-8", errors="ignore")
    g = _grupos(txt)
    if GRUPO_PISTA not in g:
        raise RuntimeError(f"no esta el grupo {GRUPO_PISTA}: {list(g)}")

    pista = [parse_d(m.group(1))
             for m in re.finditer(r'<path[^>]*\sd="([^"]+)"', g[GRUPO_PISTA])]
    pista = [p for p in pista if len(p) > 2]

    meta = None
    if GRUPO_META in g:
        seg = []
        for m in re.finditer(r'<(?:path|line)[^>]*?(?:\sd="([^"]+)"|'
                             r'x1="([-\d.]+)"\s*y1="([-\d.]+)"\s*'
                             r'x2="([-\d.]+)"\s*y2="([-\d.]+)")', g[GRUPO_META]):
            if m.group(1):
                q = parse_d(m.group(1))
                if len(q):
                    seg.append(q.mean(0))
            elif m.group(2):
                seg.append(np.array([(float(m.group(2)) + float(m.group(4))) / 2,
                                     (float(m.group(3)) + float(m.group(5))) / 2]))
        if seg:
            meta = np.mean(seg, axis=0)
    return pista, meta


if __name__ == "__main__":
    pista, meta = cargar()
    print(f"{len(pista)} paths en el grupo de pista")
    for i, p in enumerate(pista):
        largo = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        cerrado = np.linalg.norm(p[0] - p[-1]) < 2
        print(f"  path {i}: {len(p):5d} puntos, largo {largo:8.1f}, "
              f"{'cerrado' if cerrado else 'abierto'}, "
              f"x[{p[:,0].min():.0f},{p[:,0].max():.0f}] "
              f"y[{p[:,1].min():.0f},{p[:,1].max():.0f}]")
    print(f"\nlinea de meta: {meta}")
