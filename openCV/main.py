"""Punto de entrada unico: elegis el video aca abajo y corres este archivo.

    python main.py

Hace todo lo necesario en orden y saltea lo que ya este hecho:

    1. Construye la base de landmarks si no existe   (~1 h, una sola vez)
    2. Localiza el video elegido si no esta hecho    (~10 min por video)
    3. Genera el video de salida con el punto        (~30 s)

El resultado queda en out/salida_<video>_p<peso>n<n>.mp4
"""
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------

# Que video renderizar. Opciones: aprilia, yamaha, ktm, suzuki, ducati, honda.
# "aprilia" es el unico que NO se uso para construir la base, asi que es el que
# muestra el rendimiento real del sistema; los otros cinco son fuentes.
VIDEO = "aprilia"

# Videos que aportan landmarks a la base. Cada moto lleva una linea de carrera
# distinta, asi que ve las mismas estructuras desde otro angulo: mas fuentes = mas
# cobertura. Con 1 fuente el sistema acierta el 67%, con 2 el 73%, con 5 el 100%.
# Hacen falta al menos 2, porque un candidato tiene que confirmarse desde otra moto
# (ese es el filtro que descarta los recortes del piloto y del carenado).
#
# IMPORTANTE: el video que quieras MEDIR conviene dejarlo afuera. Si lo incluis, el
# sistema ya conoce sus landmarks y el porcentaje de aciertos deja de ser honesto.
# Por eso 'aprilia' no esta en esta lista.
#
# Si cambias esta lista hay que reconstruir la base (~1 h); main.py avisa solo.
FUENTES = ["yamaha", "ktm", "suzuki", "ducati", "honda"]

# Criterio de aceptacion de una posicion: peso minimo del consenso y cuantos
# landmarks distintos tienen que coincidir. Mas alto = el punto aparece menos
# seguido pero se equivoca menos. Valores medidos sobre el aprilia:
#     ( 0, 1) -> 92% del tiempo con posicion, 87% de aciertos
#     (20, 1) -> 82%, 92%
#     (40, 2) -> 57%, 100%     <- el que viene puesto
#     (60, 2) -> 51%, 100%
PESO_MIN, N_LM_MIN = 40, 2

# Poner en True para rehacer la localizacion aunque ya este calculada
# (por ejemplo si cambiaste algo del metodo).
FORZAR_LOCALIZACION = False

# ---------------------------------------------------------------------------

RAIZ = Path(__file__).parent
DATA = RAIZ / "data"
OUT = RAIZ / "out"
BASE = OUT / "landmarks_multi.npz"


def paso(n, txt):
    print(f"\n{'=' * 66}\n  PASO {n}: {txt}\n{'=' * 66}")


def correr(modulo, argv):
    """Ejecutar el main() de un modulo pasandole argumentos de linea de comando."""
    viejo = sys.argv
    try:
        sys.argv = [modulo.__name__] + [str(a) for a in argv]
        modulo.main()
    finally:
        sys.argv = viejo


def base_coincide():
    """True si la base guardada se construyo con las fuentes configuradas hoy."""
    import numpy as np
    z = np.load(BASE)
    if "fuentes" not in z:
        return True          # base vieja sin el dato: no hay forma de saber
    guardadas = sorted(str(v) for v in z["fuentes"])
    pedidas = sorted(f"lemans2018_{n}.mp4" for n in FUENTES)
    if guardadas == pedidas:
        return True
    print("\nLa base guardada NO corresponde a las fuentes configuradas:")
    print("   guardada:  " + ", ".join(guardadas))
    print("   pedida:    " + ", ".join(pedidas))
    print("   -> se reconstruye")
    return False


def main():
    import localizar4

    if VIDEO not in localizar4.VIDEOS:
        print(f"ERROR: '{VIDEO}' no es un video valido.")
        print("Opciones:", ", ".join(sorted(localizar4.VIDEOS)))
        return 1

    archivo = DATA / localizar4.VIDEOS[VIDEO][0]
    if not archivo.exists():
        print(f"ERROR: falta {archivo}")
        print("Ver el README para el comando de descarga.")
        return 1

    OUT.mkdir(exist_ok=True)
    t_ini = time.time()
    print(f"video: {VIDEO}   criterio: peso>={PESO_MIN}, landmarks>={N_LM_MIN}")

    # 1. Base de landmarks -----------------------------------------------------
    import landmarks4
    landmarks4.usar_fuentes(FUENTES)

    hay_base = BASE.exists() and base_coincide()
    if hay_base:
        print(f"\nbase de landmarks: ya existe ({BASE.name}), se reutiliza")
    else:
        paso(1, f"construir la base con {len(FUENTES)} fuentes "
                f"({', '.join(FUENTES)}) — ~1 h")
        landmarks4.main()
        if not BASE.exists():
            print("ERROR: la base no se genero")
            return 1

    # 2. Localizacion ----------------------------------------------------------
    pos = OUT / f"pos4_{VIDEO}.npz"
    if pos.exists() and not FORZAR_LOCALIZACION:
        print(f"localizacion: ya existe ({pos.name}), se reutiliza")
        print("   (poner FORZAR_LOCALIZACION = True para recalcularla)")
    else:
        paso(2, f"localizar '{VIDEO}' contra la base (~10 min)")
        correr(localizar4, [VIDEO])
        if not pos.exists():
            print("ERROR: la localizacion no genero salida")
            return 1

    # 3. Render ----------------------------------------------------------------
    paso(3, "generar el video con el punto sobre el plano")
    import render
    correr(render, [VIDEO, PESO_MIN, N_LM_MIN])

    salida = OUT / f"salida_{VIDEO}_p{PESO_MIN}n{N_LM_MIN}.mp4"
    print(f"\n{'=' * 66}")
    print(f"  LISTO en {(time.time() - t_ini) / 60:.1f} min")
    print(f"  {salida}")
    print(f"{'=' * 66}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
