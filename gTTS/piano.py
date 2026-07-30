"""
Piano con gTTS.

La idea: gTTS no sirve como fuente de notas si le pedimos que diga "do", "re",
"mi"... Cada palabra sale con su propia entonacion, asi que sumarle semitonos
relativos no afina nada.

Lo que hacemos aca es al reves:
  1. Pedimos UNA sola silaba a gTTS (siempre la misma, "la").
  2. Le MEDIMOS la frecuencia real (f0) con autocorrelacion.
  3. Recortamos unos pocos ciclos de la vocal y los repetimos en loop a la
     velocidad justa para que suenen a la frecuencia de la nota que queremos.

Al repetir el ciclo en loop el tono queda constante (el habla no lo es) y al
partir de una f0 medida la afinacion es absoluta, no relativa.

Uso:
    python piano.py                             -> teclado en tiempo real
    python piano.py "1155665-"                  -> renderiza a melodia.wav
    python piano.py "1155665-" estrellita.wav   -> renderiza al archivo que digas
"""

import os
import sys
import threading

import numpy as np
import soundfile as sf
from gtts import gTTS

FRECUENCIA_MUESTREO = 24000   # gTTS entrega 24 kHz
SILABA_BASE = "la"            # vocal abierta y sostenida: la que mejor loopea
CARPETA_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "cache_voz")

# --- teoria musical ---------------------------------------------------------

SEMITONOS_DESDE_DO = {"do": 0, "do#": 1, "re": 2, "re#": 3, "mi": 4, "fa": 5,
                      "fa#": 6, "sol": 7, "sol#": 8, "la": 9, "la#": 10,
                      "si": 11}


def nota_a_midi(nombre_nota, octava):
    """'la', 4 -> 69"""
    return 12 * (octava + 1) + SEMITONOS_DESDE_DO[nombre_nota]


def midi_a_frecuencia(numero_midi):
    """Numero MIDI -> Hz. Referencia: la4 = midi 69 = 440 Hz."""
    return 440.0 * 2.0 ** ((numero_midi - 69) / 12.0)


# Distribucion tipo piano sobre el teclado de la compu:
#
#      w e     t y u
#     a s d f g h j k
#
NOTA_POR_TECLA = {
    "a": ("do",  4), "w": ("do#", 4), "s": ("re",  4), "e": ("re#", 4),
    "d": ("mi",  4), "f": ("fa",  4), "t": ("fa#", 4), "g": ("sol", 4),
    "y": ("sol#", 4), "h": ("la",  4), "u": ("la#", 4), "j": ("si",  4),
    "k": ("do",  5), "o": ("do#", 5), "l": ("re",  5), "p": ("re#", 5),
    "ñ": ("mi",  5),
}

# Para el modo melodia (compatible con la version anterior: 1..8)
NOTA_POR_NUMERO = {"1": ("do", 4), "2": ("re", 4), "3": ("mi", 4),
                   "4": ("fa", 4), "5": ("sol", 4), "6": ("la", 4),
                   "7": ("si", 4), "8": ("do", 5)}


# --- analisis de la voz -----------------------------------------------------

def obtener_muestra_de_voz():
    """Descarga (o lee del cache) la silaba base y la devuelve mono normalizada."""
    os.makedirs(CARPETA_CACHE, exist_ok=True)
    ruta_mp3 = os.path.join(CARPETA_CACHE, f"{SILABA_BASE}.mp3")
    if not os.path.exists(ruta_mp3):
        print(f"Descargando la silaba '{SILABA_BASE}' de gTTS...")
        gTTS(text=SILABA_BASE, lang="es", slow=False).save(ruta_mp3)

    onda_voz, frecuencia_muestreo = sf.read(ruta_mp3)
    if onda_voz.ndim > 1:                       # por las dudas
        onda_voz = onda_voz.mean(axis=1)
    onda_voz = onda_voz.astype(np.float64)
    onda_voz /= max(np.abs(onda_voz).max(), 1e-9)
    return onda_voz, frecuencia_muestreo


def detectar_frecuencia_fundamental(muestras, frecuencia_muestreo,
                                    hz_minimo=70.0, hz_maximo=400.0):
    """Frecuencia fundamental por autocorrelacion, con interpolacion parabolica.

    La autocorrelacion compara la señal consigo misma corrida N muestras. El
    corrimiento donde mas se parece es justo un periodo, asi que la frecuencia
    sale de dividir la frecuencia de muestreo por ese corrimiento."""
    muestras = (muestras - muestras.mean()) * np.hanning(len(muestras))
    autocorrelacion = np.correlate(muestras, muestras, "full")[len(muestras) - 1:]

    corrimiento_minimo = int(frecuencia_muestreo / hz_maximo)
    corrimiento_maximo = min(int(frecuencia_muestreo / hz_minimo),
                             len(autocorrelacion) - 2)
    corrimiento = corrimiento_minimo + int(
        np.argmax(autocorrelacion[corrimiento_minimo:corrimiento_maximo]))

    # el pico real casi nunca cae justo en una muestra: lo afinamos con una
    # parabola que pasa por los tres puntos vecinos
    pico_izquierdo = autocorrelacion[corrimiento - 1]
    pico_central = autocorrelacion[corrimiento]
    pico_derecho = autocorrelacion[corrimiento + 1]
    curvatura = pico_izquierdo - 2 * pico_central + pico_derecho
    if curvatura != 0:
        corrimiento += 0.5 * (pico_izquierdo - pico_derecho) / curvatura

    return frecuencia_muestreo / corrimiento


def extraer_ciclo_de_vocal(onda_voz, frecuencia_muestreo, periodos_por_ciclo=4):
    """Recorta unos pocos periodos de la parte mas estable de la vocal.

    Devuelve (ciclo, frecuencia_del_ciclo). El ciclo arranca en un cruce por
    cero ascendente para que al repetirlo en loop no haya un salto (un
    'click')."""
    muestras_por_ventana = int(0.04 * frecuencia_muestreo)          # 40 ms
    energia_por_ventana = np.convolve(onda_voz ** 2,
                                      np.ones(muestras_por_ventana), "valid")
    inicio_vocal = int(np.argmax(energia_por_ventana))              # nucleo

    frecuencia_estimada = detectar_frecuencia_fundamental(
        onda_voz[inicio_vocal:inicio_vocal + muestras_por_ventana],
        frecuencia_muestreo)
    muestras_por_periodo = frecuencia_muestreo / frecuencia_estimada
    largo_del_ciclo = int(round(periodos_por_ciclo * muestras_por_periodo))

    corte = inicio_vocal
    corte_maximo = min(inicio_vocal + int(muestras_por_periodo),
                       len(onda_voz) - largo_del_ciclo - 2)
    while corte < corte_maximo and not (onda_voz[corte] <= 0 < onda_voz[corte + 1]):
        corte += 1

    ciclo = onda_voz[corte:corte + largo_del_ciclo]

    # el ciclo quedo redondeado a muestras enteras, asi que su frecuencia real
    # no es exactamente la estimada: la recalculamos o quedariamos desafinados
    # unos cents en todas las notas por igual
    frecuencia_del_ciclo = periodos_por_ciclo * frecuencia_muestreo / len(ciclo)
    return ciclo, frecuencia_del_ciclo


# --- sintesis ---------------------------------------------------------------

def envolvente_adsr(total_muestras, frecuencia_muestreo, ataque_seg=0.02,
                    caida_seg=0.10, nivel_sostenido=0.75, liberacion_seg=0.25):
    """Curva de volumen. Sin esto cada nota empieza y termina con un chasquido."""
    envolvente = np.full(total_muestras, nivel_sostenido)
    muestras_ataque = min(int(ataque_seg * frecuencia_muestreo), total_muestras)
    muestras_caida = min(int(caida_seg * frecuencia_muestreo),
                         total_muestras - muestras_ataque)
    muestras_liberacion = min(int(liberacion_seg * frecuencia_muestreo),
                              total_muestras - muestras_ataque - muestras_caida)

    envolvente[:muestras_ataque] = np.linspace(0.0, 1.0, muestras_ataque)
    envolvente[muestras_ataque:muestras_ataque + muestras_caida] = np.linspace(
        1.0, nivel_sostenido, muestras_caida)
    if muestras_liberacion > 0:
        envolvente[total_muestras - muestras_liberacion:] *= np.linspace(
            1.0, 0.0, muestras_liberacion)
    return envolvente


def sintetizar_nota(ciclo, frecuencia_del_ciclo, frecuencia_objetivo,
                    duracion_seg=0.6, frecuencia_muestreo=FRECUENCIA_MUESTREO,
                    vibrato_hz=4.5, vibrato_profundidad=0.004):
    """Repite el ciclo de voz en loop a la velocidad que da `frecuencia_objetivo`.

    El ciclo contiene N periodos a `frecuencia_del_ciclo`. Si lo leemos con paso
    objetivo/ciclo, cada periodo dura lo que tiene que durar y la fundamental
    pasa a ser la objetivo exacta. Como es un loop, el tono no se mueve: suena a
    nota, no a palabra."""
    total_muestras = int(duracion_seg * frecuencia_muestreo)
    paso_de_lectura = frecuencia_objetivo / frecuencia_del_ciclo

    # un vibrato minimo evita que el loop suene a maquina
    tiempos = np.arange(total_muestras) / frecuencia_muestreo
    pasos_con_vibrato = paso_de_lectura * (
        1.0 + vibrato_profundidad * np.sin(2 * np.pi * vibrato_hz * tiempos))
    posiciones = np.cumsum(pasos_con_vibrato) % len(ciclo)

    # interpolacion lineal circular (repetimos la primer muestra al final para
    # que la ultima posicion empalme con el arranque del ciclo)
    tabla_circular = np.append(ciclo, ciclo[0])
    onda = np.interp(posiciones, np.arange(len(tabla_circular)), tabla_circular)

    return onda * envolvente_adsr(total_muestras, frecuencia_muestreo)


def sintetizar_silaba_completa(onda_voz, frecuencia_de_la_voz,
                               frecuencia_objetivo):
    """Modo alternativo: resamplea la silaba entera hasta que su f0 sea la
    objetivo.

    Se sigue entendiendo la palabra, pero cambia la duracion (efecto cinta) y
    arrastra la curva de entonacion del habla. Sirve para comparar contra
    sintetizar_nota() y mostrar por que el loop suena mejor."""
    factor_de_resampleo = frecuencia_objetivo / frecuencia_de_la_voz
    total_muestras = int(len(onda_voz) / factor_de_resampleo)
    posiciones = np.arange(total_muestras) * factor_de_resampleo
    return np.interp(posiciones, np.arange(len(onda_voz)), onda_voz)


# --- reproduccion en tiempo real -------------------------------------------

class NotaSonando:
    """Una nota que se esta reproduciendo, con el punto por donde va."""

    def __init__(self, onda):
        self.onda = onda
        self.posicion = 0

    @property
    def termino(self):
        return self.posicion >= len(self.onda)


class Mezclador:
    """Suma las notas que estan sonando, asi se pueden tocar acordes."""

    def __init__(self, volumen=0.6):
        self.notas_sonando = []
        self.volumen = volumen
        self.candado = threading.Lock()

    def tocar(self, onda_de_la_nota):
        with self.candado:
            self.notas_sonando.append(NotaSonando(onda_de_la_nota))

    def rellenar_salida(self, salida, cantidad_muestras, tiempo, estado):
        """Callback que sounddevice llama cada vez que necesita mas audio."""
        suma = np.zeros(cantidad_muestras)
        with self.candado:
            for nota in list(self.notas_sonando):
                trozo = nota.onda[nota.posicion:nota.posicion + cantidad_muestras]
                suma[:len(trozo)] += trozo
                nota.posicion += cantidad_muestras
                if nota.termino:
                    self.notas_sonando.remove(nota)
        salida[:, 0] = np.clip(suma * self.volumen, -1.0, 1.0)


def modo_teclado(ciclo, frecuencia_del_ciclo):
    try:
        import msvcrt          # solo existe en Windows
    except ImportError:
        print("El teclado en tiempo real anda solo en Windows (usa msvcrt).")
        print("En Mac o Linux podes generar el .wav igual:")
        print('    python piano.py "1-1-5-5-6-6-5---"')
        return
    import sounddevice as sd

    print("Pre-sintetizando las notas...")
    onda_por_tecla = {}
    for tecla, (nombre_nota, octava) in NOTA_POR_TECLA.items():
        frecuencia_objetivo = midi_a_frecuencia(nota_a_midi(nombre_nota, octava))
        onda_por_tecla[tecla] = sintetizar_nota(
            ciclo, frecuencia_del_ciclo, frecuencia_objetivo).astype(np.float32)

    print()
    print("=" * 54)
    print("            PIANO gTTS - tiempo real")
    print("=" * 54)
    print("    negras:   w e     t y u        o p")
    print("    blancas: a s d f g h j k      l   ñ")
    print("             do re mi fa sol la si do")
    print()
    print(f"    voz base medida en {frecuencia_del_ciclo:.1f} Hz"
          f"  ->  afinada a la4 = 440 Hz")
    print("    ESC para salir")
    print("=" * 54)

    mezclador = Mezclador()
    with sd.OutputStream(samplerate=FRECUENCIA_MUESTREO, channels=1,
                         dtype="float32", blocksize=512,
                         callback=mezclador.rellenar_salida):
        while True:
            caracter = msvcrt.getwch()
            if caracter == "\x1b":                        # ESC
                break
            tecla = caracter.lower()
            if tecla in onda_por_tecla:
                nombre_nota, octava = NOTA_POR_TECLA[tecla]
                frecuencia_objetivo = midi_a_frecuencia(
                    nota_a_midi(nombre_nota, octava))
                print(f"  {nombre_nota}{octava} ({frecuencia_objetivo:.1f} Hz)")
                mezclador.tocar(onda_por_tecla[tecla])

    print("\nChau.")


# --- modo melodia -----------------------------------------------------------

DURACION_UNIDAD = 0.30      # cuanto dura una nota "corta", en segundos
SILENCIO_ENTRE_NOTAS = 0.03


class EventoMusical:
    """Una nota (o un silencio, si frecuencia es None) y cuanto dura."""

    def __init__(self, frecuencia, unidades=1):
        self.frecuencia = frecuencia
        self.unidades = unidades

    @property
    def es_silencio(self):
        return self.frecuencia is None


def leer_secuencia(secuencia):
    """Convierte el texto de la melodia en una lista de EventoMusical.

    Acepta los numeros 1..8 (do4 a do5) o las mismas letras del teclado, que
    ademas llegan a las negras. Simbolos:
        -   alarga la nota anterior una unidad
        .   silencio
        ' ' se ignora (solo para que se lea comodo)"""
    eventos = []
    for caracter in secuencia:
        if caracter == " ":
            continue
        if caracter == "-":
            if eventos:
                eventos[-1].unidades += 1
            continue
        if caracter == ".":
            eventos.append(EventoMusical(None))
            continue

        nombre_y_octava = (NOTA_POR_NUMERO.get(caracter)
                           or NOTA_POR_TECLA.get(caracter.lower()))
        if nombre_y_octava is None:
            print(f"Aviso: ignoro el caracter '{caracter}'.")
            continue
        eventos.append(
            EventoMusical(midi_a_frecuencia(nota_a_midi(*nombre_y_octava))))
    return eventos


def modo_melodia(ciclo, frecuencia_del_ciclo, secuencia,
                 archivo_salida="melodia.wav"):
    """Renderiza una secuencia de notas a un .wav."""
    eventos = leer_secuencia(secuencia)
    if not eventos:
        print("No ingresaste ninguna nota valida.")
        return

    fragmentos_de_audio = []
    for evento in eventos:
        duracion_seg = evento.unidades * DURACION_UNIDAD
        if evento.es_silencio:
            fragmentos_de_audio.append(
                np.zeros(int(FRECUENCIA_MUESTREO * duracion_seg)))
        else:
            fragmentos_de_audio.append(sintetizar_nota(
                ciclo, frecuencia_del_ciclo, evento.frecuencia,
                duracion_seg=duracion_seg))
            fragmentos_de_audio.append(  # respiro para que no se peguen
                np.zeros(int(FRECUENCIA_MUESTREO * SILENCIO_ENTRE_NOTAS)))

    audio_final = np.concatenate(fragmentos_de_audio)
    sf.write(archivo_salida, audio_final, FRECUENCIA_MUESTREO)
    print(f"Listo: {archivo_salida}  "
          f"({len(audio_final) / FRECUENCIA_MUESTREO:.1f} s, "
          f"{len(eventos)} notas)")


def main():
    onda_voz, frecuencia_muestreo = obtener_muestra_de_voz()
    if frecuencia_muestreo != FRECUENCIA_MUESTREO:
        print(f"Aviso: gTTS devolvio {frecuencia_muestreo} Hz "
              f"(esperabamos {FRECUENCIA_MUESTREO}).")
    ciclo, frecuencia_del_ciclo = extraer_ciclo_de_vocal(onda_voz,
                                                         frecuencia_muestreo)
    print(f"Silaba '{SILABA_BASE}': frecuencia medida = "
          f"{frecuencia_del_ciclo:.1f} Hz, ciclo de {len(ciclo)} muestras.")

    if len(sys.argv) > 1:
        archivo_salida = sys.argv[2] if len(sys.argv) > 2 else "melodia.wav"
        modo_melodia(ciclo, frecuencia_del_ciclo, sys.argv[1], archivo_salida)
    else:
        modo_teclado(ciclo, frecuencia_del_ciclo)


if __name__ == "__main__":
    main()
