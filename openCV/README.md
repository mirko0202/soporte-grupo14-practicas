# Localización visual en circuito desde onboard de MotoGP

Detecta en qué parte de la pista está la moto comparando el **fondo** del frame contra
una base de estructuras del circuito, y marca la posición con un punto sobre el plano.
Circuito Bugatti, Le Mans.

Cada frame se resuelve por su cuenta, solo por imagen: no mira el frame anterior ni
usa el tiempo del video de consulta. Por eso funciona igual con un video entero que
con un recorte de 30 segundos (verificado con `test_recorte.py`: 99% de posiciones
idénticas).

## Cómo funciona

**Base de landmarks.** Se recortan trozos del fondo de cinco onboards del mismo fin de
semana (Yamaha, KTM, Suzuki, Ducati, Honda). Cada recorte es una estructura concreta
del circuito, etiquetada con su posición en la vuelta. Los candidatos se proponen
solos, buscando las zonas del fondo con más densidad de puntos SIFT.

Usar cinco fuentes no es solo "más candidatos": cada moto lleva una línea de carrera
distinta, así que ve las mismas estructuras desde otro ángulo y capta las que desde
las otras quedaban tapadas por su propio piloto.

**Filtros automáticos**, que dejaron 445 landmarks de 2686 candidatos:

1. *No cae sobre gráficos de transmisión*. Los carteles superpuestos por la producción
   ("MotoGP | Le Mans", "Aprilia OnBoard") son idénticos en todos los videos, así que
   un recorte sobre ellos matchea contra cualquier frame. Se detectan solos midiendo
   la desviación temporal de cada píxel: un gráfico es estático, el paisaje a 300 km/h
   no. Descartó 513.
2. *Textura suficiente* para que SIFT pueda describirlo.
3. *Único en su propia vuelta*. Un cartel que se enciende en dos puntos del circuito
   no sirve para localizar por sí solo.
4. *Verificación geométrica RANSAC*.
5. *Visible, único y verificado desde otra moto*. Un landmark real es una estructura
   del circuito; uno que en realidad era el mono del piloto no aparece desde otra
   moto. Descartó 990. No hace falta definir dónde está el piloto: se cae solo.

**Localización por consenso.** Para cada frame se buscan todos los landmarks con un
índice FLANN, y los mejor votados pasan por RANSAC. Cada detección reparte su peso
entre sus posiciones posibles (un landmark ambiguo con tres posiciones aporta un
tercio a cada una), los pesos se acumulan sobre el eje de la vuelta y gana el punto
donde más coinciden. Así dos estructuras repetidas se desambiguan entre sí.

## Resultados

Medido sobre el onboard de **Aprilia**, que nunca se usó para construir la base:

| criterio (peso, nº landmarks) | cobertura | aciertos <5 s | error mediano |
|---|---|---|---|
| 0, 1 | 90.9% | 89% | 0.5 s |
| 20, 1 | 78.3% | 96% | 0.4 s |
| **40, 2** (el que usa el render) | **54.7%** | **100%** | **0.4 s** |
| 60, 2 | 44.9% | 100% | 0.4 s |

El criterio es una perilla: más cobertura o más precisión. `localizar4.py` imprime la
tabla completa en cada corrida.

**La construcción de la base no es determinista.** FLANN usa árboles con divisiones al
azar y RANSAC muestrea al azar, así que un candidato al borde de un umbral puede entrar
o no según la corrida. Reconstruyendo se obtuvo un 63% de landmarks en común y 445 en
lugar de 476, pero el rendimiento fue equivalente (mismo 100% de aciertos en el
criterio del render) y el reparto por sectores del circuito casi idéntico. Lo que
importa es que cada zona tenga estructuras reconocibles, no cuáles exactamente. Si
hiciera falta reproducibilidad estricta, habría que fijar `cv2.setRNGSeed()` y usar
matching exacto en lugar de FLANN aproximado, a costa de velocidad.

## Requisitos

```bash
pip install opencv-python numpy scipy yt-dlp
```

## Videos

Cinco onboards fuente + el de prueba (Aprilia), todos del GP de Francia 2018:

```bash
python -m yt_dlp -f 136 -o "data/lemans2018_yamaha.mp4" "https://www.youtube.com/watch?v=fz0n8Khp9RA"
```

Los demás son `SsNOV98d1xM` (KTM), `RsjwJq_QnF0` (Aprilia), `j1rzhLVF9uM` (Suzuki),
`_LWYeWY7qM4` (Ducati), `tBTcJQ4sZ1o` (Honda). Hace falta además
`data/Le-Mans-Bugatti-1513.svg`, el plano vectorial del circuito.

## Cómo correrlo

```bash
python main.py
```

Eso es todo. El video a renderizar y el criterio de aceptación se eligen editando el
bloque de configuración arriba de `main.py`:

```python
VIDEO = "aprilia"          # aprilia, yamaha, ktm, suzuki, ducati, honda
PESO_MIN, N_LM_MIN = 40, 2
```

`main.py` encadena los tres pasos y saltea lo que ya esté hecho: construir la base
(~1 h, una sola vez), localizar el video (~10 min por video) y renderizar (~30 s).
El resultado queda en `out/salida_<video>_p<peso>n<n>.mp4`.

Cada paso también se puede correr suelto:

```bash
python localizar4.py aprilia 34 67
```

Ese ejemplo localiza solo el recorte de 34 a 67 segundos, para comprobar que funciona
con un fragmento y no solo con la vuelta entera.

## Archivos

- `main.py` — punto de entrada, encadena todo
- `landmarks4.py` — construye la base de landmarks
- `landmarks2.py` — propuesta, filtrado, índice FLANN, verificación RANSAC
- `hud.py` — detecta gráficos de transmisión por varianza temporal
- `localizar4.py` — localización por consenso
- `render.py` — video de salida
- `trazado_svg.py` + `svg_track.py` — trazado del circuito desde el SVG
- `velocidad.py` — corrección de velocidad no constante

`out/landmarks_multi.npz` es la base ya construida; si está, `main.py` no la recalcula.

## Detalles que costaron encontrar

**La ventana de vuelta son 94 s, no 106.** Se usaba el tramo continuo sin cortes del
clip (6.4 s a ~112 s), pero ese tramo incluye la aproximación antes de cruzar la línea
y un resto después. La vuelta real va de 10.4 s a 104.4 s. Con la ventana inflada la
fase quedaba estirada y el punto caía corrido sobre el plano hasta 11 s. Era invisible
para las métricas, porque usaban la misma ventana equivocada en la referencia y en la
consulta: el sistema medía bien contra su propia convención errónea.

**Las etiquetas son fracción de tiempo, pero el punto se dibuja por distancia.** Eso
supone velocidad constante y no lo es: en el 25% más recto del trazado la moto pasa
solo el 14% del tiempo, así que el punto queda atrasado hasta 195 m en la recta de
arranque. `velocidad.py` lo corrige estimando el perfil de velocidad de la geometría
del trazado. La intensidad se gradúa con `FUERZA_VEL` en `render.py` (hoy 0.6) porque
los valores extremos del modelo (300 y 70 km/h) son un supuesto, no telemetría.

**El muestreo tiene que ir sobre índices de frame enteros.** Con `np.arange` sobre
segundos, la acumulación de punto flotante hacía que dos corridas del mismo video que
arrancan en tiempos distintos leyeran frames distintos para el mismo instante. En uso
normal era invisible; rompía cualquier comparación entre corridas.

**Métricas que engañan.** La continuidad no sirve para evaluar una salida suavizada
temporalmente: el modelo de transición la impone por construcción. El avance monótono
tampoco: cuando las predicciones se apelotonan en pocos valores por *hubness*, muchas
diferencias dan cero y cuentan como monótonas — dio 86.7% sobre un resultado que era
2/12 correcto. Y el error entre fases hay que medirlo **circular**: la vuelta cierra,
así que fase 0.99 y 0.01 son casi el mismo lugar.

## Enfoques descartados

Antes de llegar a los landmarks recortados se probaron y midieron otros caminos, todos
peores: descriptores globales de frame por tiles (andan bien dentro de un mismo video
—8/8 pares verificados— pero dan 2/12 comparando entre videos distintos, porque la
moto y el piloto ocupan medio cuadro); embeddings de ResNet50 con pooling GeM (peor
que el descriptor a mano, el pooling global tira la disposición espacial que es lo que
identifica un lugar); nivelado por horizonte (en Le Mans el horizonte casi nunca se ve);
y recuperación de fase de vuelta por Laplacian eigenmaps. Cruzar con video de 2024
tampoco funciona: el arco Dunlop fue repintado entre 2018 y 2024.
