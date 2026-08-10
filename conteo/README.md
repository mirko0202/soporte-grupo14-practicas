# Conteo manual de puertas y personas en video

Esta aplicación abre un video, permite dibujar manualmente una o más puertas en el primer fotograma y luego cuenta cuántas personas cruzan cada puerta.

## Requisitos

- Python 3
- OpenCV
- NumPy
- Ultralytics YOLO

## Instalación

Abre una terminal en la carpeta del proyecto e instala las dependencias:

```bash
pip install -r requirements.txt
```

## Cómo ejecutar el programa

Ejecuta el archivo principal indicando el video de entrada:

```bash
python main.py --video videos/video2.mp4
```

Al iniciar, el programa muestra el primer fotograma del video para que dibujes las puertas con el mouse.

## Selección manual de puertas

1. Arrastra con el mouse para dibujar una puerta rectangular.
2. La puerta se guarda como `Door 1`, `Door 2`, `Door 3`, etc.
3. Repite el proceso para todas las puertas que quieras definir.
4. Presiona `ENTER` o `SPACE` para terminar la selección y comenzar el procesamiento.

Atajos durante la selección:

- `U`: deshacer la última puerta.
- `C`: borrar todas las puertas.
- `ESC`: cancelar la selección.

## Durante la reproducción

- `Q` o `ESC`: detener el programa.
- El video muestra las puertas dibujadas manualmente.
- Se muestran las cajas de las personas y sus IDs temporales.
- Se muestran los contadores de cada puerta en una esquina del video.

## Salida en consola

Cuando el video termina o el usuario detiene la ejecución, el programa imprime los conteos finales con este formato:

```text
Door 1: 12
Door 2: 7
Door 3: 4
```

## Cómo funciona

- Las puertas no se detectan automáticamente.
- El usuario define manualmente las puertas al inicio.
- Las personas se detectan con YOLO.
- Cada persona se sigue entre fotogramas con un identificador temporal.
- El conteo aumenta solo cuando una persona cruza la línea o zona de conteo de una puerta.

## Limitaciones

- La precisión depende de cómo se dibujen las puertas al inicio.
- Si el video tiene mucha oclusión o movimiento rápido, el seguimiento puede perder personas brevemente.
- Si quieres usar otro modelo YOLO para personas, puedes pasar la ruta con `--person-model`.

## Ejemplo con otro modelo de personas

```bash
python main.py --video videos/video2.mp4 --person-model yolov8n.pt
```