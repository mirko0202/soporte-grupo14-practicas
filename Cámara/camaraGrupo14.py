import cv2
import os

cap = cv2.VideoCapture(0)
i = 0

grabando = False
video = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo obtener el fotograma de la cámara.")
        break

    frame_visual = frame.copy()

    if grabando and video is not None:
        video.write(frame)
        cv2.putText(frame_visual, "* GRABANDO *", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Mi primer OpenCV", frame_visual)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('f'):
        cv2.imwrite(f'mi_frame{i}.png', frame)
        print(f"Foto guardada como: mi_frame{i}.png")
        i += 1

    elif key == ord('g'):
        if not grabando:
            alto, ancho, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video = cv2.VideoWriter('video_grabado.avi', fourcc, 20.0, (ancho, alto))
            grabando = True
            print("Grabación iniciada...")
        else:
            grabando = False
            if video is not None:
                video.release()
                video = None
            print("Grabación detenida y guardada.")

    elif key == ord('q'):
        print("Cerrando aplicación...")
        break

if video is not None:
    video.release()
cap.release()
cv2.destroyAllWindows()


