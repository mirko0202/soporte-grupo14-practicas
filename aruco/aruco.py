import cv2
import numpy as np

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

cap = cv2.VideoCapture(0)

print("Buscando marcadores ArUco 6x6... (ESC para salir)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    centros_proyectados = []

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            centro_x = int(np.mean(corners[i][0][:, 0]))
            centro_y = int(np.mean(corners[i][0][:, 1]))
            centros_proyectados.append([centro_x, centro_y])

            cv2.circle(frame, (centro_x, centro_y), 6, (0, 0, 255), -1)

        cantidad = len(centros_proyectados)

        if cantidad >= 3:
     
            puntos = np.array(centros_proyectados, dtype=np.int32)

            hull = cv2.convexHull(puntos)
 
            cv2.polylines(frame, [hull], True, (255, 0, 0), 3)
            
        elif cantidad == 2:

            cv2.line(frame, tuple(centros_proyectados[0]), tuple(centros_proyectados[1]), (255, 0, 0), 3)

    cv2.imshow("Red de ArUco Conectados", frame)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()