import tkinter as tk
import vosk
import pyaudio
import json
import threading
import random
import os
import sys

COLORES = {
    "rojo": "red",
    "azul": "blue",
    "verde": "green",
    "amarillo": "yellow",
    "naranja": "orange",
    "blanco": "white",
    "violeta": "purple" 
}

COMBINACIONES = {
    "verde": ["amarillo", "azul"],
    "naranja": ["amarillo", "rojo"],
    "violeta": ["rojo", "azul"]
}

class JuegoColoresVosk:
    def __init__(self, root):
        self.root = root
        self.root.title("Adivina la Mezcla por Voz")
        self.root.geometry("600x500")
        
        if not os.path.exists("model"):
            print("ERROR: No se encontró la carpeta 'model'.")
            sys.exit(1)
            
        self.modelo = vosk.Model("model")
        self.color_actual = ""

        self.label_titulo = tk.Label(root, text="", font=("Arial", 28, "bold"))
        self.label_titulo.pack(pady=40)
        
        self.label_escuchando = tk.Label(root, text="...", font=("Courier", 20, "italic"), fg="#333333")
        self.label_escuchando.pack(expand=True)

        self.label_estado = tk.Label(root, text="Cargando motor...", font=("Arial", 12))
        self.label_estado.pack(side="bottom", pady=20)

        self.siguiente_color()
        threading.Thread(target=self.escuchar_vosk, daemon=True).start()

    def siguiente_color(self):
        opciones = list(COLORES.keys())
        if self.color_actual in opciones:
            opciones.remove(self.color_actual)
            
        self.color_actual = random.choice(opciones)
        color_hex = COLORES[self.color_actual]

        color_texto = "black" if self.color_actual in ["blanco", "amarillo"] else "white"

        if self.color_actual in COMBINACIONES:
            self.label_titulo.config(text="¡Dime los colores que lo forman!", bg=color_hex, fg=color_texto)
        else:
            self.label_titulo.config(text="¡Nombra el color!", bg=color_hex, fg=color_texto)
            
        self.root.config(bg=color_hex)
        self.label_escuchando.config(bg=color_hex, text="...")
        self.label_estado.config(bg=color_hex, fg=color_texto)

    def escuchar_vosk(self):
        recognizer = vosk.KaldiRecognizer(self.modelo, 16000)
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        stream.start_stream()
        
        self.root.after(0, lambda: self.label_estado.config(text="Micrófono activo"))

        while True:
            data = stream.read(4000, exception_on_overflow=False)
            
            if recognizer.AcceptWaveform(data):
                resultado = json.loads(recognizer.Result())
                texto = resultado.get("text", "").lower()
            else:
                resultado = json.loads(recognizer.PartialResult())
                texto = resultado.get("partial", "").lower()
            
            if texto:

                self.root.after(0, lambda t=texto: self.label_escuchando.config(text=f"> {t} <"))

                acierto = False
                
                if self.color_actual in COMBINACIONES:

                    req1, req2 = COMBINACIONES[self.color_actual]
  
                    if req1 in texto and req2 in texto:
                        acierto = True
                else:
                    if self.color_actual in texto:
                        acierto = True
                if acierto:
                    recognizer.Reset()
                    self.root.after(0, lambda: self.label_estado.config(text="¡Excelente!"))
                    self.root.after(400, self.siguiente_color)

if __name__ == "__main__":
    ventana = tk.Tk()
    app = JuegoColoresVosk(ventana)
    ventana.mainloop()