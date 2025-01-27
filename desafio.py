import os
import random
import time
from tkinter import Toplevel, Canvas, Label, Button, simpledialog
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

class ReceptiveSkillTest:
    def __init__(self, main_app):
        self.main_app = main_app
        self.running = False
        self.score = 0
        self.timer = 60 
        self.model = YOLO("best.pt")
        self.cap = None
        self.letter_to_match = ""
        self.preparation_time = 5
        self.remaining_time_for_letter = 10

    def start_test(self):
        #Esconder ventana principal 
        self.main_app.root.withdraw()

        #Abrimos ventana del desafío
        self.challenge_window = Toplevel()
        self.challenge_window.title("RECEPTIVE SKILL TEST")
        self.challenge_window.geometry("1500x832")
        self.challenge_window.protocol("WM_DELETE_WINDOW", self.end_test)

        #Espacio de cámara
        self.canvas = Canvas(self.challenge_window, bg="#D9D9D9", width=640, height=480)
        self.canvas.place(x=50, y=150)

        #Puntaje
        self.score_label = Label(
            self.challenge_window, text="Puntaje: 0", font=("Arial", 24)
        )
        self.score_label.place(x=750, y=150)

        #Letra aleatoria
        self.letter_label = Label(
            self.challenge_window, text="", font=("Arial", 48), fg="blue"
        )
        self.letter_label.place(x=750, y=300)

        #Preparación tiempo
        self.preparation_label = Label(
            self.challenge_window, text="Prepárate...", font=("Arial", 36), fg="red"
        )
        self.preparation_label.place(x=750, y=400)

        #Inicia la cámara
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        #Inicia el test
        self.running = True
        self.update_camera()
        self.challenge_window.after(1000, self.update_timer)

    def generate_new_letter(self):
        self.letter_to_match = random.choice("ABCDEFGHIKLMNOPQRSTUVWXY")
        self.letter_label.config(text=f"Haz la seña: {self.letter_to_match}")
        self.remaining_time_for_letter = 10

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
            self.canvas.imgtk = imgtk

            #hacer la predicción según el modelo YOLO best.pt
            results = self.model.predict(source=frame, stream=True)
            for result in results:
                if result.boxes:  #Si hay detección entonces...
                    detected_letter = self.model.names[int(result.boxes.cls[0])]
                    if detected_letter == self.letter_to_match:
                        self.score += 1
                        self.score_label.config(text=f"Puntaje: {self.score}")
                        self.generate_new_letter()

        if self.running:
            self.challenge_window.after(10, self.update_camera)

    def update_timer(self):
        self.timer -= 1

        if self.timer > 55 - self.preparation_time:
            self.preparation_label.config(text=f"Prepárate... {self.timer - (55 - self.preparation_time)}")
        elif self.timer == 55 - self.preparation_time:
            self.preparation_label.config(text="")
            self.generate_new_letter()
        else:
            self.remaining_time_for_letter -= 1
            if self.remaining_time_for_letter <= 0:
                self.generate_new_letter()

        if self.timer <= 0 or self.score == 5:
            self.end_test()
        else:
            self.challenge_window.after(1000, self.update_timer)

    def end_test(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        #Mensaje de retroalimentación xd (no le puse fracasado porque no me lo permiten :v)
        feedback = ""
        if self.score <= 2:
            feedback = "Vamos, puedes hacerlo mejor..."
        elif self.score <= 4:
            feedback = "Ya casi lo tienes!"
        else:
            feedback = "¡Que CRACK!"

        #guarda el puntajeceems
        name = simpledialog.askstring("Finalizado", f"{feedback}\nIngresa tu nombre:")
        if name:
            count = len([f for f in os.listdir('SignEc-resultados-palabras-y-tests') if os.path.isfile(os.path.join('SignEc-resultados-palabras-y-tests', f))]) + 1
            filename = os.path.join('SignEc-resultados-palabras-y-tests', f"{name}_test_{count}.txt")
            with open(filename, "w") as file:
                file.write(f"Nombre: {name}\nPuntaje: {self.score}/5\nMensaje: {feedback}\n")

        # cierra el desafio y abre nuevamente la ventana inicial
        self.challenge_window.destroy()
        self.main_app.root.deiconify()
