import os
import random
import time
from pathlib import Path
from tkinter import Toplevel, Canvas, Label, Button, PhotoImage, simpledialog
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets\frame0")

DETECTED_SIGNS_PATH = OUTPUT_PATH / 'SignEc-resultados-palabras-y-tests'
DETECTED_SIGNS_PATH.mkdir(parents=True, exist_ok=True)

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

class ReceptiveSkillTest:
    def __init__(self, main_app):
        self.main_app = main_app
        self.running = False
        self.score = 0
        self.timer = 60  # Tiempo total del desafío
        self.model = YOLO("best.pt")
        self.cap = None
        self.contador_test = 0
        self.letter_to_match = ""
        self.preparation_time = 5
        self.remaining_time_for_letter = 10
        self.used_letters = set()  # Set para guardar letras ya utilizadas
        self.letters_shown = 0  # Contador de letras mostradas

    def start_test(self):
        # Esconder ventana principal
        self.main_app.root.withdraw()

        # Crear la ventana del desafío
        self.challenge_window = Toplevel()
        self.challenge_window.title("SignEc - Desafío Receptive Skill Test")
        self.challenge_window.geometry("1100x680")
        self.challenge_window.configure(bg="#FFFFFF")
        self.challenge_window.protocol("WM_DELETE_WINDOW", self.end_test)

        # Canvas principal
        self.canvas = Canvas(
            self.challenge_window,
            bg="#FFFFFF",
            height=680,
            width=1100,
            bd=0,
            highlightthickness=0,
            relief="ridge",
        )
        self.canvas.place(x=0, y=0)

        # Fondo del desafío
        self.background_image = PhotoImage(file=relative_to_assets("challengue_bg.png"))
        self.canvas.create_image(0, 0, image=self.background_image, anchor="nw")

        # Espacio de cámara (izquierda)
        self.camera_canvas = Canvas(self.challenge_window, bg="#D9D9D9", width=640, height=480)
        self.camera_canvas.place(x=30, y=100)

        # Espacio de texto (derecha)
        self.right_frame = Canvas(self.challenge_window, bg="#FFFFFF", width=370, height=480)
        self.right_frame.place(x=700, y=100)

        # Puntaje
        self.score_label = Label(
            self.right_frame, text="Puntaje: 0", font=("IMPACT", 40), fg="#FFFFFF", bg="#ffce00"
        )
        self.score_label.place(relx=0.5, rely=0.1, anchor="center")

        # Letra aleatoria centrada
        self.letter_label = Label(
            self.right_frame, text="", font=("IMPACT", 30), fg="blue", bg="#FFFFFF"
        )
        self.letter_label.place(relx=0.5, rely=0.4, anchor="center")

        # Preparación tiempo centrado
        self.preparation_label = Label(
            self.right_frame, text="¿PREPARADO?...", font=("ROG Fonts", 25), fg="#009cff", bg="#FFFFFF"
        )
        self.preparation_label.place(relx=0.5, rely=0.6, anchor="center")

        # Inicializar cámara
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Inicia el test
        self.running = True
        self.update_camera()
        self.challenge_window.after(1000, self.update_timer)

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_canvas.create_image(0, 0, anchor="nw", image=imgtk)
            self.camera_canvas.imgtk = imgtk

            # Detección del modelo
            results = self.model.predict(source=frame, stream=True)
            for result in results:
                if result.boxes:
                    detected_letter = self.model.names[int(result.boxes.cls[0])]
                    if detected_letter == self.letter_to_match:
                        self.score += 1
                        self.score_label.config(text=f"Puntaje: {self.score}")
                        self.generate_new_letter()

        if self.running:
            self.challenge_window.after(10, self.update_camera)

    def generate_new_letter(self):
        available_letters = [letter for letter in "ABCDEFGHIKLMNOPQRSTUVWXY" if letter not in self.used_letters]
        if available_letters:
            self.letter_to_match = random.choice(available_letters)
            self.used_letters.add(self.letter_to_match)
            self.letter_label.config(text=f"Haz la seña: {self.letter_to_match}")
            self.remaining_time_for_letter = 10
            self.letters_shown += 1
        else:
            self.end_test()

    def update_timer(self):
        self.timer -= 1
        if self.timer > 55 - self.preparation_time:
            self.preparation_label.config(text=f"¿PREPARADO?... {self.timer - (55 - self.preparation_time)}")
        elif self.timer == 55 - self.preparation_time:
            self.preparation_label.config(text="")
            self.generate_new_letter()
        else:
            self.remaining_time_for_letter -= 1
            if self.remaining_time_for_letter <= 0:
                self.generate_new_letter()

        if self.timer <= 0 or self.letters_shown == 5:
            self.end_test()
        else:
            self.challenge_window.after(1000, self.update_timer)

    def end_test(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        feedback = "¡Buen trabajo!" if self.score >= 3 else "Puedes hacerlo mejor la próxima vez."
        name = simpledialog.askstring("Finalizado", f"{feedback}\nIngresa tu nombre:")
        if name:
            self.contador_test += 1
            filename = os.path.join(OUTPUT_PATH, f"{name}_testN°{self.contador_test}.txt")
            file_path = DETECTED_SIGNS_PATH / filename
            with open(file_path, "w") as file:
                file.write(f"Nombre: {name}\nPuntaje: {self.score}\n{feedback}\n")

        self.challenge_window.destroy()
        self.main_app.root.deiconify()
