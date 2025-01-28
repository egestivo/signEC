from tkinter import Tk, Canvas, Button, PhotoImage, Text, Toplevel, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText
from pathlib import Path
import cv2
from PIL import Image, ImageTk
import time
from test import ReceptiveSkillTest
from ultralytics import YOLO


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets\frame0")

DETECTED_SIGNS_PATH = OUTPUT_PATH / 'SignEc-resultados-palabras-y-tests'
DETECTED_SIGNS_PATH.mkdir(parents=True, exist_ok=True)

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

class SignDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SignEc - Lenguaje de Señas")
        self.root.geometry("1450x832")
        self.root.configure(bg="#FFFFFF")

        self.last_detection_time = 0
        self.detection_delay = 6
        self.detected_text = ""
        self.running = False
        self.detection_counter = 0

        self.setup_ui()
        self.cap = None

    def setup_ui(self):
        # Canvas
        self.canvas = Canvas(
            self.root,
            bg="#FFFFFF",
            height=832,
            width=1450,
            bd=0,
            highlightthickness=0,
            relief="ridge",
        )
        self.canvas.place(x=0, y=0)

        #background de niños
        self.background_image = PhotoImage(file=relative_to_assets("canvas-bg.png"))
        self.canvas.create_image(0, 0, image=self.background_image, anchor="nw")

        # cámara
        self.camera_rect = self.canvas.create_rectangle(62.0, 245.0, 702.0, 725.0, fill="#D9D9D9", outline="")

        # text box para escribir
        self.text_box = ScrolledText(
            self.root, wrap="word", font=("Times New Roman", 16), state="disabled"
        )
        self.text_box.place(x=770.0, y=270.0, width=460.0, height=59.0)

        # botones
        self.setup_buttons()

        # Logo
        self.logo_image = PhotoImage(file=relative_to_assets("image_1.png"))
        self.canvas.create_image(991.0, 124.0, image=self.logo_image)

        self.logo_bnw_image = PhotoImage(file=relative_to_assets("logo_bnw.png"))

        self.iniciartext = self.canvas.create_text(
            771.0,
            357.0,
            anchor="nw",
            text="INICIAR",
            fill="#000000",
            font=("IMPACT", 50 * -1)
        )

        self.reiniciartext = self.canvas.create_text(
            770.0,
            507.0,
            anchor="nw",
            text="GUÍA DE\nSEÑAS",
            fill="#000000",
            font=("IMPACT", 50 * -1)
        )

        self.guiatext = self.canvas.create_text(
            1030.0,
            355.0,
            anchor="nw",
            text="REINICIAR",
            fill="#000000",
            font=("IMPACT", 50 * -1)
        )

        self.desafiotext = self.canvas.create_text(
            1040.0,
            540.0,
            anchor="nw",
            text="DESAFÍO",
            fill="#000000",
            font=("IMPACT", 50 * -1)
        )

        self.bnwguitext = self.canvas.create_text(
            1332.0,
            370.0,
            anchor="nw",
            text="B/N GUI",
            fill="#000000",
            font=("IMPACT", 25 * -1)    
        )

        self.colorguitext = self.canvas.create_text(
            1319.0,
            588.0,
            anchor="nw",
            text="COLOR GUI",
            fill="#000000",
            font=("IMPACT", 25 * -1)    
        )

    def setup_buttons(self):
        button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
        self.start_button = Button(
            image=button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=self.start_camera,
            relief="flat",
        )
        self.start_button.image = button_image_1
        self.start_button.place(x=800.0, y=415.0, width=90.0, height=90.0)

        button_image_2 = PhotoImage(file=relative_to_assets("button_2.png"))
        self.reset_button = Button(
            image=button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=self.reset_text,
            relief="flat",
        )
        self.reset_button.image = button_image_2
        self.reset_button.place(x=1080.0, y=417.0, width=90.0, height=90.0)

        button_image_3 = PhotoImage(file=relative_to_assets("button_3.png"))
        self.guide_button = Button(
            image=button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=self.show_guide,
            relief="flat",
        )
        self.guide_button.image = button_image_3
        self.guide_button.place(x=800.0, y=635.0, width=90.0, height=90.0)

        def iniciar_desafio():
            challenge = ReceptiveSkillTest(main_app=self)
            challenge.start_test()

        button_image_4 = PhotoImage(file=relative_to_assets("button_4.png"))
        self.challenge_button = Button(
            image=button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=iniciar_desafio,
            relief="flat",
        )

        self.challenge_button.image = button_image_4
        self.challenge_button.place(x=1080.0, y=635.0, width=90.0, height=90.0)

        # Botón de Color
        button_color_image = PhotoImage(file=relative_to_assets("color_button.png"))
        self.color_button = Button(
            image=button_color_image,
            borderwidth=0,
            highlightthickness=0,
            command=self.set_color_mode,
            relief="flat",
        )
        self.color_button.image = button_color_image
        self.color_button.place(x=1325.0, y=635.0, width=90.0, height=90.0)

        # Botón de Blanco y Negro
        button_bnw_image = PhotoImage(file=relative_to_assets("bnw_button.png"))
        self.bnw_button = Button(
            image=button_bnw_image,
            borderwidth=0,
            highlightthickness=0,
            command=self.set_bnw_mode,
            relief="flat",
        )
        self.bnw_button.image = button_bnw_image
        self.bnw_button.place(x=1325.0, y=417.0, width=90.0, height=90.0)

    def start_camera(self):
        
        # Variables
        self.last_detection_time = 0
        self.detection_delay = 5
        self.detected_text = "" 
        self.running = True

        # Modelo yolo
        self.model = YOLO('best.pt')

        # iniciar cámara
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # cuadrito de cámara
        self.camera_frame = self.canvas.create_image(62.0, 245.0, anchor="nw")

        self.update_camera()

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #realiza la detección según el tiempo de espera entre letra y letra
            current_time = time.time()
            if current_time - self.last_detection_time > self.detection_delay:
                results = self.model.predict(source=frame, stream=True)
                for result in results:
                    if result.boxes:  #si hay detección entonces...
                        self.last_detection_time = current_time
                        detected_letter = self.model.names[int(result.boxes.cls[0])]
                        self.update_text_box(detected_letter)
                        break

            # convertimos la imagen de opencv a un formato para tkinter
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.itemconfig(self.camera_frame, image=imgtk)
            self.canvas.image = imgtk

        if self.running:
            self.root.after(10, self.update_camera)

    def update_text_box(self, letter):
        self.detected_text += letter 
        self.text_box.configure(state="normal")
        self.text_box.delete("1.0", "end")
        self.text_box.insert("end", self.detected_text)
        self.text_box.configure(state="disabled")

    #guardamos en un archivo plano
    def save_detected_word(self):
        self.detection_counter += 1
        file_name = f"palabra_{self.detection_counter}.txt"
        file_path = DETECTED_SIGNS_PATH / file_name
        with open(file_path, "w") as f:
            f.write(self.detected_text)

    def reset_text(self):
        #guardamos el archivo antes de borrar  la palabra en el cuadro textbox
        if self.detected_text:
            self.save_detected_word()
        self.detected_text = ""  
        self.text_box.configure(state="normal")
        self.text_box.delete("1.0", "end")
        self.text_box.configure(state="disabled")

    def show_guide(self):
        guide_window = Toplevel(self.root)
        guide_window.title("SignEC - Guía de Señas")
        guide_window.geometry("1270x760")
        guide_image = PhotoImage(file=relative_to_assets("guia_senas.png"))
        guide_label = Canvas(guide_window, width=1249, height=720)
        guide_label.create_image(0,0, image=guide_image, anchor="nw")
        guide_label.image = guide_image
        guide_label.pack()

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()
    
    def set_color_mode(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.setup_ui()

    def set_bnw_mode(self):
        self.running = False
        if self.cap:
            self.cap.release()
        # cuadrito de cámara
        self.camera_frame = self.canvas.create_image(62.0, 245.0, anchor="nw")

        #fondo negro
        self.root.configure(bg="#000000")
        self.canvas.configure(bg="#000000")

        self.canvas.create_rectangle(0, 0, 1450, 832, fill="#000000", outline="")

        self.camera_rect = self.canvas.create_rectangle(62.0, 245.0, 702.0, 725.0, fill="#FFFFFF", outline="")

        # Logo
        self.logo_image = PhotoImage(file=relative_to_assets("logo_bnw.png"))
        self.canvas.create_image(991.0, 124.0, image=self.logo_image)

        self.bnw_set_buttons_text()
        
    def bnw_set_buttons_text(self):
        self.iniciartext = self.canvas.create_text(
            771.0,
            357.0,
            anchor="nw",
            text="INICIAR",
            fill="#FFFFFF",
            font=("IMPACT", 50 * -1)
        )

        self.reiniciartext = self.canvas.create_text(
            770.0,
            507.0,
            anchor="nw",
            text="GUÍA DE\nSEÑAS",
            fill="#FFFFFF",
            font=("IMPACT", 50 * -1)
        )

        self.guiatext = self.canvas.create_text(
            1030.0,
            355.0,
            anchor="nw",
            text="REINICIAR",
            fill="#FFFFFF",
            font=("IMPACT", 50 * -1)
        )

        self.desafiotext = self.canvas.create_text(
            1040.0,
            540.0,
            anchor="nw",
            text="DESAFÍO",
            fill="#FFFFFF",
            font=("IMPACT", 50 * -1)
        )

        self.bnwguitext = self.canvas.create_text(
            1332.0,
            370.0,
            anchor="nw",
            text="B/N GUI",
            fill="#FFFFFF",
            font=("IMPACT", 25 * -1)    
        )

        self.colorguitext = self.canvas.create_text(
            1319.0,
            588.0,
            anchor="nw",
            text="COLOR GUI",
            fill="#FFFFFF",
            font=("IMPACT", 25 * -1)    
        )

        button_image_1 = PhotoImage(file=relative_to_assets("button_1-bnw.png"))
        self.start_button = Button(
            image=button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=self.start_camera,
            relief="flat",
        )
        self.start_button.image = button_image_1
        self.start_button.place(x=800.0, y=415.0, width=90.0, height=90.0)

        button_image_2 = PhotoImage(file=relative_to_assets("button_2-bnw.png"))
        self.reset_button = Button(
            image=button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=self.reset_text,
            relief="flat",
        )
        self.reset_button.image = button_image_2
        self.reset_button.place(x=1080.0, y=417.0, width=90.0, height=90.0)

        button_image_3 = PhotoImage(file=relative_to_assets("button_3-bnw.png"))
        self.guide_button = Button(
            image=button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=self.show_guide,
            relief="flat",
        )
        self.guide_button.image = button_image_3
        self.guide_button.place(x=800.0, y=635.0, width=90.0, height=90.0)

        #def iniciar_desafio():
            #challenge = ReceptiveSkillTest(main_app=self)
            #challenge.start_test()

        button_image_4 = PhotoImage(file=relative_to_assets("button_4-bnw.png"))
        self.challenge_button = Button(
            image=button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=print("xd"),#iniciar_desafio
            relief="flat",
        )

        self.challenge_button.image = button_image_4
        self.challenge_button.place(x=1080.0, y=635.0, width=90.0, height=90.0)

if __name__ == "__main__":
    app = SignDetectionApp(Tk())
    app.root.protocol("WM_DELETE_WINDOW", app.on_close)
    app.root.mainloop()
