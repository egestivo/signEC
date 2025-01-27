import random
from tkinter import Tk, Canvas, Button, PhotoImage, Text, Toplevel, simpledialog, ttk
import tkinter
from tkinter.scrolledtext import ScrolledText
from pathlib import Path
import cv2
from PIL import Image, ImageTk
import time
from desafio import ReceptiveSkillTest
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
        self.root.geometry("1500x832")
        self.root.configure(bg="#FFFFFF")

        self.last_detection_time = 0
        self.detection_delay = 6  # seconds
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

        # Camera rectangle
        self.camera_rect = self.canvas.create_rectangle(62.0, 245.0, 702.0, 725.0, fill="#D9D9D9", outline="")

        # Text box for detected letters
        self.text_box = ScrolledText(
            self.root, wrap="word", font=("Arial", 16), state="disabled"
        )
        self.text_box.place(x=770.0, y=270.0, width=460.0, height=59.0)

        # Buttons
        self.setup_buttons()

        # Logo
        self.logo_image = PhotoImage(file=relative_to_assets("image_1.png"))
        self.canvas.create_image(991.0, 124.0, image=self.logo_image)

        self.canvas.create_text(
            771.0,
            357.0,
            anchor="nw",
            text="INICIAR",
            fill="#000000",
            font=("AlumniSans Regular", 50 * -1)
        )

        self.canvas.create_text(
            770.0,
            507.0,
            anchor="nw",
            text="GUÍA DE\nSEÑAS",
            fill="#000000",
            font=("AlumniSans Regular", 50 * -1)
        )

        self.canvas.create_text(
            1083.0,
            355.0,
            anchor="nw",
            text="REINICIAR",
            fill="#000000",
            font=("AlumniSans Regular", 50 * -1)
        )

        self.canvas.create_text(
            1096.0,
            507.0,
            anchor="nw",
            text="DESAFÍO",
            fill="#000000",
            font=("AlumniSans Regular", 50 * -1)
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
        self.start_button.place(x=781.0, y=415.0, width=90.0, height=90.0)

        button_image_2 = PhotoImage(file=relative_to_assets("button_2.png"))
        self.reset_button = Button(
            image=button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=self.reset_text,
            relief="flat",
        )
        self.reset_button.image = button_image_2
        self.reset_button.place(x=1112.0, y=417.0, width=90.0, height=90.0)

        button_image_3 = PhotoImage(file=relative_to_assets("button_3.png"))
        self.guide_button = Button(
            image=button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=self.show_guide,
            relief="flat",
        )
        self.guide_button.image = button_image_3
        self.guide_button.place(x=780.0, y=635.0, width=90.0, height=90.0)

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
        self.challenge_button.place(x=1112.0, y=635.0, width=90.0, height=90.0)

    def start_camera(self):
        # Variables
        self.last_detection_time = 0
        self.detection_delay = 5  # seconds
        self.detected_text = ""  # Reset the detected text
        self.running = True

        # Model
        self.model = YOLO('best.pt')

        # Start Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Match the width of the rectangle
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Match the height of the rectangle

        # Camera frame
        self.camera_frame = self.canvas.create_image(62.0, 245.0, anchor="nw")

        self.update_camera()

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run YOLO model every `detection_delay` seconds
            current_time = time.time()
            if current_time - self.last_detection_time > self.detection_delay:
                results = self.model.predict(source=frame, stream=True)
                for result in results:
                    if result.boxes:  # Check for detections
                        self.last_detection_time = current_time
                        # Assuming the first detected class corresponds to the letter
                        detected_letter = self.model.names[int(result.boxes.cls[0])]
                        self.update_text_box(detected_letter)
                        break

            # Convert image to tkinter-compatible format
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480))  # Adjust the image to the size of the camera rectangle
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update the camera frame on canvas
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

    def save_detected_word(self):
        # Save the entire word to a file
        self.detection_counter += 1
        file_name = f"palabra_{self.detection_counter}.txt"
        file_path = DETECTED_SIGNS_PATH / file_name
        with open(file_path, "w") as f:
            f.write(self.detected_text)

    def reset_text(self):
        # Save the detected word before resetting
        if self.detected_text:
            self.save_detected_word()

        self.detected_text = ""  # Clear the detected word
        self.text_box.configure(state="normal")
        self.text_box.delete("1.0", "end")
        self.text_box.configure(state="disabled")

    def show_guide(self):
        guide_window = Toplevel(self.root)
        guide_window.title("Guía de Señas")
        guide_window.geometry("500x500")
        guide_image = PhotoImage(file=relative_to_assets("guide.png"))
        guide_label = Canvas(guide_window, width=500, height=500)
        guide_label.create_image(250, 250, image=guide_image)
        guide_label.image = guide_image
        guide_label.pack()

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    app = SignDetectionApp(Tk())
    app.root.protocol("WM_DELETE_WINDOW", app.on_close)
    app.root.mainloop()
