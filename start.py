import cv2
from facerec_model import Facerecognition
import tkinter as tk
from tkinter import Button,Frame
from PIL import Image, ImageTk

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detect App")
        self.root.geometry("800x600")

        self.header_label = tk.Label(self.root, text="Face Detect and Display Name", font=("Helvetica", 16))
        self.header_label.pack(pady=10)

        button_width = 10  
        button_height = 2

        button_frame = Frame(root)
        button_frame.pack(side=tk.TOP, pady=10)

        self.start_button = Button(button_frame, text="Start", command=self.start, width=button_width, height=button_height,bg="#2196F3",fg="white")
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.pause_button = Button(button_frame, text="Pause", command=self.pause, width=button_width, height=button_height,bg="red",fg="white")
        self.pause_button.pack(side=tk.LEFT, padx=10)

        self.resume_button = Button(button_frame, text="Resume", command=self.resume, width=button_width, height=button_height,bg="green",fg="white")
        self.resume_button.pack(side=tk.LEFT, padx=10)

        self.exit_button = Button(button_frame, text="Exit", command=self.exit, width=button_width, height=button_height,bg="#607D8B",fg="white")
        self.exit_button.pack(side=tk.LEFT, padx=10)

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.cap = None
        self.running = False

        self.sfr = Facerecognition()
        self.sfr.load_images("images/")

    def start(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.show_frame()

    def pause(self):
        self.running = False

    def resume(self):
        self.running = True
        self.show_frame()

    def exit(self):
        if self.running:
            self.running = False
            self.cap.release()
        self.root.destroy()

    def show_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                face_loci, face_name = self.sfr.detect_faces(frame)
                for face_loc, name in zip(face_loci, face_name):
                    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (8, 8, 288), 4)

                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            self.video_label.after(2, self.show_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
