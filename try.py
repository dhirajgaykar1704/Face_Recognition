import cv2
from facerec_model import Facerecognition
import tkinter as tk
from tkinter import Button, Frame, Label
from tkinter import ttk
from PIL import Image, ImageTk

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detect App")
        self.root.geometry("800x650")
        self.root.resizable(0,0)

        self.header_label = tk.Label(self.root, text="Face Detect and Display Name", font=("Helvetica", 16))
        self.header_label.pack(pady=10)

        self.style = ttk.Style()
        self.style.configure('TButton', font=('Helvetica', 12), padding=10)
        
        self.style.configure('TLabel', font=('Helvetica', 12), padding=10)
        
        button_frame = Frame(root)
        button_frame.pack(side=tk.TOP, pady=10)

        self.start_button = ttk.Button(button_frame, text="Start", command=self.start)
        self.start_button.grid(row=0, column=0, padx=10)

        self.pause_button = ttk.Button(button_frame, text="Pause", command=self.pause)
        self.pause_button.grid(row=0, column=1, padx=10)

        self.resume_button = ttk.Button(button_frame, text="Resume", command=self.resume)
        self.resume_button.grid(row=0, column=2, padx=10)

        self.exit_button = ttk.Button(button_frame, text="Exit", command=self.exit)
        self.exit_button.grid(row=0, column=3, padx=10)

        self.video_label = Label(root)
        self.video_label.pack()

        self.status_label = Label(root, text="Status: Waiting")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.cap = None
        self.running = False

        self.sfr = Facerecognition()
        self.sfr.load_images("images/")

        self.bind_keys()

    def bind_keys(self):
        self.root.bind('<s>', lambda event: self.start())
        self.root.bind('<p>', lambda event: self.pause())
        self.root.bind('<r>', lambda event: self.resume())
        self.root.bind('<e>', lambda event: self.exit())

    def start(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.update_status("Error: Unable to open camera")
                return
            self.running = True
            self.update_status("Status: Running")
            self.show_frame()

    def pause(self):
        self.running = False
        self.update_status("Status: Paused")

    def resume(self):
        if not self.running:
            self.running = True
            self.update_status("Status: Resumed")
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
            self.video_label.after(10, self.show_frame)

    def update_status(self, message):
        self.status_label.config(text=message)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
