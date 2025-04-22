import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import numpy as np

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection")
        self.root.geometry("800x600")

        # Load YOLO model
        self.model = YOLO('best_11n.pt')
        
        # Variables
        self.image_path = None
        self.current_image = None
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Buttons frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        # Buttons
        tk.Button(btn_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Predict", command=self.predict).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=5)
        
        # Image display
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)
        
    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if self.image_path:
            image = Image.open(self.image_path)
            image = image.resize((600, 400))
            self.current_image = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.current_image)
            
    def predict(self):
        if self.image_path:
            # Run inference
            results = self.model(self.image_path)
            
            # Get the plotted image with predictions
            img = results[0].plot()
            
            # Convert to PIL Image and display
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((600, 400))
            self.current_image = ImageTk.PhotoImage(img)
            self.image_label.configure(image=self.current_image)
            
    def reset(self):
        self.image_path = None
        self.image_label.configure(image='')
        self.current_image = None

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
