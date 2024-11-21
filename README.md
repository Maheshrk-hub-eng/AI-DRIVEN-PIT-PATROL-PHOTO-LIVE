# AI-DRIVEN-PIT-PATROL-PHOTO-LIVE
from ultralytics import YOLO
import cv2
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import threading

# Load the YOLO model
model = YOLO("best.pt")
class_names = model.names

# Global Variables
frame = None
stop_flag = False


def process_frame(img):
    """Process a single frame for pothole detection."""
    img = cv2.resize(img, (1020, 500))  # Resize for faster processing
    h, w, _ = img.shape
    results = model.predict(img)
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
    if masks is not None:
        masks = masks.data.cpu()
        for seg, box in zip(masks.data.cpu().numpy(), boxes):
            seg = cv2.resize(seg, (w, h))
            contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                d = int(box.cls)
                c = class_names[d]
                x, y, x1, y1 = cv2.boundingRect(contour)
                cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return img


def update_frame(label, cap):
    """Update the live video feed in the GUI."""
    global frame, stop_flag
    if stop_flag:
        cap.release()
        return
    ret, img = cap.read()
    if not ret:
        label.after(10, update_frame, label, cap)
        return
    # Process the frame
    img = process_frame(img)
    # Convert frame to ImageTk format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame = ImageTk.PhotoImage(Image.fromarray(img_rgb))
    # Update the GUI label
    label.configure(image=frame)
    label.image = frame
    # Schedule the next frame update
    label.after(10, update_frame, label, cap)


def start_camera(label):
    """Start the camera and begin live detection."""
    global stop_flag
    stop_flag = False
    cap = cv2.VideoCapture(0)  # Open default camera
    # Create a separate thread for video capture and processing
    threading.Thread(target=update_frame, args=(label, cap), daemon=True).start()


def stop_camera():
    """Stop the live detection."""
    global stop_flag
    stop_flag = True


# GUI Setup
root = tk.Tk()
root.title("AI DRIVEN PATH HOLE DETECTOR ")
root.geometry("1100x600")

# Frame for live feed
video_label = Label(root)
video_label.pack()
# Load the background image using Pillow
bg_image = Image.open("pp.jpg")  # Replace with your image path
bg_image_resized = bg_image.resize((1550, 890))  # Resize image to fit window size
bg_photo = ImageTk.PhotoImage(bg_image_resized)
# Create a Label widget to display the background image
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)  # Stretch the image to fill the window
def remove_bg_image():
    bg_label.destroy()

# heading
heading_label = Label(root, text="AI-Driven Path Hole Detector", font=("Times", 30, "bold"), fg="white",bg="blue3")
heading_label.pack(pady=10)

# Start/Stop Buttons
start_button = tk.Button(root, text="Start Detection",bg="blue",fg="white",font=("Arial", 20), command=lambda: [start_camera(video_label), remove_bg_image()])
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Detection",bg="red",fg="white",font=("Arial", 20), command=stop_camera)
stop_button.pack(pady=10)

# Start GUI
root.mainloop()
