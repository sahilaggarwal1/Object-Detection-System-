import os
import cv2
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from datetime import datetime
from PIL import Image, ImageTk

# Load the YOLOv3 model and configuration
config_file = 'yolov3.cfg'
weights_file = 'yolov3.weights'
class_labels_file = 'coco.names'

net = cv2.dnn.readNet(config_file, weights_file)

# Load class labels
with open(class_labels_file, 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

# Known object width (e.g., average width of a person in cm) and focal length (to be calibrated)
KNOWN_WIDTH = 50.0  # in cm
FOCAL_LENGTH = 700  # needs to be calibrated for your camera

# Global list to store detected objects with confidence values
detected_objects_list = []


def get_output_layers(net):
    """Get the names of the output layers of the YOLO model."""
    layer_names = net.getLayerNames()
    output_layers_indices = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in output_layers_indices]
    return output_layers


def draw_detected_objects(frame, class_ids, confidences, boxes, class_labels, confidence_threshold):
    """Draw bounding boxes around detected objects with distance and time details."""
    detected_objects = []  # To store details of all detected objects

    for i in range(len(class_ids)):
        class_id = class_ids[i]
        confidence = confidences[i]
        if confidence > confidence_threshold:
            color = (255, 0, 0)
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{class_labels[class_id]}: {confidence:.2f}"
            distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, w)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"{label}, {distance:.2f} cm",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, current_time, (x, y + h + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            object_details = {
                "label": class_labels[class_id],
                "confidence": confidence,
                "distance": distance,
                "time": current_time
            }
            detected_objects.append(object_details)

    update_detected_objects_list(detected_objects)


def calculate_distance(known_width, focal_length, observed_width):
    """Calculate distance from the camera to the object."""
    distance = (known_width * focal_length) / observed_width
    return distance


def detect_objects_in_image(image_path, confidence_threshold):
    """Detect objects in an image file."""
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # Prepare the image for YOLOv3
    blob = cv2.dnn.blobFromImage(
        img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Forward pass through the network
    outs = net.forward(get_output_layers(net))

    # Process the detected objects
    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Draw bounding boxes
    draw_detected_objects(img, class_ids, confidences,
                          boxes, class_labels, confidence_threshold)

    # Resize the image to fit in the window
    img_resized = cv2.resize(img, (800, 600))
    # Show the image in a separate window
    cv2.imshow('Detected Objects in Image', img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_objects_in_video(video_path, confidence_threshold):
    """Detect objects in a video file."""
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        # Prepare the frame for YOLOv3
        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        # Forward pass through the network
        outs = net.forward(get_output_layers(net))

        # Process the detected objects
        class_ids, confidences, boxes = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Draw bounding boxes
        draw_detected_objects(
            frame, class_ids, confidences, boxes, class_labels, confidence_threshold)
        frame_resized = cv2.resize(frame, (800, 600))
        cv2.imshow('Detected Objects in Video', frame_resized)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # Press 'q' or 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_objects_in_live_video(confidence_threshold):
    """Detect objects in live video from the webcam."""
    cap = cv2.VideoCapture(0)  # Open the default camera (index 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        # Prepare the frame for YOLOv3
        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        # Forward pass through the network
        outs = net.forward(get_output_layers(net))

        # Process the detected objects
        class_ids, confidences, boxes = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Draw bounding boxes
        draw_detected_objects(
            frame, class_ids, confidences, boxes, class_labels, confidence_threshold)
        frame_resized = cv2.resize(frame, (800, 600))
        cv2.imshow('Detected Objects in Live Video', frame_resized)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # Press 'q' or 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


def choose_video():
    """Open a file dialog to choose a video file and detect objects in it."""
    file_path = filedialog.askopenfilename()
    if file_path:
        detect_objects_in_video(file_path, confidence_threshold.get() / 100)


def process_live_video():
    """Process live video from the webcam."""
    detect_objects_in_live_video(confidence_threshold.get() / 100)


def choose_image():
    """Open a file dialog to choose an image file and detect objects in it."""
    file_path = filedialog.askopenfilename()
    if file_path:
        detect_objects_in_image(file_path, confidence_threshold.get() / 100)


def display_image(img):
    """Display the image in the Tkinter window."""
    # This function is now empty to prevent image display
    pass


def start_detection(option):
    """Start the detection process based on the selected option."""
    if option == "image":
        choose_image()
    elif option == "video":
        choose_video()
    elif option == "live":
        process_live_video()


def update_detected_objects_list(detected_objects):
    """Update the detected objects list in the text widget."""
    detected_objects_text.config(state=tk.NORMAL)
    detected_objects_text.delete(1.0, tk.END)
    for obj in detected_objects:
        detected_objects_text.insert(
            tk.END, f"{obj['label']}: {obj['confidence']:.2f}, Distance: {obj['distance']:.2f} cm, Time: {obj['time']}\n")
    detected_objects_text.config(state=tk.DISABLED)


# Create the main application window
app = tk.Tk()
app.title("Object Detection Application")
app.geometry("800x600")
app.resizable(True, True)

# Main frame for detection and results
main_frame = ttk.Frame(app)
main_frame.pack(fill=tk.BOTH, expand=True)

title_label = ttk.Label(
    main_frame, text="Select Detection Mode", font=("Arial", 16))
title_label.pack(pady=20)

image_button = ttk.Button(main_frame, text="Image",
                          command=lambda: start_detection("image"))
image_button.pack(pady=5)

video_button = ttk.Button(main_frame, text="Video",
                          command=lambda: start_detection("video"))
video_button.pack(pady=5)

live_video_button = ttk.Button(
    main_frame, text="Live Video", command=lambda: start_detection("live"))
live_video_button.pack(pady=5)

# Confidence threshold slider
confidence_threshold = tk.IntVar(value=60)
confidence_slider = ttk.Scale(
    main_frame, from_=0, to=100, orient='horizontal', variable=confidence_threshold)
confidence_slider.pack(pady=20)
confidence_label = ttk.Label(main_frame, text="Confidence Threshold: 60%")
confidence_label.pack(pady=5)


def update_confidence_label(event):
    confidence_label.config(
        text=f"Confidence Threshold: {confidence_threshold.get()}%")


confidence_slider.bind("<Motion>", update_confidence_label)

# Text widget to display detected objects and their details
detected_objects_text = tk.Text(main_frame, height=10, state=tk.DISABLED)
detected_objects_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Start the Tkinter main loop
app.mainloop()
