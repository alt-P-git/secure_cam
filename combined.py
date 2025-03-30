import os
import cv2
import numpy as np
import gradio as gr
from transformers import pipeline
from PIL import Image
from ultralytics import YOLO
import socket
import json
import threading

# Load MobileNetSSD for People Counting
PATH_PROTOTXT = os.path.join('saved_model/MobileNetSSD_deploy.prototxt')
PATH_MODEL = os.path.join('saved_model/MobileNetSSD_deploy.caffemodel')

CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

NET = cv2.dnn.readNetFromCaffe(PATH_PROTOTXT, PATH_MODEL)

# Load weapon detection model
classNames = ['Grenade', 'Knife', 'Pistol', 'Rifle', 'Shotgun']
weapon_model = YOLO('best.pt', verbose=False)

# Load additional models
fire_detector = pipeline("image-classification", model="EdBianchi/vit-fire-detection")
mask_detector = pipeline("image-classification", model="Heem2/Facemask-detection")

# Shared variable for alert data
alert_signal = {"Alert": 0}  # Default alert signal

# Lock for thread-safe access to alert_signal
alert_lock = threading.Lock()

# Function to receive alerts from coms.py
def receive_alerts(host='127.0.0.1', port=65432):
    """Start a server to receive alert signals from coms.py."""
    global alert_signal
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        print(f"Alert signal server started. Listening on {host}:{port}...")
        
        while True:
            conn, addr = server_socket.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    with alert_lock:
                        alert_signal = json.loads(data.decode('utf-8'))
                        print(f"Received Alert Signal: {alert_signal}")

# Start the alert signal server in a separate thread
alert_thread = threading.Thread(target=receive_alerts, daemon=True)
alert_thread.start()

def process_frame():
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        def person_counting_and_mask(frame, threshold=0.7):
            counting = 0
            masked_detected = False
            H, W = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            NET.setInput(blob)
            detections = NET.forward()

            for i in detections.shape[2]:
                confidence = detections[0, 0, i, 2]
                class_idx = int(detections[0, 0, i, 1])

                if CLASSES[class_idx] == "person" and confidence > threshold:
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    x_min, y_min, x_max, y_max = box.astype("int")
                    counting += 1
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    face_crop = frame[y_min:y_max, x_min:x_max]
                    if face_crop.size == 0:
                        continue  # Skip if no face detected

                    pil_face = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                    mask_result = mask_detector(pil_face)[0]['label']
                    
                    if "withmask" in mask_result.lower():
                        mask_label = "Masked"
                        color = (0, 0, 255)  # Red for masked
                        masked_detected = True
                    else:
                        mask_label = "No Mask"
                        color = (0, 255, 0)  # Green for no mask

                    # Display the mask status near the face
                    cv2.putText(frame, mask_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            return frame, counting, masked_detected

        processed_frame, people_count, masked_detected = person_counting_and_mask(frame.copy())
        
        # Fire detection
        fire_result = fire_detector(pil_image)[0]['label']
        fire_detected = "fire" in fire_result.lower()
        
        # Weapon detection
        results = weapon_model(frame_rgb, verbose=False)  # Run inference

        gun_detected = False  # Initialize flag

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())  # Get the class index
                if classNames[class_id] in classNames:  # Check if detected class is in weapons list
                    gun_detected = True
                    break
        
        # Prepare JSON data for alert
        alert_data = {
            "Fire": fire_detected,
            "Gun": gun_detected,
            "Masked": masked_detected,
            "People_Count": people_count
        }

        # Display the alert signal received from coms.py
        with alert_lock:
            external_alert = alert_signal.get("Alert", 0)

        cv2.putText(processed_frame, f"External Alert: {external_alert}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if external_alert else (0, 255, 0), 2)
        cv2.putText(processed_frame, f"People Count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Fire: {fire_result}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if fire_detected else (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Gun: {'Detected' if gun_detected else 'None'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if gun_detected else (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Final Output: {int(fire_detected or gun_detected or people_count > 1 or masked_detected)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        yield processed_frame, people_count, fire_result, gun_detected, int(fire_detected or gun_detected or people_count > 1 or masked_detected)
    
    cap.release()

# Gradio Interface
iface = gr.Interface(
    fn=process_frame,
    inputs=[],
    outputs=[
        gr.Image(label="Live Detection", streaming=True),
        gr.Number(label="People Count"),
        gr.Label(label="Fire Detection"),
        gr.Label(label="Gun Detection"),
        gr.Number(label="Final Output"),
        gr.Label(label="External Alert")  # Display external alert
    ],
    live=True,
    title="Multi-Object Detection with Live Webcam Feed"
)

iface.launch()