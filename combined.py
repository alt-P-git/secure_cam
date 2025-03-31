import os
import cv2
import numpy as np
import gradio as gr
from transformers import pipeline
from PIL import Image
from ultralytics import YOLO

import time
import serial
import threading
from collections import deque

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

global alert
alert = False

arduino = serial.Serial('COM4', 9600)

DISTANCE_THRESHOLD = 20  # cm
GAS_THRESHOLD = 150  # Adjust based on testing
SCORE_THRESHOLD = 0.4  # Threshold for triggering an alert

# Weights for parameters
WEIGHTS = {
    "Obstruction": 0.4,
    "Smoke": 0.4,
    "Fire": 0.25,
    "Gun": 0.1,
    "Masked": 0.03,
    "People_Count": 0.02
}

alert_data = {
    "Obstruction": False,
    "Smoke": False,
    "Fire": False,
    "Gun": False,
    "Masked": False,
    "People_Count": 0
}

score_buffer = deque(maxlen=5)

def calculate_weighted_score(data):
    """Calculate the weighted score based on the input data."""
    score = 0
    for key, weight in WEIGHTS.items():
        if key == "People_Count":
            # Normalize People_Count to a value between 0 and 1
            normalized_people_count = min(data[key] / 10, 1)  # Assume max 10 people
            score += normalized_people_count * weight
        else:
            score += (1 if data[key] else 0) * weight
    return score

def stabilize_score(new_score):
    """Stabilize the score using a moving average."""
    score_buffer.append(new_score)
    return sum(score_buffer) / len(score_buffer)

def process_frame():
    global alert
    global arduino
    global alert_data

    cap = cv2.VideoCapture(2)
    
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

            for i in range(detections.shape[2]):
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

        # # Fetch Arduino Data
        # try:
        #     arduino_data = arduino.readline().decode('utf-8').strip()
        #     distance, gas_value = map(int, arduino_data.split(','))
        #     obstruction = distance < DISTANCE_THRESHOLD
        #     smoke_detected = gas_value > GAS_THRESHOLD
        # except Exception as e:
        #     print(f"Error reading Arduino data: {e}")
        #     obstruction, smoke_detected = False, False

        # Update alert_data based on detection results
        if fire_detected:
            alert_data["Fire"] = True
        else:
            alert_data["Fire"] = False
        if gun_detected:
            alert_data["Gun"] = True
        else:
            alert_data["Gun"] = False
        if masked_detected:
            alert_data["Masked"] = True
        else:
            alert_data["Masked"] = False
        if people_count > 0:
            alert_data["People_Count"] = people_count
        else:
            alert_data["People_Count"] = 0


        
        cv2.putText(processed_frame, f"People Count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Fire: {fire_result}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if fire_detected else (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Gun: {'Detected' if gun_detected else 'None'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if gun_detected else (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Final Output: {int(fire_detected or gun_detected or people_count > 1 or masked_detected)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        alert_string = "Alert" if alert else "No Alert"
        gun_string = "Detected" if gun_detected else "None"
        
        yield processed_frame, people_count, fire_result, gun_string, int(fire_detected or gun_detected or people_count > 1 or masked_detected), alert_string
    
    cap.release()

def read_arduino_data():
    global alert_data, alert
    while True:
        try:
            arduino_data = arduino.readline().decode('utf-8').strip()
            if arduino_data:
                distance, gas_value = map(int, arduino_data.split(','))
                alert_data["Obstruction"] = distance < DISTANCE_THRESHOLD
                alert_data["Smoke"] = gas_value > GAS_THRESHOLD

            raw_score = calculate_weighted_score(alert_data)
            stabilized_score = stabilize_score(raw_score)
        
            if alert == False:
                arduino.write(b'0')
                alert = stabilized_score >= SCORE_THRESHOLD
            if alert:
                arduino.write(b'1')
            print(alert_data)
            print(stabilized_score)
            print(f"Alert Status: {alert}")
        except Exception as e:
            print(f"Error reading Arduino data: {e}")
            alert_data["Obstruction"] = False
            alert_data["Smoke"] = False
        
        # time.sleep(0.5)

# Start Arduino thread
arduino_thread = threading.Thread(target=read_arduino_data)
arduino_thread.daemon = True
arduino_thread.start()

def dismiss_alert():
    global alert
    alert = False
    return "No Alert"

def update_dismiss_button(alert_status):
    if alert_status == "Alert":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

# Gradio Interface
with gr.Blocks() as UI:
    gr.Markdown("# Multi-Object Detection with Live Webcam Feed")
    
    with gr.Row():
        video_output = gr.Image(label="Live Detection")  # Removed streaming=True here
        with gr.Column():
            people_count_output = gr.Number(label="People Count")
            fire_output = gr.Label(label="Fire Detection")
            gun_output = gr.Label(label="Gun Detection")
            final_output_display = gr.Number(label="Final Output")
            alert_display = gr.Label(label="Alert Status", value="No Alert")
            dismiss_btn = gr.Button("Dismiss Alert", visible=False)
    
    # We use an Interface with live=True for streaming the generator outputs
    stream_interface = gr.Interface(
        fn=process_frame,
        inputs=[],
        outputs=[
            video_output,
            people_count_output,
            fire_output,
            gun_output,
            final_output_display,
            alert_display
        ],
        live=True  # This ensures the process_frame generator is polled continuously
    )
    
    stream_interface.render()
    
    # When alert_display changes, show/hide the dismiss button
    alert_display.change(fn=update_dismiss_button, inputs=[alert_display], outputs=[dismiss_btn])
    # Clicking dismiss updates the label to "No Alert"
    dismiss_btn.click(fn=dismiss_alert, inputs=[], outputs=[alert_display])

UI.launch()