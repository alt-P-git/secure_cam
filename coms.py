import socket
import serial
import time
import json
import threading
from collections import deque

# Connect to Arduino
arduino = serial.Serial('COM5', 9600)  # Change to your port

# Threshold values
DISTANCE_THRESHOLD = 20  # cm
GAS_THRESHOLD = 400  # Adjust based on testing
SCORE_THRESHOLD = 0.7  # Threshold for triggering an alert
DELAY_BETWEEN_REQUESTS = 2  # seconds
last_sent_time = 0

# Weights for parameters
WEIGHTS = {
    "Obstruction": 0.4,
    "Smoke": 0.4,
    "Fire": 0.1,
    "Gun": 0.05,
    "Masked": 0.03,
    "People_Count": 0.02
}

# Shared variable for alert data
alert_data = {
    "Obstruction": False,
    "Smoke": False,
    "Fire": False,
    "Gun": False,
    "Masked": False,
    "People_Count": 0
}

# Lock for thread-safe access to alert_data
data_lock = threading.Lock()

# Moving average buffer
score_buffer = deque(maxlen=5)  # Keep the last 5 scores

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

def read_from_arduino():
    """Read data from Arduino and update alert_data."""
    global last_sent_time
    while True:
        try:
            # Read data from Arduino
            data = arduino.readline().decode('utf-8').strip()
            if data:
                distance, gas_value = map(int, data.split(','))
                print(f"Distance: {distance} cm, Gas Value: {gas_value}")

                # Prepare local alert data
                local_alert_data = {
                    "Obstruction": distance < DISTANCE_THRESHOLD,
                    "Smoke": gas_value > GAS_THRESHOLD
                }

                # Update shared alert_data with a lock
                with data_lock:
                    alert_data.update(local_alert_data)

                # Calculate weighted score
                raw_score = calculate_weighted_score(alert_data)

                # Stabilize the score using a moving average
                stabilized_score = stabilize_score(raw_score)

                print("Stabilized Score: ", stabilized_score)
                # Check if the stabilized score exceeds the threshold
                if stabilized_score > SCORE_THRESHOLD:
                    alert_signal = {"Alert": 1}
                    arduino.write(b'1')  # Turn buzzer ON
                else:
                    alert_signal = {"Alert": 0}
                    arduino.write(b'0')  # Turn buzzer OFF

                # Send JSON data to combined.py server if conditions are met
                current_time = time.time()
                if current_time - last_sent_time > DELAY_BETWEEN_REQUESTS:
                    send_to_combined_server(alert_signal)
                    last_sent_time = current_time

        except Exception as e:
            print(f"Error reading from Arduino: {e}")

def send_to_combined_server(data, host='127.0.0.1', port=65432):
    """Send JSON data to the combined.py server."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((host, port))
            client_socket.sendall(json.dumps(data).encode('utf-8'))
    except ConnectionRefusedError:
        print("Failed to connect to the combined.py server. Is it running?")

def start_socket_server(host='127.0.0.1', port=65432):
    """Start the server to listen for alerts."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        print(f"Server started. Listening on {host}:{port}...")
        
        while True:
            conn, addr = server_socket.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    print(f"ALERT: {data.decode('utf-8')}")  # Print the received alert

if __name__ == "__main__":
    # Start the Arduino reading thread
    arduino_thread = threading.Thread(target=read_from_arduino, daemon=True)
    arduino_thread.start()

    # Start the socket server
    start_socket_server()
