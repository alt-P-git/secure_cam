import socket
import serial
import time
import json
import threading

# Connect to Arduino
arduino = serial.Serial('COM5', 9600)  # Change to your port

# Threshold values
DISTANCE_THRESHOLD = 20  # cm
GAS_THRESHOLD = 400  # Adjust based on testing

# Delay to prevent spamming the server
DELAY_BETWEEN_REQUESTS = 2  # seconds
last_sent_time = 0

# Shared variable for alert data
alert_data = {
    "Obstruction": False,
    "Smoke": False
}

# Lock for thread-safe access to alert_data
data_lock = threading.Lock()

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

                # Prepare JSON data
                local_alert_data = {
                    "Obstruction": False,
                    "Smoke": False
                }

                # Check conditions
                if distance < DISTANCE_THRESHOLD:
                    local_alert_data["Obstruction"] = True
                if gas_value > GAS_THRESHOLD:
                    local_alert_data["Smoke"] = True

                # Update shared alert_data with a lock
                with data_lock:
                    alert_data.update(local_alert_data)

                # Send JSON data to combined.py server if conditions are met
                current_time = time.time()
                if current_time - last_sent_time > DELAY_BETWEEN_REQUESTS:
                    send_to_combined_server(local_alert_data)
                    last_sent_time = current_time

                # Send commands to Arduino for buzzer control
                if local_alert_data["Obstruction"] or local_alert_data["Smoke"]:
                    arduino.write(b'1')  # Turn buzzer ON
                else:
                    arduino.write(b'0')  # Turn buzzer OFF

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
