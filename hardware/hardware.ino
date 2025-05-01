#define TRIG_PIN 9
#define ECHO_PIN 10
#define GAS_SENSOR_PIN A0
#define BUZZER_PIN 8

long duration;
int distance;
int gasValue;

void setup() {
  Serial.begin(9600);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(GAS_SENSOR_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
}

void loop() {
  // Ultrasonic Sensor Reading
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  duration = pulseIn(ECHO_PIN, HIGH);
  distance = duration * 0.034 / 2;

  // MQ2 Sensor Reading
  gasValue = analogRead(GAS_SENSOR_PIN);

  // Send data to Python script
  Serial.print(distance);
  Serial.print(",");
  Serial.println(gasValue);

  // Check if alarm should be on/off based on Python response
  if (Serial.available()) {
    char state = Serial.read();
    if (state == '1') {
      digitalWrite(BUZZER_PIN, HIGH);
      delay(400);
      digitalWrite(BUZZER_PIN, LOW);
    } else {
      digitalWrite(BUZZER_PIN, LOW);
      delay(400);
    }
  }
}