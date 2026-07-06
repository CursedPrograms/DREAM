/*
 * DFRobot C4001 mmWave Presence Sensor (SEN0609, 25m, UART)
 * ----------------------------------------------------------
 * Requires the DFRobot_C4001 library (Arduino IDE -> Manage Libraries -> "DFRobot_C4001")
 *
 * Wiring (Arduino Uno/Nano):
 *   Sensor TX  -> Arduino pin 2  (our RX)
 *   Sensor RX  -> Arduino pin 3  (our TX)
 *   OUT signal -> Arduino pin 6  (HIGH when presence detected)
 *   VIN -> 5V, GND -> GND
 *
 * Sensor UART runs at 9600 baud (fixed).
 */

#include "DFRobot_C4001.h"

const uint8_t RX_PIN  = 2;   // Arduino receives here (sensor TX)
const uint8_t TX_PIN  = 3;   // Arduino transmits here (sensor RX)
const uint8_t OUT_PIN = 6;   // HIGH when presence detected

SoftwareSerial mySerial(RX_PIN, TX_PIN);
DFRobot_C4001_UART radar(&mySerial, 9600);

void setup() {
  pinMode(OUT_PIN, OUTPUT);
  digitalWrite(OUT_PIN, LOW);

  Serial.begin(115200);   // debug output to PC

  while (!radar.begin()) {
    Serial.println(F("No device found! Check wiring (sensor TX->2, RX->3)"));
    delay(1000);
  }
  Serial.println(F("C4001 connected!"));

  // Presence-detection mode
  radar.setSensorMode(eExitMode);

  // Detection range in cm: min 30cm, max/trigger 10m (adjust up to 2500 = 25m)
  radar.setDetectionRange(/*min*/30, /*max*/1000, /*trig*/1000);

  // Sensitivity 0-9 (higher = more sensitive)
  radar.setTrigSensitivity(1);   // sensitivity to trigger detection
  radar.setKeepSensitivity(2);   // sensitivity to hold detection

  // Timing: trigger delay in 0.01s units, keep timeout in 0.5s units
  // trig=100 -> 1s before triggering, keep=4 -> holds "present" for 2s after last motion
  radar.setDelay(/*trig*/100, /*keep*/4);

  Serial.println(F("Configured. Watching for presence..."));
}

void loop() {
  bool present = radar.motionDetection();

  digitalWrite(OUT_PIN, present ? HIGH : LOW);

  if (present) {
    Serial.println(F("Presence detected -> pin 6 HIGH"));
  }

  delay(100);
}
