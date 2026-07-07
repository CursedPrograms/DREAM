#include <Adafruit_NeoPixel.h>
#include <SoftwareSerial.h>
#include "DFRobot_C4001.h"

// Pin definitions
const int PIR_PIN    = 4;
const int BUZZER_PIN = 3;
const int RGB_PIN    = 6;
const int NUM_PIXELS = 40;

Adafruit_NeoPixel rgb = Adafruit_NeoPixel(NUM_PIXELS, RGB_PIN, NEO_GRB + NEO_KHZ800);

// mmWave presence + ranging sensor (DFRobot C4001 / SEN0609), run in
// eSpeedMode so we get distance/speed over UART instead of just a bare
// presence pin. Sensor TX -> pin 9, Sensor RX -> pin 10.
const uint8_t MMWAVE_RX_PIN = 9;
const uint8_t MMWAVE_TX_PIN = 10;
SoftwareSerial mmwaveSerial(MMWAVE_RX_PIN, MMWAVE_TX_PIN);
DFRobot_C4001_UART radar(&mmwaveSerial, 9600);

unsigned long lastRangePrint = 0;
const unsigned long RANGE_PRINT_INTERVAL = 500; // ms between "RANGE" prints

// Alarm settings
const unsigned long ALARM_DURATION = 2500;
const unsigned long TONE_DURATION  = 2000;

bool alarmActive     = false;
unsigned long alarmStart  = 0;
bool toneActive      = false;
unsigned long toneStart   = 0;
bool lastMotionState = LOW;

// The mmWave sensor reports "present" continuously the whole time someone
// is in the room (not just a single edge like the PIR), so it can't be
// allowed to restart the alarm every loop. Only let a trigger (from either
// sensor) start a new alarm if it's been at least this long since the last
// one fired, i.e. we haven't seen a human in a while.
const unsigned long TRIGGER_COOLDOWN = 300000UL; // 5 minutes
unsigned long lastTriggerTime = 0;
bool haveTriggeredOnce = false;

// PIRs commonly glitch HIGH for the first 10-60s after power-up while they
// calibrate. Without this, that glitch fires the alarm at boot and burns
// the whole TRIGGER_COOLDOWN before anyone's actually walked in.
const unsigned long PIR_WARMUP_MS = 30000; // 30s
unsigned long bootTime = 0;

// Alarm pulsing
bool buzzerOn = false;
unsigned long lastPulse = 0;
const unsigned long PULSE_INTERVAL = 200;

// Rainbow state
uint16_t rainbowHue = 0;
unsigned long lastRainbow = 0;
const unsigned long RAINBOW_INTERVAL = 20; // ms between rainbow steps (lower = faster)

void setRGB(uint8_t r, uint8_t g, uint8_t b) {
  for (int i = 0; i < NUM_PIXELS; i++) rgb.setPixelColor(i, rgb.Color(r, g, b));
  rgb.show();
}

void updateRainbow() {
  if (millis() - lastRainbow < RAINBOW_INTERVAL) return;
  lastRainbow = millis();

  // Spread the hue across all pixels for a flowing wave effect
  for (int i = 0; i < NUM_PIXELS; i++) {
    uint16_t pixelHue = rainbowHue + (i * 65536L / NUM_PIXELS);
    rgb.setPixelColor(i, rgb.gamma32(rgb.ColorHSV(pixelHue)));
  }
  rgb.show();

  rainbowHue += 256; // increment; wraps naturally at 65536
}

// Shared gate for anything that wants to start the alarm (PIR edge or
// mmWave presence). Ignores the request if we're already alarming, or if
// we triggered too recently (see TRIGGER_COOLDOWN above).
void triggerAlarm() {
  unsigned long now = millis();

  if (now - bootTime < PIR_WARMUP_MS) return; // let the PIR settle first
  if (alarmActive) return;
  if (haveTriggeredOnce && (now - lastTriggerTime < TRIGGER_COOLDOWN)) return;

  Serial.println("MOTION");
  alarmActive = true;
  alarmStart  = now;
  buzzerOn    = false;
  lastPulse   = now;
  setRGB(255, 0, 0);

  lastTriggerTime   = now;
  haveTriggeredOnce = true;
}

void setup() {
  pinMode(PIR_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  Serial.begin(9600);

  bootTime = millis();
  lastMotionState = digitalRead(PIR_PIN); // don't treat the boot state as an edge

  rgb.begin();
  rgb.setBrightness(100);
  setRGB(0, 0, 0);

  while (!radar.begin()) {
    delay(1000); // keep retrying; don't block Python's serial link on this
  }

  radar.setSensorMode(eSpeedMode);
  // min/max range in cm (30cm~10m), threshold is a dimensionless 0.1 unit
  radar.setDetectThres(/*min*/ 30, /*max*/ 1000, /*thres*/ 10);
}

void loop() {
  // --------- Motion detection (PIR edge + mmWave presence) ---------
  int motionState = digitalRead(PIR_PIN);
  if (motionState == HIGH && lastMotionState == LOW) {
    triggerAlarm();
  }
  lastMotionState = motionState;

  uint8_t targetCount = radar.getTargetNumber();
  if (targetCount > 0) {
    triggerAlarm();

    if (millis() - lastRangePrint >= RANGE_PRINT_INTERVAL) {
      lastRangePrint = millis();
      Serial.print("RANGE ");
      Serial.print(radar.getTargetRange());
      Serial.println(" m");
    }
  }

  // --------- Motion alarm (pulsing) ---------
  if (alarmActive) {
    if (millis() - alarmStart < ALARM_DURATION) {
      if (millis() - lastPulse >= PULSE_INTERVAL) {
        lastPulse = millis();
        buzzerOn  = !buzzerOn;
        if (buzzerOn) {
          tone(BUZZER_PIN, 1000);
          setRGB(255, 0, 0);
        } else {
          noTone(BUZZER_PIN);
          setRGB(0, 0, 0);
        }
      }
    } else {
      alarmActive = false;
      noTone(BUZZER_PIN);
      setRGB(0, 255, 0); // green = all clear
    }
  }

  // --------- Serial command ---------
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.equalsIgnoreCase("BUZZER")) {
      toneActive = true;
      toneStart  = millis();
      tone(BUZZER_PIN, 1500);
      setRGB(255, 165, 0);
    }
  }

  // --------- Manual tone (2s) ---------
  if (toneActive) {
    if (millis() - toneStart >= TONE_DURATION) {
      toneActive = false;
      noTone(BUZZER_PIN);
      setRGB(0, 0, 0);
    }
  }

  // --------- Idle rainbow ---------
  if (!alarmActive && !toneActive) {
    updateRainbow();
  }
}
