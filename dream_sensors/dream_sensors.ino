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
bool lastPresenceState = false; // mmWave target-present edge, for PRESENT/ABSENT prints
bool radarReady = false; // set once radar.begin() succeeds in setup() — see there

// Alarm settings — driven by the PIR only. The mmWave sensor is reserved for
// DREAM's sleep/wake + distance sensing (PRESENT/ABSENT/RANGE below) and never
// triggers the alarm.
const unsigned long ALARM_DURATION = 2500;
const unsigned long TONE_DURATION  = 2000;

bool alarmEnabled    = true; // toggled by "ALARM ON"/"ALARM OFF" over serial; default on
bool alarmActive     = false;
unsigned long alarmStart  = 0;
bool toneActive      = false;
unsigned long toneStart   = 0;
bool lastMotionState = LOW;

// Only let a PIR edge start a new alarm if it's been at least this long
// since the last one fired, i.e. we haven't seen a human in a while.
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

// Manual RGB override — voice-controlled via "RGB <COLOR>"/"RGB OFF"/"RGB
// RAINBOW" over serial. While true, the idle rainbow animation is suppressed
// and the pixels hold whatever solid color was last requested. Note the
// alarm/buzzer effects below still take priority and will overwrite the
// pixels for their duration regardless of manualRGB.
bool manualRGB = false;

struct NamedColor { const char* name; uint8_t r, g, b; };
const NamedColor RGB_COLOR_TABLE[] = {
  {"RED",    255, 0,   0},
  {"GREEN",  0,   255, 0},
  {"BLUE",   0,   0,   255},
  {"YELLOW", 255, 255, 0},
  {"ORANGE", 255, 80,  0},
  {"PURPLE", 160, 0,   255},
  {"PINK",   255, 20,  120},
  {"CYAN",   0,   255, 255},
  {"WHITE",  255, 255, 255},
};
const int RGB_COLOR_COUNT = sizeof(RGB_COLOR_TABLE) / sizeof(RGB_COLOR_TABLE[0]);

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

// Gate for the PIR-triggered alarm. Ignores the request if the alarm has
// been switched off, we're already alarming, or we triggered too recently
// (see TRIGGER_COOLDOWN above).
void triggerAlarm() {
  unsigned long now = millis();

  if (!alarmEnabled) return;
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

  // Retry the mmWave sensor for a few seconds, then give up and move on —
  // it used to retry here forever, which meant a disconnected/unresponsive
  // radar module silently blocked loop() from ever running at all, taking
  // the PIR alarm, the RGB lights, and every serial command down with it.
  // radarReady gates the mmWave-only code in loop() below; everything else
  // now runs regardless of whether the radar ever comes up.
  unsigned long radarDeadline = millis() + 5000;
  while (!radarReady && millis() < radarDeadline) {
    radarReady = radar.begin();
    if (!radarReady) delay(200);
  }
  if (radarReady) {
    radar.setSensorMode(eSpeedMode);
    // min/max range in cm (30cm~10m), threshold is a dimensionless 0.1 unit
    radar.setDetectThres(/*min*/ 30, /*max*/ 1000, /*thres*/ 10);
  } else {
    Serial.println("RADAR_ERROR init failed — mmWave disabled, alarm/RGB still active");
  }
}

void loop() {
  // --------- Alarm: PIR edge only ---------
  int motionState = digitalRead(PIR_PIN);
  if (motionState == HIGH && lastMotionState == LOW) {
    triggerAlarm();
  }
  lastMotionState = motionState;

  // --------- DREAM sleep/wake + distance: mmWave only ---------
  // Never touches the alarm. PRESENT/ABSENT mark presence edges (DREAM
  // wakes on PRESENT while asleep); RANGE streams while a target is in
  // view so DREAM knows how far away they are. Skipped entirely if the
  // sensor never came up in setup() — see radarReady.
  if (radarReady) {
    uint8_t targetCount = radar.getTargetNumber();
    bool present = targetCount > 0;
    if (present != lastPresenceState) {
      Serial.println(present ? "PRESENT" : "ABSENT");
      lastPresenceState = present;
    }
    if (present && millis() - lastRangePrint >= RANGE_PRINT_INTERVAL) {
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
    } else if (cmd.equalsIgnoreCase("ALARM ON")) {
      alarmEnabled = true;
      Serial.println("ALARM_STATE ON");
    } else if (cmd.equalsIgnoreCase("ALARM OFF")) {
      alarmEnabled = false;
      if (alarmActive) {
        alarmActive = false;
        noTone(BUZZER_PIN);
        setRGB(0, 0, 0);
      }
      Serial.println("ALARM_STATE OFF");
    } else if (cmd.equalsIgnoreCase("RGB OFF")) {
      manualRGB = true;
      setRGB(0, 0, 0);
      Serial.println("RGB_STATE OFF");
    } else if (cmd.equalsIgnoreCase("RGB RAINBOW")) {
      manualRGB = false;
      Serial.println("RGB_STATE RAINBOW");
    } else if (cmd.startsWith("RGB ") || cmd.startsWith("rgb ")) {
      String colorName = cmd.substring(4);
      colorName.trim();
      bool matched = false;
      for (int i = 0; i < RGB_COLOR_COUNT; i++) {
        if (colorName.equalsIgnoreCase(RGB_COLOR_TABLE[i].name)) {
          manualRGB = true;
          setRGB(RGB_COLOR_TABLE[i].r, RGB_COLOR_TABLE[i].g, RGB_COLOR_TABLE[i].b);
          Serial.print("RGB_STATE ");
          Serial.println(RGB_COLOR_TABLE[i].name);
          matched = true;
          break;
        }
      }
      if (!matched) Serial.println("RGB_ERROR unknown color");
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
  if (!alarmActive && !toneActive && !manualRGB) {
    updateRainbow();
  }
}
