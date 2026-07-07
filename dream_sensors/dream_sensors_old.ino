#include <Adafruit_NeoPixel.h>

// Pin definitions
const int PIR_PIN    = 4;
const int BUZZER_PIN = 3;
const int RGB_PIN    = 6;
const int NUM_PIXELS = 40;

Adafruit_NeoPixel rgb = Adafruit_NeoPixel(NUM_PIXELS, RGB_PIN, NEO_GRB + NEO_KHZ800);

// Alarm settings
const unsigned long ALARM_DURATION = 2500;
const unsigned long TONE_DURATION  = 2000;

bool alarmActive     = false;
unsigned long alarmStart  = 0;
bool toneActive      = false;
unsigned long toneStart   = 0;
bool lastMotionState = LOW;

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

void setup() {
  pinMode(PIR_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  Serial.begin(9600);

  rgb.begin();
  rgb.setBrightness(100);
  setRGB(0, 0, 0);
}

void loop() {
  // --------- Motion detection ---------
  int motionState = digitalRead(PIR_PIN);
  if (motionState == HIGH && lastMotionState == LOW) {
    Serial.println("MOTION");
    alarmActive  = true;
    alarmStart   = millis();
    buzzerOn     = false;
    lastPulse    = millis();
    setRGB(255, 0, 0);
  }
  lastMotionState = motionState;

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
