// Pin definitions
const int PIR_PIN = 2;
const int BUZZER_PIN = 3;

// Motion alarm settings
const unsigned long ALARM_DURATION = 30000; // 30 seconds
bool alarmActive = false;
unsigned long alarmStart = 0;

// Buzzer tone command settings
bool toneActive = false;
unsigned long toneStart = 0;
const unsigned long TONE_DURATION = 2000; // 2 seconds

bool lastMotionState = LOW;

void setup()
{
  pinMode(PIR_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  Serial.begin(9600);
}

void loop()
{
  // --------- Motion detection ---------
  int motionState = digitalRead(PIR_PIN);

  if (motionState == HIGH && lastMotionState == LOW)
  {
    Serial.println("MOTION"); // Send to Python
    alarmActive = true;
    alarmStart = millis();
  }

  lastMotionState = motionState;

  // --------- Handle motion alarm (30s) ---------
  if (alarmActive)
  {
    if (millis() - alarmStart < ALARM_DURATION)
    {
      // simple pulsing alarm
      if ((millis() / 200) % 2 == 0)
      {
        digitalWrite(BUZZER_PIN, HIGH);
      }
      else
      {
        digitalWrite(BUZZER_PIN, LOW);
      }
    }
    else
    {
      alarmActive = false;
      digitalWrite(BUZZER_PIN, LOW);
    }
  }

  // --------- Handle serial command ---------
  if (Serial.available() > 0)
  {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim(); // remove whitespace/newlines

    if (cmd.equalsIgnoreCase("BUZZER"))
    {
      toneActive = true;
      toneStart = millis();
    }
  }

  // --------- Play 2s tone if command received ---------
  if (toneActive)
  {
    if (millis() - toneStart < TONE_DURATION)
    {
      digitalWrite(BUZZER_PIN, HIGH);
    }
    else
    {
      toneActive = false;
      digitalWrite(BUZZER_PIN, LOW);
    }
  }
}
