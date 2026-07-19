#!/usr/bin/env python3

import serial

ser = serial.Serial('/dev/ttyUSB0', 9600)

while True:
    line = ser.readline().decode().strip()
    
    if line == "MOTION":
        print("Motion detected")
