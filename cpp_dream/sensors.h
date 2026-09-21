// sensors.h - the sensor board (dream_sensors.ino): finds it by asking each
// serial port "WHO" ("I am Dream"), reads PRESENT/ABSENT/RANGE lines, and
// sends it alarm and light commands.
#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

// Serial ports Windows knows about: {"COM6", "USB-SERIAL CH340 (COM6)"}.
std::vector<std::pair<std::string, std::string>> listComPorts();

// The port whose board answers "I am Dream", or "" if none does.
std::string findDreamBoard();

// Starts the background thread that keeps the board connected (reconnecting if
// it drops) and passes every line it prints to onLine.
void sensorsStart(std::function<void(const std::string&)> onLine);

// Writes one line to the board ("ALARM ON", "RGB RED"...). False if not connected.
bool sendSensorCommand(const std::string& cmd);
