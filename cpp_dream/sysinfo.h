// sysinfo.h - system stats and the network scan ("check the wifi").
#pragma once

#include <string>

// Spoken summary: "CPU is at 12 percent. RAM usage is ...".
std::string buildStatsSummary();

// Pings the local /24, reads the ARP table, and returns a spoken summary of
// the devices found. Takes several seconds.
std::string runNetworkScan();
