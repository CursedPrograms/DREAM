#!/bin/bash

# ============================================================
#  ComCentre Hotspot Stop Script
# ============================================================

IFACE="wlxf48ceb5115ef"
INTERNET_IFACE="enp4s0"

echo ""
echo "[*] Stopping ComCentre hotspot..."

sudo killall hostapd 2>/dev/null || true
sudo systemctl stop dnsmasq 2>/dev/null || true
sudo ip addr flush dev "$IFACE"
sudo ip link set "$IFACE" down

# Remove iptables rules
sudo iptables -t nat -D POSTROUTING -o "$INTERNET_IFACE" -j MASQUERADE 2>/dev/null || true
sudo iptables -D FORWARD -i "$IFACE" -o "$INTERNET_IFACE" -j ACCEPT 2>/dev/null || true
sudo iptables -D FORWARD -i "$INTERNET_IFACE" -o "$IFACE" -m state --state RELATED,ESTABLISHED -j ACCEPT 2>/dev/null || true

# Restart NetworkManager so normal wifi works again
sudo systemctl start NetworkManager

echo "[✓] Hotspot stopped. NetworkManager restored."
echo ""
