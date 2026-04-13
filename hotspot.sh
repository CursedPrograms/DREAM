#!/bin/bash

# ============================================================
#  ComCentre Hotspot Script
#  Chipset: Realtek RTL8192EU (D-Link DWA-131 Rev E1)
#  Interface: wlxf48ceb5115ef
# ============================================================

set -e

IFACE="wlxf48ceb5115ef"
SSID="ComCentre"
PASS="comcentre123"
HOTSPOT_IP="192.168.4.1"
DHCP_RANGE="192.168.4.10,192.168.4.50,12h"
INTERNET_IFACE="enp4s0"

HOSTAPD_CONF="/etc/hostapd/comcentre.conf"
DNSMASQ_CONF="/etc/dnsmasq.d/comcentre.conf"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║      ComCentre Hotspot Startup       ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── Kill anything using the interface ────────────────────────
echo "[1/6] Releasing interface..."
sudo systemctl stop NetworkManager 2>/dev/null || true
sudo killall wpa_supplicant 2>/dev/null || true
sudo ip link set "$IFACE" down
sleep 1

# ── Assign static IP to interface ────────────────────────────
echo "[2/6] Assigning static IP $HOTSPOT_IP to $IFACE..."
sudo ip link set "$IFACE" up
sudo ip addr flush dev "$IFACE"
sudo ip addr add "$HOTSPOT_IP/24" dev "$IFACE"

# ── Write hostapd config ──────────────────────────────────────
echo "[3/6] Writing hostapd config..."
sudo tee "$HOSTAPD_CONF" > /dev/null <<EOF
interface=$IFACE
driver=nl80211
ssid=$SSID
hw_mode=g
channel=6
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=$PASS
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP
EOF

# ── Write dnsmasq config ──────────────────────────────────────
echo "[4/6] Writing dnsmasq config..."
sudo tee "$DNSMASQ_CONF" > /dev/null <<EOF
interface=$IFACE
dhcp-range=$DHCP_RANGE
dhcp-option=3,$HOTSPOT_IP
dhcp-option=6,$HOTSPOT_IP
server=8.8.8.8
log-queries
log-dhcp
address=/#/$HOTSPOT_IP
EOF

# ── Enable IP forwarding + NAT ────────────────────────────────
echo "[5/6] Enabling IP forwarding and NAT..."
sudo sysctl -w net.ipv4.ip_forward=1 > /dev/null
sudo iptables -t nat -A POSTROUTING -o "$INTERNET_IFACE" -j MASQUERADE 2>/dev/null || true
sudo iptables -A FORWARD -i "$IFACE" -o "$INTERNET_IFACE" -j ACCEPT 2>/dev/null || true
sudo iptables -A FORWARD -i "$INTERNET_IFACE" -o "$IFACE" -m state --state RELATED,ESTABLISHED -j ACCEPT 2>/dev/null || true

# ── Start hostapd and dnsmasq ─────────────────────────────────
echo "[6/6] Starting hostapd and dnsmasq..."
sudo systemctl restart dnsmasq
sudo hostapd "$HOSTAPD_CONF" -B

echo ""
echo "╔══════════════════════════════════════╗"
echo "║       Hotspot is LIVE ✓              ║"
echo "╠══════════════════════════════════════╣"
echo "║  SSID     : ComCentre               ║"
echo "║  Password : comcentre123            ║"
echo "║  URL      : http://192.168.4.1      ║"
echo "╚══════════════════════════════════════╝"
echo ""
