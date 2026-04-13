#!/usr/bin/env python3

import os
import io
import base64
import tempfile
import subprocess
import threading
import requests
import socket
import json
from flask import Flask, request, jsonify, render_template, render_template_string
from zeroconf import ServiceInfo, Zeroconf, ServiceBrowser
import whisper

# ── Config & Directories ──────────────────────────────────────
COMCENTRE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR    = os.path.join(COMCENTRE_DIR, "voices")
VOICE_MODEL   = os.path.join(VOICES_DIR, "en_US-amy-high.onnx")
OLLAMA_URL    = "http://localhost:11434/api/generate"
VISION_MODEL  = "llava:13b"
CHAT_MODEL    = "phi3:mini"

# Network Discovery Config
THIS_NAME = "COMMAND_CENTRE"
THIS_PORT = 5009
TYPE = "_flask-link._tcp.local."
found_servers = {}

# Ensure we have the config loaded
CONFIG_PATH = os.path.join(COMCENTRE_DIR, "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    SYSTEM_PROMPT = cfg.get("Config", {}).get("DREAM", {}).get("SystemPrompt", "You are a helpful assistant.")
else:
    SYSTEM_PROMPT = "You are a helpful assistant."

app = Flask(__name__, template_folder="templates")

# ── Discovery Logic ───────────────────────────────────────────
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

class MyListener:
    def remove_service(self, zeroconf, type, name):
        short_name = name.split('.')[0]
        if short_name in found_servers:
            del found_servers[short_name]
    def add_service(self, zeroconf, type, name):
        self.update_service(zeroconf, type, name)
    def update_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info:
            addresses = [socket.inet_ntoa(addr) for addr in info.addresses]
            if addresses:
                short_name = name.split('.')[0]
                if short_name != THIS_NAME:
                    found_servers[short_name] = f"http://{addresses[0]}:{info.port}"

# Initialize Discovery
my_ip = get_ip()
zeroconf = Zeroconf(interfaces=[my_ip])
info = ServiceInfo(
    TYPE, f"{THIS_NAME}.{TYPE}",
    addresses=[socket.inet_aton(my_ip)],
    port=THIS_PORT,
    properties={'version': '1.0'}
)
zeroconf.register_service(info)
browser = ServiceBrowser(zeroconf, TYPE, MyListener())

# ── AI Models ─────────────────────────────────────────────────
print("[ComCentre] Loading Whisper...")
whisper_model = whisper.load_model("large")
print("[ComCentre] Whisper ready.")

conversation_history = []
history_lock = threading.Lock()

# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    # You can return your template, or a status view
    return render_template("index.html")

@app.route("/network")
def network_status():
    """Quick view of who is online"""
    nodes = ""
    for name, url in found_servers.items():
        nodes += f"<li><b>{name}</b>: {url}</li>"
    return render_template_string(f"<h1>Network Nodes</h1><ul>{nodes if nodes else 'Scanning...'}</ul><p>My IP: {my_ip}</p>")

@app.route("/ping")
def ping():
    return f"{THIS_NAME} alive"

@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    audio_file = request.files["audio"]
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_file.save(tmp.name)
    tmp.close()
    try:
        result = whisper_model.transcribe(tmp.name, language="en", fp16=False)
        return jsonify({"text": result["text"].strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp.name): os.unlink(tmp.name)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message"}), 400
    user_message = data["message"]
    with history_lock:
        conversation_history.append({"role": "user", "content": user_message})
        full_prompt = f"System: {SYSTEM_PROMPT}\n\n"
        for msg in conversation_history[-10:]:
            role = "You" if msg["role"] == "assistant" else "Human"
            full_prompt += f"{role}: {msg['content']}\n"
        full_prompt += "You:"

    payload = {"model": CHAT_MODEL, "prompt": full_prompt, "stream": False}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response = r.json().get("response", "").strip()
        if "<think>" in response:
            end = response.find("</think>")
            if end != -1: response = response[end + 8:].strip()
        with history_lock:
            conversation_history.append({"role": "assistant", "content": response})
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/speak", methods=["POST"])
def speak():
    data = request.get_json()
    text = data.get("text", "")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        cmd = ["piper", "--model", VOICE_MODEL, "--output_file", tmp.name]
        proc = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True)
        with open(tmp.name, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        return jsonify({"audio": audio_b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp.name): os.unlink(tmp.name)

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=THIS_PORT, debug=False, threaded=True)
    finally:
        print("Cleaning up Zeroconf...")
        zeroconf.unregister_service(info)
        zeroconf.close()