"""
Finds which serial port an Arduino is on by asking it who it is, so DREAM and
ARM can run from the same PC without hard-coding COM ports.

    PC:   WHO
    COM6: I am Arm      (or "I am Dream")

The sketches (arm.ino, dream_sensors.ino) answer "WHO" with that line. This
file is kept identical in both the DREAM and ARM-Robot-v01 repos.
"""

import re
import threading
import time

import serial
from serial.tools import list_ports

BOOT_WAIT_S = 2.0    # boards auto-reset when the port opens
REPLY_WAIT_S = 1.5
_IAM = re.compile(r"I am (\w+)", re.IGNORECASE)


def _probe(port, baud):
    """Returns the name the board on `port` reports, or None."""
    try:
        with serial.Serial(port, baud, timeout=0.3) as ser:
            time.sleep(BOOT_WAIT_S)
            ser.reset_input_buffer()
            ser.write(b"WHO\n")
            ser.flush()
            deadline = time.time() + REPLY_WAIT_S
            while time.time() < deadline:
                m = _IAM.search(ser.readline().decode(errors="ignore"))
                if m:
                    return m.group(1).lower()
    except (serial.SerialException, OSError):
        pass  # busy (another program has it) or not a usable port
    return None


def identify_ports(baud_rates):
    """Probes every serial port at each baud in `baud_rates`.
    Returns {port: name}."""
    found = {}
    lock = threading.Lock()

    def worker(port):
        for baud in baud_rates:  # one port at a time, so bauds don't collide
            name = _probe(port, baud)
            if name:
                with lock:
                    found[port] = name
                return

    threads = [threading.Thread(target=worker, args=(p.device,), daemon=True)
               for p in list_ports.comports()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return found


def find_board(name, baud):
    """Port of the board that answers "I am <name>" at `baud`, or None."""
    name = name.lower()
    for port, who in identify_ports([baud]).items():
        if who == name:
            return port
    return None


if __name__ == "__main__":
    # python board_id.py  ->  lists every port and who answered
    for port, who in sorted(identify_ports([9600, 115200]).items()):
        print(f"{port}: I am {who.capitalize()}")
