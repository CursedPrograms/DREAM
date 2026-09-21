#include "sensors.h"

#include "common.h"
#include "state.h"

#include <windows.h>
#include <setupapi.h>

#include <algorithm>
#include <mutex>
#include <thread>

namespace {

constexpr int SENSOR_BAUD = 9600;

// "Ports" device setup class GUID, written out so this doesn't need devguid.h.
const GUID GUID_PORTS_CLASS = {0x4D36E978, 0xE325, 0x11CE, {0xBF, 0xC1, 0x08, 0x00, 0x2B, 0xE1, 0x03, 0x18}};

const char* ARDUINO_HINTS[] = {"arduino", "ch340", "usb-serial", "usb serial", "cp210", "ftdi"};

class Serial {
public:
    ~Serial() { close(); }

    bool open(const std::string& device, int baud) {
        close();
        HANDLE h = CreateFileA(("\\\\.\\" + device).c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
        if (h == INVALID_HANDLE_VALUE) return false;
        DCB dcb = {};
        dcb.DCBlength = sizeof(DCB);
        if (!GetCommState(h, &dcb)) { CloseHandle(h); return false; }
        dcb.BaudRate = DWORD(baud);
        dcb.ByteSize = 8;
        dcb.Parity = NOPARITY;
        dcb.StopBits = ONESTOPBIT;
        dcb.fBinary = TRUE;
        dcb.fParity = FALSE;
        if (!SetCommState(h, &dcb)) { CloseHandle(h); return false; }
        COMMTIMEOUTS t = {};
        t.ReadIntervalTimeout = MAXDWORD; // reads return at once with whatever has arrived
        t.WriteTotalTimeoutConstant = 1000;
        SetCommTimeouts(h, &t);
        h_ = h;
        buf_.clear();
        return true;
    }

    void close() {
        if (h_ != INVALID_HANDLE_VALUE) { CloseHandle(h_); h_ = INVALID_HANDLE_VALUE; }
    }

    bool isOpen() const { return h_ != INVALID_HANDLE_VALUE; }

    void flushInput() {
        if (isOpen()) PurgeComm(h_, PURGE_RXCLEAR);
        buf_.clear();
    }

    bool writeLine(const std::string& line) {
        if (!isOpen()) return false;
        std::string out = line + "\n";
        DWORD written = 0;
        return WriteFile(h_, out.data(), DWORD(out.size()), &written, nullptr) && written == out.size();
    }

    // One line (without the newline), or "" if none arrives within timeoutMs.
    // `broken` is set if the device has gone away.
    std::string readLine(int timeoutMs, bool& broken) {
        double deadline = dream::nowMs() + timeoutMs;
        for (;;) {
            size_t nl = buf_.find('\n');
            if (nl != std::string::npos) {
                std::string line = dream::trim(buf_.substr(0, nl));
                buf_.erase(0, nl + 1);
                if (!line.empty()) return line;
                continue;
            }
            char chunk[256];
            DWORD got = 0;
            if (!ReadFile(h_, chunk, sizeof(chunk), &got, nullptr)) { broken = true; return ""; }
            if (got > 0) { buf_.append(chunk, got); continue; }
            if (dream::nowMs() >= deadline) return "";
            dream::sleepMs(10);
        }
    }

private:
    HANDLE h_ = INVALID_HANDLE_VALUE;
    std::string buf_;
};

bool looksLikeArduino(const std::string& description) {
    std::string lower = dream::toLower(description);
    for (const char* hint : ARDUINO_HINTS) {
        if (lower.find(hint) != std::string::npos) return true;
    }
    return false;
}

// True if the board on `device` answers "WHO" with "I am Dream".
bool answersIAmDream(const std::string& device) {
    Serial s;
    if (!s.open(device, SENSOR_BAUD)) return false; // busy (another program has it) or unusable
    dream::sleepMs(2000);                           // the board resets when the port opens
    s.flushInput();
    s.writeLine("WHO");
    double deadline = dream::nowMs() + 1500;
    bool broken = false;
    while (dream::nowMs() < deadline && !broken) {
        std::string line = s.readLine(300, broken);
        if (dream::toLower(line).find("i am dream") != std::string::npos) return true;
    }
    return false;
}

std::mutex g_writeMutex;
Serial g_serial;
std::function<void(const std::string&)> g_onLine;

void sensorLoop() {
    bool warned = false;
    while (S().running) {
        std::string port = findDreamBoard();
        if (port.empty()) {
            if (!warned) dream::logf("Sensor board not found - retrying every 5s (alarm, lights and presence are off)");
            warned = true;
            for (int i = 0; i < 50 && S().running; i++) dream::sleepMs(100);
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(g_writeMutex);
            if (!g_serial.open(port, SENSOR_BAUD)) {
                if (!warned) dream::logf("Sensor board on %s is busy or unavailable - retrying", port.c_str());
                warned = true;
                continue;
            }
        }
        warned = false;
        dream::logf("Sensor board connected on %s", port.c_str());
        dream::sleepMs(2000); // let it finish booting after the reset

        bool broken = false;
        while (S().running && !broken) {
            std::string line = g_serial.readLine(500, broken);
            if (!line.empty() && g_onLine) g_onLine(line);
        }
        {
            std::lock_guard<std::mutex> lock(g_writeMutex);
            g_serial.close();
        }
        if (S().running) dream::logf("Sensor board disconnected - reconnecting");
        dream::sleepMs(2000);
    }
}

} // namespace

std::vector<std::pair<std::string, std::string>> listComPorts() {
    std::vector<std::pair<std::string, std::string>> ports;
    HDEVINFO devInfo = SetupDiGetClassDevsA(&GUID_PORTS_CLASS, nullptr, nullptr, DIGCF_PRESENT);
    if (devInfo == INVALID_HANDLE_VALUE) return ports;

    SP_DEVINFO_DATA data;
    data.cbSize = sizeof(SP_DEVINFO_DATA);
    for (DWORD i = 0; SetupDiEnumDeviceInfo(devInfo, i, &data); i++) {
        char name[256] = {0};
        if (!SetupDiGetDeviceRegistryPropertyA(devInfo, &data, SPDRP_FRIENDLYNAME, nullptr, reinterpret_cast<PBYTE>(name),
                                               sizeof(name) - 1, nullptr)) {
            SetupDiGetDeviceRegistryPropertyA(devInfo, &data, SPDRP_DEVICEDESC, nullptr, reinterpret_cast<PBYTE>(name),
                                              sizeof(name) - 1, nullptr);
        }
        std::string friendly(name);
        size_t open = friendly.rfind('('), close = friendly.rfind(')');
        if (open == std::string::npos || close == std::string::npos || close <= open) continue;
        std::string com = friendly.substr(open + 1, close - open - 1);
        if (com.size() > 3 && dream::toLower(com.substr(0, 3)) == "com") ports.push_back({com, friendly});
    }
    SetupDiDestroyDeviceInfoList(devInfo);
    return ports;
}

std::string findDreamBoard() {
    for (const auto& [port, description] : listComPorts()) {
        if (looksLikeArduino(description) && answersIAmDream(port)) return port;
    }
    return "";
}

void sensorsStart(std::function<void(const std::string&)> onLine) {
    g_onLine = std::move(onLine);
    std::thread(sensorLoop).detach();
}

bool sendSensorCommand(const std::string& cmd) {
    std::lock_guard<std::mutex> lock(g_writeMutex);
    return g_serial.isOpen() && g_serial.writeLine(cmd);
}
