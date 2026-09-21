#include "sysinfo.h"

#include "common.h"

#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <icmpapi.h>
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <map>
#include <mutex>
#include <thread>

namespace {

double cpuPercent() {
    auto toU64 = [](const FILETIME& f) { return (uint64_t(f.dwHighDateTime) << 32) | f.dwLowDateTime; };
    FILETIME idle1, kern1, user1, idle2, kern2, user2;
    GetSystemTimes(&idle1, &kern1, &user1);
    dream::sleepMs(250);
    GetSystemTimes(&idle2, &kern2, &user2);
    uint64_t idle = toU64(idle2) - toU64(idle1);
    uint64_t total = (toU64(kern2) - toU64(kern1)) + (toU64(user2) - toU64(user1)); // kernel time includes idle
    return total ? 100.0 * double(total - idle) / double(total) : 0.0;
}

std::string fmt(const char* f, double a = 0, double b = 0, double c = 0) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), f, a, b, c);
    return buf;
}

bool isRandomized(const std::string& mac) { // locally-administered bit set: a phone's per-network private address
    unsigned first = 0;
    return std::sscanf(mac.c_str(), "%x", &first) == 1 && (first & 0x02);
}

std::string guessType(const std::string& mac, const std::string& hostname) {
    if (isRandomized(mac)) return "phone or tablet with randomized MAC";
    std::string h = dream::toLower(hostname);
    auto has = [&](std::initializer_list<const char*> words) {
        for (const char* w : words) if (h.find(w) != std::string::npos) return true;
        return false;
    };
    if (has({"iphone", "apple", "ipad"})) return "Apple device";
    if (has({"samsung", "android", "xiaomi", "huawei"})) return "Android device";
    if (has({"router", "gateway", "dlink", "tp-link", "asus", "netgear"})) return "router";
    if (has({"windows", "desktop", "laptop", "pc"})) return "Windows PC";
    if (has({"ubuntu", "linux", "debian", "raspi"})) return "Linux device";
    if (has({"tv", "cast", "roku", "echo", "alexa"})) return "smart TV or IoT device";
    return "unknown device";
}

std::string hostnameOf(const std::string& ip) {
    sockaddr_in sa = {};
    sa.sin_family = AF_INET;
    inet_pton(AF_INET, ip.c_str(), &sa.sin_addr);
    char host[NI_MAXHOST] = {0};
    if (getnameinfo(reinterpret_cast<sockaddr*>(&sa), sizeof(sa), host, sizeof(host), nullptr, 0, NI_NAMEREQD) == 0) return host;
    return "";
}

// This machine's LAN address: connect() on a UDP socket picks the outgoing
// interface without sending anything.
std::string myAddress() {
    SOCKET s = socket(AF_INET, SOCK_DGRAM, 0);
    if (s == INVALID_SOCKET) return "";
    sockaddr_in to = {};
    to.sin_family = AF_INET;
    to.sin_port = htons(80);
    inet_pton(AF_INET, "8.8.8.8", &to.sin_addr);
    std::string out;
    if (connect(s, reinterpret_cast<sockaddr*>(&to), sizeof(to)) == 0) {
        sockaddr_in me = {};
        int len = sizeof(me);
        if (getsockname(s, reinterpret_cast<sockaddr*>(&me), &len) == 0) {
            char buf[INET_ADDRSTRLEN] = {0};
            inet_ntop(AF_INET, &me.sin_addr, buf, sizeof(buf));
            out = buf;
        }
    }
    closesocket(s);
    return out;
}

void pingSweep(const std::string& prefix) { // prefix "192.168.0."; replies fill the ARP table
    std::atomic<int> next{1};
    auto worker = [&] {
        HANDLE icmp = IcmpCreateFile();
        if (icmp == INVALID_HANDLE_VALUE) return;
        char payload[8] = "dream";
        std::vector<char> reply(sizeof(ICMP_ECHO_REPLY) + sizeof(payload) + 8);
        for (int host; (host = next++) <= 254;) {
            in_addr addr;
            inet_pton(AF_INET, (prefix + std::to_string(host)).c_str(), &addr);
            IcmpSendEcho(icmp, addr.S_un.S_addr, payload, sizeof(payload), nullptr, reply.data(), DWORD(reply.size()), 1000);
        }
        IcmpCloseHandle(icmp);
    };
    std::vector<std::thread> threads;
    for (int i = 0; i < 64; i++) threads.emplace_back(worker);
    for (auto& t : threads) t.join();
}

} // namespace

std::string buildStatsSummary() {
    MEMORYSTATUSEX mem = {sizeof(mem)};
    GlobalMemoryStatusEx(&mem);
    double gb = 1073741824.0;
    double ramTotal = double(mem.ullTotalPhys) / gb;
    double ramUsed = ramTotal - double(mem.ullAvailPhys) / gb;

    ULARGE_INTEGER freeBytes, totalBytes, totalFree;
    GetDiskFreeSpaceExA("C:\\", &freeBytes, &totalBytes, &totalFree);
    double diskTotal = double(totalBytes.QuadPart) / gb;
    double diskUsed = diskTotal - double(totalFree.QuadPart) / gb;

    return fmt("CPU is at %.0f percent. ", cpuPercent()) +
           fmt("RAM usage is %.1f of ", ramUsed) + fmt("%.1f gigabytes, that's %.0f percent. ", ramTotal, mem.dwMemoryLoad) +
           fmt("disk is %.0f of ", diskUsed) + fmt("%.0f gigabytes used.", diskTotal);
}

std::string runNetworkScan() {
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);

    std::string me = myAddress();
    size_t dot = me.rfind('.');
    if (me.empty() || dot == std::string::npos) return "Sorry, the network scan failed.";
    std::string prefix = me.substr(0, dot + 1);
    dream::logf("Scanning %s0/24...", prefix.c_str());

    pingSweep(prefix);

    struct Device { std::string ip, mac, host, type; bool me; };
    std::vector<Device> devices;
    ULONG size = 0;
    GetIpNetTable(nullptr, &size, TRUE);
    std::vector<char> table(size);
    auto* t = reinterpret_cast<MIB_IPNETTABLE*>(table.data());
    if (GetIpNetTable(t, &size, TRUE) == NO_ERROR) {
        for (DWORD i = 0; i < t->dwNumEntries; i++) {
            const MIB_IPNETROW& row = t->table[i];
            if (row.dwType == MIB_IPNET_TYPE_INVALID || row.dwPhysAddrLen != 6) continue;
            char ip[INET_ADDRSTRLEN];
            in_addr a;
            a.S_un.S_addr = row.dwAddr;
            inet_ntop(AF_INET, &a, ip, sizeof(ip));
            std::string sip = ip;
            if (sip.rfind(prefix, 0) != 0) continue;                        // other subnets/interfaces
            if (row.bPhysAddr[0] == 0xFF || (row.bPhysAddr[0] & 1)) continue; // broadcast / multicast
            char mac[24];
            std::snprintf(mac, sizeof(mac), "%02X:%02X:%02X:%02X:%02X:%02X", row.bPhysAddr[0], row.bPhysAddr[1],
                          row.bPhysAddr[2], row.bPhysAddr[3], row.bPhysAddr[4], row.bPhysAddr[5]);
            devices.push_back({sip, mac, "", "", sip == me});
        }
    }
    auto ipKey = [](const std::string& ip) { unsigned a, b, c, d; std::sscanf(ip.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d); return d; };
    std::sort(devices.begin(), devices.end(), [&](const Device& x, const Device& y) { return ipKey(x.ip) < ipKey(y.ip); });
    devices.erase(std::unique(devices.begin(), devices.end(), [](const Device& x, const Device& y) { return x.ip == y.ip; }), devices.end());

    std::vector<std::thread> lookups; // reverse DNS can be slow: do them all at once
    for (auto& d : devices) lookups.emplace_back([&d] { d.host = hostnameOf(d.ip); d.type = guessType(d.mac, d.host); });
    for (auto& th : lookups) th.join();

    size_t count = devices.size();
    size_t phones = std::count_if(devices.begin(), devices.end(), [](const Device& d) {
        return d.type.find("phone") != std::string::npos || d.type.find("tablet") != std::string::npos;
    });

    std::string summary = "I found " + std::to_string(count) + " device" + (count != 1 ? "s" : "") + " on the network.";
    for (const auto& d : devices) {
        summary += d.me ? " " + d.ip + " is this machine."
                        : " " + d.ip + ": " + d.type + (d.host.empty() ? "" : ", hostname " + d.host) + ".";
    }
    if (summary.size() > 600) {
        summary = "I found " + std::to_string(count) + " devices on your network. " + std::to_string(phones) +
                  " appear to be phones or tablets. The rest include " + std::to_string(count - phones) + " other devices.";
    }
    return summary;
}
