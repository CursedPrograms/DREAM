#include "http.h"

#include <windows.h>
#include <winhttp.h>

namespace {

std::wstring widen(const std::string& s) {
    if (s.empty()) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
    std::wstring w(size_t(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), w.data(), n);
    return w;
}

std::string lastError(const char* what) {
    return std::string(what) + " failed (error " + std::to_string(GetLastError()) + ")";
}

HttpResult request(const wchar_t* method, const std::string& host, int port, const std::string& path,
                   const std::string* body, int timeoutMs) {
    HttpResult r;
    HINTERNET session = WinHttpOpen(L"dream-cpp/1.0", WINHTTP_ACCESS_TYPE_NO_PROXY, WINHTTP_NO_PROXY_NAME,
                                    WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session) { r.error = lastError("WinHttpOpen"); return r; }
    HINTERNET conn = WinHttpConnect(session, widen(host).c_str(), INTERNET_PORT(port), 0);
    HINTERNET req = conn ? WinHttpOpenRequest(conn, method, widen(path).c_str(), nullptr, WINHTTP_NO_REFERER,
                                              WINHTTP_DEFAULT_ACCEPT_TYPES, 0)
                         : nullptr;
    auto cleanup = [&] {
        if (req) WinHttpCloseHandle(req);
        if (conn) WinHttpCloseHandle(conn);
        WinHttpCloseHandle(session);
    };
    if (!req) { r.error = lastError("WinHttpOpenRequest"); cleanup(); return r; }

    WinHttpSetTimeouts(req, timeoutMs, timeoutMs, timeoutMs, timeoutMs);

    static const wchar_t* jsonHeader = L"Content-Type: application/json\r\n";
    BOOL ok = WinHttpSendRequest(req, body ? jsonHeader : WINHTTP_NO_ADDITIONAL_HEADERS, body ? DWORD(-1) : 0,
                                 body ? const_cast<char*>(body->data()) : WINHTTP_NO_REQUEST_DATA,
                                 body ? DWORD(body->size()) : 0, body ? DWORD(body->size()) : 0, 0);
    if (ok) ok = WinHttpReceiveResponse(req, nullptr);
    if (!ok) {
        r.timedOut = GetLastError() == ERROR_WINHTTP_TIMEOUT;
        r.error = r.timedOut ? "timed out" : lastError("request");
        cleanup();
        return r;
    }

    DWORD status = 0, size = sizeof(status);
    WinHttpQueryHeaders(req, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX,
                        &status, &size, WINHTTP_NO_HEADER_INDEX);
    r.status = int(status);

    for (;;) {
        DWORD avail = 0;
        if (!WinHttpQueryDataAvailable(req, &avail)) {
            r.timedOut = GetLastError() == ERROR_WINHTTP_TIMEOUT;
            r.error = r.timedOut ? "timed out" : lastError("read");
            break;
        }
        if (avail == 0) break;
        std::string chunk(avail, '\0');
        DWORD got = 0;
        if (!WinHttpReadData(req, chunk.data(), avail, &got)) { r.error = lastError("read"); break; }
        r.body.append(chunk.data(), got);
    }
    cleanup();
    return r;
}

} // namespace

HttpResult httpPost(const std::string& host, int port, const std::string& path, const std::string& jsonBody, int timeoutMs) {
    return request(L"POST", host, port, path, &jsonBody, timeoutMs);
}

HttpResult httpGet(const std::string& host, int port, const std::string& path, int timeoutMs) {
    return request(L"GET", host, port, path, nullptr, timeoutMs);
}
