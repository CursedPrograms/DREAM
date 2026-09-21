// http.h - a minimal blocking HTTP client (WinHTTP), enough to talk to Ollama.
#pragma once

#include <string>

struct HttpResult {
    int status = 0;        // HTTP status, 0 if the request never completed
    std::string body;
    std::string error;     // set when the request failed to complete
    bool timedOut = false;
};

// Plain http:// only (Ollama is on localhost).
HttpResult httpPost(const std::string& host, int port, const std::string& path,
                    const std::string& jsonBody, int timeoutMs);
HttpResult httpGet(const std::string& host, int port, const std::string& path, int timeoutMs);
