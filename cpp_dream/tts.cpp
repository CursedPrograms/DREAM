#include "tts.h"

#include "common.h"
#include "config.h"

#include <windows.h>

#include <filesystem>

bool ttsSynthesize(const std::string& text, const std::string& wavPath) {
    if (text.empty()) return false;
    if (cfg().voiceModel.empty()) {
        dream::logf("TTS: no voice model (*.onnx) in %s", cfg().voicesDir.c_str());
        return false;
    }

    // stdin pipe for the text; stdout/stderr thrown away (Piper is chatty).
    SECURITY_ATTRIBUTES sa = {sizeof(sa), nullptr, TRUE};
    HANDLE inRead = nullptr, inWrite = nullptr;
    if (!CreatePipe(&inRead, &inWrite, &sa, 0)) return false;
    SetHandleInformation(inWrite, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul = CreateFileA("NUL", GENERIC_WRITE, FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);

    STARTUPINFOA si = {sizeof(si)};
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    si.hStdInput = inRead;
    si.hStdOutput = nul;
    si.hStdError = nul;

    std::string cmd = "\"" + cfg().piperExe + "\" -m \"" + cfg().voiceModel + "\" -f \"" + wavPath + "\"";
    PROCESS_INFORMATION pi = {};
    BOOL started = CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    CloseHandle(inRead);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!started) {
        CloseHandle(inWrite);
        dream::logf("TTS: could not start %s (error %lu)", cfg().piperExe.c_str(), GetLastError());
        return false;
    }

    DWORD written = 0;
    WriteFile(inWrite, text.data(), DWORD(text.size()), &written, nullptr);
    CloseHandle(inWrite); // EOF: Piper starts speaking

    bool ok = false;
    if (WaitForSingleObject(pi.hProcess, 15000) == WAIT_OBJECT_0) {
        DWORD code = 1;
        GetExitCodeProcess(pi.hProcess, &code);
        std::error_code ec;
        ok = code == 0 && std::filesystem::file_size(wavPath, ec) > 100;
    } else {
        TerminateProcess(pi.hProcess, 1);
        dream::logf("TTS: Piper took too long");
    }
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return ok;
}
