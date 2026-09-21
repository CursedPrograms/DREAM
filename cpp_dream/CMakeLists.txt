cmake_minimum_required(VERSION 3.20)
project(dream_cpp CXX C)

if(NOT WIN32)
    message(FATAL_ERROR "dream.exe uses the Win32 API (window, mic, speakers, serial) and only builds on Windows.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- Libraries fetched at configure time (needs internet the first time) ----
include(FetchContent)

# Speech recognition: whisper.cpp, built into the exe (the C++ stand-in for the
# PyTorch Whisper that dream.py uses). Needs models/ggml-tiny.en.bin at runtime.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)   # no libgomp DLL to ship
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)   # build for any x86-64 with AVX2, not just this CPU
set(GGML_AVX2 ON CACHE BOOL "" FORCE)
set(GGML_FMA ON CACHE BOOL "" FORCE)
set(GGML_F16C ON CACHE BOOL "" FORCE)
FetchContent_Declare(whisper
    GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
    GIT_TAG v1.9.4
    GIT_SHALLOW ON)

# JSON for Ollama's replies and config.json.
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
    GIT_SHALLOW ON)

FetchContent_MakeAvailable(whisper json)

# ---- Speech model (77 MB) ----
# Downloaded once, next to the sources; dream.exe looks for models/ggml-tiny.en.bin
# beside itself, one folder up, or in cpp_dream/models under the repo.
# (Uses curl, which ships with Windows 10/11: some CMake builds have no CA
# certificates, so file(DOWNLOAD) can't do https.)
set(WHISPER_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/models/ggml-tiny.en.bin")
set(WHISPER_MODEL_URL "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin")
if(NOT EXISTS "${WHISPER_MODEL}")
    message(STATUS "Downloading the Whisper model (77 MB) - this can take a minute...")
    find_program(CURL_EXE NAMES curl.exe curl)
    set(dl_code 1)
    if(CURL_EXE)
        get_filename_component(model_dir "${WHISPER_MODEL}" DIRECTORY)
        file(MAKE_DIRECTORY "${model_dir}")
        execute_process(COMMAND "${CURL_EXE}" -L -f -sS -o "${WHISPER_MODEL}.part" "${WHISPER_MODEL_URL}"
                        RESULT_VARIABLE dl_code)
    endif()
    set(dl_size 0)
    if(EXISTS "${WHISPER_MODEL}.part")
        file(SIZE "${WHISPER_MODEL}.part" dl_size)
    endif()
    if(dl_code EQUAL 0 AND dl_size GREATER 70000000)
        file(RENAME "${WHISPER_MODEL}.part" "${WHISPER_MODEL}")
    else()
        file(REMOVE "${WHISPER_MODEL}.part")
        message(WARNING "Could not download the Whisper model. Put ggml-tiny.en.bin in cpp_dream/models/ yourself "
                        "(${WHISPER_MODEL_URL}).")
    endif()
endif()

# ---- dream.exe ----
add_executable(dream
    main.cpp
    brain.cpp
    audio.cpp
    common.cpp
    config.cpp
    display.cpp
    http.cpp
    llm.cpp
    memory.cpp
    sensors.cpp
    state.cpp
    stt.cpp
    sysinfo.cpp
    tts.cpp
    video.cpp
    wav.cpp
)
target_compile_options(dream PRIVATE -O2 -Wall -Wno-unused-function)
target_link_libraries(dream PRIVATE
    whisper nlohmann_json::nlohmann_json
    winmm winhttp mfplat mfreadwrite mfuuid ole32 uuid setupapi iphlpapi ws2_32 user32 gdi32)
# One self-contained exe: no libstdc++/libgcc/winpthreads DLLs to carry around.
target_link_options(dream PRIVATE -static)

# A small program to check Whisper works on its own: whisper_probe <model> <wav>
add_executable(whisper_probe EXCLUDE_FROM_ALL whisper_probe.cpp wav.cpp)
target_link_libraries(whisper_probe PRIVATE whisper)
target_link_options(whisper_probe PRIVATE -static)
