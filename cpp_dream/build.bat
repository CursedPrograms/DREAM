@echo off
REM ============================================================
REM  Builds dream.exe  ->  cpp_dream\build\dream.exe
REM  Run setup.bat first (once) to get the compiler, the speech model and the voice.
REM  The first build downloads whisper.cpp and compiles it - expect several minutes.
REM  Later builds only recompile what changed.
REM ============================================================
cd /d "%~dp0"

set "MISSING="
where g++   >nul 2>&1 || set "MISSING=%MISSING% g++"
where cmake >nul 2>&1 || set "MISSING=%MISSING% cmake"
where ninja >nul 2>&1 || set "MISSING=%MISSING% ninja"
where git   >nul 2>&1 || set "MISSING=%MISSING% git"
if defined MISSING (
    echo Missing build tools:%MISSING%
    echo Run setup.bat to install them, then open a NEW terminal window and run build.bat again.
    exit /b 1
)

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release || goto :fail
cmake --build build --target dream -j || goto :fail

echo.
echo Built: %~dp0build\dream.exe
echo.
echo   build\dream.exe --selftest    check every part works
echo   build\dream.exe               run DREAM  ^(Esc or Q quits^)
echo   build\dream.exe --windowed    run in a normal window
echo   build\dream.exe --help        all options
exit /b 0

:fail
echo.
echo Build failed - see the messages above.
exit /b 1
