@echo off
REM DREAM's EMOCA head -> rigged dream_rig.blend + dream_rig.fbx in emoca_test\output
REM   make_rig.bat                               rig the head from the last fit
REM   make_rig.bat --fit                         refit from the photos first, then rig
REM   make_rig.bat --fit --images images\a.jpg   refit from other photos
REM Needs Blender 4.x and the EMOCA/FLAME models (download_weights.bat, EMOCA = Y).
setlocal
cd /d "%~dp0.."
venv311\Scripts\python.exe emoca_test\make_rig.py %*
if errorlevel 1 pause
