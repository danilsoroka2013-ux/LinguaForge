@echo off
rem Запуск LinguaForge (нужен Python 3.9+ и numpy)
cd /d "%~dp0"
python main.py
if errorlevel 1 pause
