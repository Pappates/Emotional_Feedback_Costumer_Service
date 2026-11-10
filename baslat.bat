@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title Duygu Avcisi

rem --- Python kontrolu ---
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (where python >nul 2>nul && set "PY=python")
if not defined PY (
  echo [HATA] Python bulunamadi. https://www.python.org/downloads/ adresinden Python 3.10+ kurun.
  echo Kurulumda "Add python.exe to PATH" kutusunu isaretlemeyi unutmayin.
  pause & exit /b 1
)

rem --- Ilk calistirmada sanal ortam ve paketler ---
if not exist ".venv\Scripts\python.exe" (
  echo Ilk kurulum yapiliyor, bu bir-iki dakika surebilir...
  %PY% -m venv .venv || (echo [HATA] Sanal ortam olusturulamadi. & pause & exit /b 1)
  ".venv\Scripts\python.exe" -m pip install --upgrade pip -q
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt -q || (echo [HATA] Paketler kurulamadi. Internet/proxy ayarlarini kontrol edin. & pause & exit /b 1)
)

rem --- API anahtari (.env) ---
if not exist ".env" (
  echo.
  echo Gemini API anahtarinizi yapistirin ^(https://aistudio.google.com/apikey^).
  echo Bos birakip Enter'a basarsaniz uygulama demo modunda calisir.
  set /p "KEY=Anahtar: "
  call :writeenv
)

echo.
echo Uygulama baslatiliyor: http://localhost:8000  ^(kapatmak icin bu pencereyi kapatin^)
start "" cmd /c "timeout /t 3 >nul & start http://localhost:8000"
".venv\Scripts\python.exe" app.py
pause
exit /b 0

:writeenv
if defined KEY (> ".env" echo GEMINI_API_KEY=%KEY%) else (> ".env" echo # GEMINI_API_KEY=buraya_anahtar)
exit /b 0
