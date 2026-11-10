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
  echo Bulut yerine bilgisayardaki lokal modelleri kullanmak icin "lokal" yazin.
  set /p "KEY=Anahtar: "
  call :writeenv
)

rem --- Lokal mod (.env icinde SAGLAYICI=lokal) ---
findstr /i /r "^SAGLAYICI=lokal" ".env" >nul 2>nul
if not errorlevel 1 call :lokal

echo.
echo Uygulama baslatiliyor: http://localhost:8000  ^(kapatmak icin bu pencereyi kapatin^)
start "" cmd /c "timeout /t 3 >nul & start http://localhost:8000"
".venv\Scripts\python.exe" app.py
pause
exit /b 0

:writeenv
if /i "%KEY%"=="lokal" (> ".env" echo SAGLAYICI=lokal& exit /b 0)
if defined KEY (> ".env" echo GEMINI_API_KEY=%KEY%) else (> ".env" echo # GEMINI_API_KEY=buraya_anahtar)
exit /b 0

:lokal
if not exist ".venv\lokal-kurulu.txt" (
  echo.
  echo Lokal model paketleri kuruluyor ^(torch, whisper, transformers, piper^). Ilk seferde 5-10 dakika surebilir...
  ".venv\Scripts\python.exe" -m pip install -r requirements-lokal.txt -q || (echo [HATA] Lokal paketler kurulamadi. Internet/proxy ayarlarini kontrol edin. & pause & exit /b 1)
  > ".venv\lokal-kurulu.txt" echo ok
)
set "LLM=gemma3:4b"
for /f "usebackq tokens=1,* delims==" %%a in (".env") do if /i "%%a"=="LOKAL_LLM" set "LLM=%%b"
where ollama >nul 2>nul
if errorlevel 1 (
  echo.
  echo [UYARI] Ollama bulunamadi. Diyalog modeli icin Ollama gerekli:
  echo   1^) https://ollama.com/download adresinden kurun
  echo   2^) Bu pencereyi kapatip baslat.bat'i yeniden calistirin ^(model otomatik indirilir^)
  echo.
  exit /b 0
)
ollama list | findstr /i /c:"%LLM%" >nul || (echo %LLM% modeli indiriliyor ^(ilk sefer, ~3 GB^)... & ollama pull %LLM%)
exit /b 0
