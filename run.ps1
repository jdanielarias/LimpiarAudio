# Arranca el servidor de LimpiarAudio
# Uso:  .\run.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& .\.venv311\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
