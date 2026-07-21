# ============================================================
#  LimpiarAudio — instalación reproducible del entorno (Windows)
# ============================================================
# El backend de FASE 2 corre sobre Python 3.11 porque DeepFilterNet y
# Resemble Enhance solo tienen soporte hasta ~3.11 (no 3.13).
# Este script deja listo el venv .venv311 con todo lo necesario.
#
# Uso:  .\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 1) uv (gestor rápido de paquetes y de versiones de Python)
if (-not (Test-Path .\.venv)) { py -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --quiet --upgrade pip uv

# 2) Python 3.11 (uv lo descarga sin permisos de administrador)
& .\.venv\Scripts\uv.exe python install 3.11

# 3) venv del backend en 3.11
& .\.venv\Scripts\uv.exe venv --python 3.11 .venv311
$UV = ".\.venv\Scripts\uv.exe"
$PY = ".venv311\Scripts\python.exe"

# 4) PyTorch CPU 2.1.2 (versión común compatible con ambos motores)
& $UV pip install --python $PY "torch==2.1.2" "torchaudio==2.1.2" "torchvision==0.16.2" `
    --index-url https://download.pytorch.org/whl/cpu

# 5) DeepFilterNet (calidad "rápida") + backend de audio soundfile
& $UV pip install --python $PY deepfilternet soundfile

# 6) Resemble Enhance (calidad "alta") SIN deepspeed (se usa un stub en el
#    código; deepspeed 0.12.4 no compila en Windows y solo se usa en training)
& $UV pip install --python $PY resemble-enhance --no-deps
& $UV pip install --python $PY "librosa==0.10.1" omegaconf pandas matplotlib `
    celluloid ptflops rich scipy resampy tabulate tqdm

# 7) API web
& $UV pip install --python $PY fastapi "uvicorn[standard]" python-multipart imageio-ffmpeg

# 8) PANNs (FASE 3, detección de sonidos): modelo Cnn14 de AudioSet (527 clases)
& $UV pip install --python $PY panns-inference

# 8b) Descarga del CSV de etiquetas y del checkpoint Cnn14 (~330 MB) a ~/panns_data.
#     panns-inference usa `wget` (ausente en Windows), así que lo hacemos con Python.
Write-Host "`n⬇  Descargando modelo Cnn14 de PANNs (~330 MB, solo la 1ª vez)…" -ForegroundColor Cyan
& $PY -c @'
import os, urllib.request
d = os.path.join(os.path.expanduser("~"), "panns_data")
os.makedirs(d, exist_ok=True)
targets = [
    ("class_labels_indices.csv",
     "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"),
    ("Cnn14_mAP=0.431.pth",
     "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"),
]
for name, url in targets:
    dst = os.path.join(d, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 3e8 or (name.endswith(".csv") and os.path.exists(dst)):
        print("  ya existe:", name); continue
    print("  descargando:", name)
    urllib.request.urlretrieve(url, dst)
print("  panns_data listo en", d)
'@

# 9) Demucs (FASE 4, separación de pistas). Los pesos htdemucs se descargan
#    solos en el primer uso (caché de huggingface_hub); htdemucs_6s (piano y
#    guitarra) se baja la primera vez que el usuario lo pide en la UI.
& $UV pip install --python $PY demucs

Write-Host "`n✅ Entorno listo. Arranca con:  .\run.ps1" -ForegroundColor Green
