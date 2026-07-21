# ============================================================
#  LimpiarAudio — instalación de AudioSep (FASE 5, opcional)
# ============================================================
# AudioSep (aislar CUALQUIER sonido por descripción) vive en un ENTORNO AISLADO
# (.venv-audiosep) para no mezclar su stack (lightning/transformers/CLAP) con el
# de los motores de las fases 1-4. El backend lo invoca por subproceso.
#
# Requiere ~4 GB de descarga (repo + 2 checkpoints) y bastante RAM/CPU (o GPU).
# Si NO tienes GPU, el aislamiento local funciona pero es lento (~40 s/sonido);
# para audios largos usa el notebook  notebooks/AudioSep_LimpiarAudio.ipynb  (Colab).
#
# Uso:  .\setup-audiosep.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$UV = ".\.venv\Scripts\uv.exe"
$PYS = ".venv-audiosep\Scripts\python.exe"

# 1) Clonar el repo de AudioSep (código vendido en third_party/)
if (-not (Test-Path .\third_party\AudioSep)) {
    New-Item -ItemType Directory -Force .\third_party | Out-Null
    git clone --depth 1 https://github.com/Audio-AGI/AudioSep.git .\third_party\AudioSep
}

# 2) Entorno aislado Python 3.10 + torch CPU
& $UV venv --python 3.10 .venv-audiosep
& $UV pip install --python $PYS "torch==2.1.2" "torchaudio==2.1.2" "torchvision==0.16.2" `
    --index-url https://download.pytorch.org/whl/cpu

# 3) Stack de inferencia de AudioSep (numpy<2 y setuptools<80 son necesarios:
#    torch 2.1 no soporta numpy 2, y lightning usa pkg_resources.declare_namespace)
& $UV pip install --python $PYS "numpy==1.26.4" "setuptools<80" `
    "lightning==2.1.3" "transformers==4.30.2" "torchlibrosa==0.1.0" "librosa==0.10.1" `
    soundfile scipy pyyaml "huggingface_hub<0.20" h5py pandas webdataset braceexpand wget ftfy regex pillow

# 4) Checkpoints (AudioSep ~1.3 GB + CLAP ~2.4 GB) -> third_party/AudioSep/checkpoint/
Write-Host "`n⬇  Descargando checkpoints de AudioSep (~3.6 GB, solo la 1ª vez)…" -ForegroundColor Cyan
& $PYS -c @'
import os, urllib.request
d = os.path.join("third_party", "AudioSep", "checkpoint")
os.makedirs(d, exist_ok=True)
targets = [
    ("audiosep_base_4M_steps.ckpt",
     "https://huggingface.co/spaces/Audio-AGI/AudioSep/resolve/main/checkpoint/audiosep_base_4M_steps.ckpt"),
    ("music_speech_audioset_epoch_15_esc_89.98.pt",
     "https://huggingface.co/spaces/Audio-AGI/AudioSep/resolve/main/checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt"),
]
for name, url in targets:
    dst = os.path.join(d, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000_000:
        print("  ya existe:", name); continue
    print("  descargando:", name, "…")
    urllib.request.urlretrieve(url, dst)
print("  checkpoints listos en", d)
'@

Write-Host "`n✅ AudioSep listo. El backend lo detecta solo (GET /info -> audiosep_available)." -ForegroundColor Green
