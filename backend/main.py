"""
LimpiarAudio — Backend (FASE 1 + FASE 2)

FASE 1: subida de audio/video -> WAV 48kHz estéreo, reproducción y descarga.
FASE 2: Modo Rápido con limpieza automática de voz:
    * quality="rapida" -> DeepFilterNet (rápido, sin GPU).
    * quality="alta"   -> Resemble denoiser (fiel; quita ruido sin regenerar la voz).
    Comparador A/B (original vs limpio) y export WAV/MP3.

Este backend corre sobre Python 3.11 (.venv311), donde DeepFilterNet y
Resemble Enhance tienen soporte.
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import types
import uuid
import wave
from pathlib import Path

# Windows: hparams.yaml de Resemble Enhance fue guardado en Linux con objetos
# pathlib.PosixPath; sin este mapeo, cargar el YAML revienta en Windows.
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath


def _install_deepspeed_stub() -> None:
    """
    Resemble Enhance declara deepspeed==0.12.4, pero solo lo usa en sus rutas de
    ENTRENAMIENTO (Engine, init_distributed, DeepSpeedConfig). La inferencia
    (denoise/enhance) carga el modelo con torch.load + load_state_dict y nunca
    toca deepspeed. deepspeed 0.12.4 no compila en Windows (necesita MSVC + ops
    C++/CUDA), así que registramos un stub en sys.modules que satisface las
    importaciones a nivel de módulo. Si algo de entrenamiento se invocara, falla
    con un mensaje claro. Se salta si hay un deepspeed real instalado.
    """
    import importlib.util
    if "deepspeed" in sys.modules or importlib.util.find_spec("deepspeed") is not None:
        return

    def _training_only(*_a, **_k):
        raise RuntimeError(
            "deepspeed es un stub (solo inferencia): esta función es de entrenamiento "
            "y no está disponible en LimpiarAudio."
        )

    ds = types.ModuleType("deepspeed")
    ds.__version__ = "0.12.4+stub"
    ds.init_distributed = _training_only

    class DeepSpeedConfig:  # usado solo en load_G/load_D (entrenamiento)
        def __init__(self, *a, **k):
            _training_only()
    ds.DeepSpeedConfig = DeepSpeedConfig

    accel = types.ModuleType("deepspeed.accelerator")

    class _Accelerator:
        def communication_backend_name(self):
            _training_only()
    accel.get_accelerator = lambda *a, **k: _Accelerator()

    runtime = types.ModuleType("deepspeed.runtime")
    engine = types.ModuleType("deepspeed.runtime.engine")

    class DeepSpeedEngine:  # resemble hace `class Engine(DeepSpeedEngine)`
        def __init__(self, *a, **k):
            _training_only()
    engine.DeepSpeedEngine = DeepSpeedEngine

    rutils = types.ModuleType("deepspeed.runtime.utils")
    rutils.clip_grad_norm_ = _training_only

    ds.accelerator = accel
    ds.runtime = runtime
    runtime.engine = engine
    runtime.utils = rutils
    sys.modules.update({
        "deepspeed": ds,
        "deepspeed.accelerator": accel,
        "deepspeed.runtime": runtime,
        "deepspeed.runtime.engine": engine,
        "deepspeed.runtime.utils": rutils,
    })


_install_deepspeed_stub()

import imageio_ffmpeg
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# --- Rutas base -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
FRONTEND_DIR = BASE_DIR / "frontend"
STORAGE_DIR.mkdir(exist_ok=True)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".mp4", ".mov"}
TARGET_SR = 48000

# Límites de duración (segundos) por calidad, para evitar procesos eternos.
# "alta" (Resemble) es ~7x más lento que tiempo real en CPU.
MAX_SECONDS = {"rapida": 3600, "alta": 300}

# Serializa los trabajos pesados de limpieza (CPU intensiva + modelos).
_process_lock = threading.Lock()

app = FastAPI(title="LimpiarAudio", version="3.0-fase3")


# ======================================================================
#  Utilidades de archivos / audio
# ======================================================================
def _session_dir(session_id: str) -> Path:
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="id de sesión inválido")
    d = STORAGE_DIR / session_id
    if not d.is_dir():
        raise HTTPException(status_code=404, detail="sesión no encontrada")
    return d


def _wav_duration_seconds(wav_path: Path) -> float:
    with contextlib.closing(wave.open(str(wav_path), "rb")) as w:
        frames, rate = w.getnframes(), w.getframerate()
        return round(frames / float(rate), 3) if rate else 0.0


def _to_wav_48k_stereo(src: Path, dst: Path) -> None:
    cmd = [
        FFMPEG, "-y", "-i", str(src),
        "-vn", "-acodec", "pcm_s16le", "-ar", str(TARGET_SR), "-ac", "2",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists():
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        raise HTTPException(
            status_code=422,
            detail=f"No se pudo leer el archivo (¿está dañado o incompleto?):\n{tail}",
        )


def _load_wav_CT(path: Path):
    """Carga un WAV como (numpy float32 (C, T), sr)."""
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (T, C)
    return data.T, sr


def _save_wav_CT(path: Path, data_CT, sr: int) -> None:
    """Guarda numpy (C, T) float32 como WAV PCM 16-bit."""
    import numpy as np
    import soundfile as sf
    arr = np.ascontiguousarray(np.clip(data_CT, -1.0, 1.0).T)  # (T, C)
    sf.write(str(path), arr, sr, subtype="PCM_16")


# ======================================================================
#  Modelos de limpieza (carga perezosa y cacheada)
# ======================================================================
_df_cache = None
_device_cache = None


def _get_device() -> str:
    global _device_cache
    if _device_cache is None:
        import torch
        _device_cache = "cuda" if torch.cuda.is_available() else "cpu"
    return _device_cache


def _get_df():
    global _df_cache
    if _df_cache is None:
        from df.enhance import init_df
        model, df_state, _ = init_df()
        _df_cache = (model, df_state)
    return _df_cache


def _clean_rapida(src: Path, dst: Path) -> None:
    """DeepFilterNet: reducción de ruido rápida (procesa cada canal)."""
    import torch
    import numpy as np
    from df.enhance import enhance

    model, df_state = _get_df()
    sr_model = df_state.sr()  # 48000
    data, sr = _load_wav_CT(src)  # (C, T) @ 48000

    if sr != sr_model:
        import torchaudio
        data = torchaudio.functional.resample(
            torch.from_numpy(data), sr, sr_model
        ).numpy()

    out_channels = []
    for ch in data:  # procesa canal a canal (robusto para mono y estéreo)
        t = torch.from_numpy(np.ascontiguousarray(ch)).unsqueeze(0)  # (1, T)
        enh = enhance(model, df_state, t)  # (1, T)
        out_channels.append(enh.squeeze(0).cpu().numpy())

    out = np.stack(out_channels, axis=0)  # (C, T)
    _save_wav_CT(dst, out, sr_model)


def _clean_alta(src: Path, dst: Path) -> None:
    """
    Resemble Enhance en modo DENOISE (fiel), NO en modo enhance (generativo).

    El modo `enhance` regenera la voz con un vocoder CFM: da resultados brillantes
    pero introduce artefactos "robóticos", amplifica/distorsiona sonidos de fondo
    ambiguos y puede alterar la voz principal. Para cumplir el objetivo —quitar
    ruido SIN cambiar la voz— usamos el `denoiser` de Resemble, que es fiel a la
    señal original y no alucina audio nuevo.
    """
    import torch
    import torchaudio
    from resemble_enhance.enhancer.inference import denoise as re_denoise

    data, sr = _load_wav_CT(src)  # (C, T)
    mono = torch.from_numpy(data).mean(dim=0)  # (T,)  Resemble trabaja en mono
    device = _get_device()

    wav, new_sr = re_denoise(mono, sr, device)  # (T',) @ 44100  (denoise fiel)

    wav = wav.unsqueeze(0)  # (1, T')
    if new_sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, new_sr, TARGET_SR)
    stereo = wav.repeat(2, 1).cpu().numpy()  # (2, T) para mantener el estándar
    _save_wav_CT(dst, stereo, TARGET_SR)


CLEANERS = {"rapida": _clean_rapida, "alta": _clean_alta}


# ======================================================================
#  FASE 3 — Modo Profesional: detección de sonidos con PANNs (Cnn14)
#  Modelo Cnn14 entrenado en AudioSet (527 clases). Se recorre el audio en
#  ventanas de ~1 s y se obtiene, por ventana, la probabilidad de cada clase.
# ======================================================================
PANN_SR = 32000            # PANNs (Cnn14) trabaja a 32 kHz
ANALYZE_WINDOW = 1.0       # duración de cada ventana (segundos)
ANALYZE_HOP = 1.0          # avance entre ventanas (1.0 = ventanas contiguas)
ANALYZE_BATCH = 16         # ventanas por lote de inferencia (memoria acotada)
# Detección por PRESENCIA SOSTENIDA para evitar falsos positivos: un pico alto en
# una sola ventana (p. ej. "animal" o "disparo" durante 1 s) NO basta; el sonido
# debe superar DETECT_THRESHOLD en al menos MIN_ACTIVE_WINDOWS ventanas.
DETECT_THRESHOLD = 0.30    # una ventana "contiene" el sonido si su prob >= 0.30
MIN_ACTIVE_WINDOWS = 2     # nº mínimo de ventanas activas para listar (~2 s de presencia)
TOP_SOUNDS = 10            # nº máximo de sonidos a devolver
MAX_ANALYZE_SECONDS = 1800  # tope de duración analizable (30 min)

_pann_cache = None


def _get_pann():
    """Carga perezosa y cacheada del modelo Cnn14 (AudioTagging) de PANNs."""
    global _pann_cache
    if _pann_cache is None:
        os.environ.setdefault("MPLBACKEND", "Agg")  # panns importa matplotlib
        from panns_inference import AudioTagging
        ckpt = Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"
        _pann_cache = AudioTagging(checkpoint_path=str(ckpt), device=_get_device())
    return _pann_cache


def _merge_segments(active, starts, duration, bridge_windows=1):
    """
    A partir de un vector booleano (ventana activa sí/no) construye tramos de
    tiempo [inicio, fin] fusionando ventanas contiguas y puenteando huecos de
    hasta `bridge_windows` ventanas para no fragmentar en exceso.
    """
    idx = [i for i, a in enumerate(active) if a]
    if not idx:
        return []
    groups = [[idx[0], idx[0]]]
    for j in idx[1:]:
        if j - groups[-1][1] <= bridge_windows + 1:
            groups[-1][1] = j
        else:
            groups.append([j, j])
    segments = []
    for a, b in groups:
        start = starts[a]
        end = min(duration, starts[b] + ANALYZE_WINDOW)
        segments.append({"start": round(start, 1), "end": round(end, 1)})
    return segments


def _analyze_audio(source: Path) -> dict:
    """Recorre el audio en ventanas de ~1 s y detecta los sonidos presentes."""
    import numpy as np
    import torch
    import torchaudio
    from panns_inference.config import labels

    from .labels_es import to_spanish

    at = _get_pann()

    data, sr = _load_wav_CT(source)     # (C, T) float32 @ 48 kHz
    mono = data.mean(axis=0)            # PANNs trabaja en mono
    if sr != PANN_SR:
        mono = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(mono)), sr, PANN_SR
        ).numpy()

    total = mono.shape[0]
    duration = total / PANN_SR
    win = int(ANALYZE_WINDOW * PANN_SR)
    hop = int(ANALYZE_HOP * PANN_SR)

    # Trocea en ventanas de 1 s. La última, si es parcial, se rellena con ceros
    # (salvo que sea muy corta, <0.3 s, en cuyo caso se descarta).
    windows, starts = [], []
    pos = 0
    while pos < total:
        chunk = mono[pos:pos + win]
        if chunk.shape[0] < win:
            if chunk.shape[0] < win * 0.3:
                break
            chunk = np.pad(chunk, (0, win - chunk.shape[0]))
        windows.append(chunk)
        starts.append(round(pos / PANN_SR, 3))
        pos += hop
    if not windows:  # audio más corto que una ventana
        windows = [np.pad(mono, (0, max(0, win - total)))[:win]]
        starts = [0.0]

    n = len(windows)
    probs = np.zeros((n, len(labels)), dtype=np.float32)
    for i in range(0, n, ANALYZE_BATCH):
        batch = np.stack(windows[i:i + ANALYZE_BATCH]).astype(np.float32)
        clipwise, _ = at.inference(batch)   # (b, 527)
        probs[i:i + batch.shape[0]] = clipwise

    # Agrega por clase con criterio de PRESENCIA SOSTENIDA (anti falsos positivos):
    # una ventana "contiene" el sonido si su prob >= DETECT_THRESHOLD, y solo se
    # lista la clase si eso ocurre en al menos `min_active` ventanas. Así un pico
    # aislado de 1 s (animal, disparo, gallo…) queda descartado.
    peak = probs.max(axis=0)                       # (527,)
    active_mask = probs >= DETECT_THRESHOLD        # (n, 527)
    active_counts = active_mask.sum(axis=0)        # ventanas activas por clase
    # En clips muy cortos no se puede exigir 2 ventanas.
    min_active = 1 if n < 4 else MIN_ACTIVE_WINDOWS

    sounds = []
    for ci in np.argsort(peak)[::-1]:              # ordena por confianza (pico) desc.
        if active_counts[ci] < min_active:         # descarta chispazos aislados
            continue
        active = active_mask[:, ci]
        sounds.append({
            "label": labels[ci],
            "label_es": to_spanish(labels[ci]),
            "confidence": round(float(peak[ci]), 3),      # 0..1  (pico, ya confirmado)
            "presence": round(float(active.mean()), 3),   # fracción de tiempo presente
            "segments": _merge_segments(active, starts, duration),
        })
        if len(sounds) >= TOP_SOUNDS:
            break

    return {
        "duration": round(duration, 3),
        "window_seconds": ANALYZE_WINDOW,
        "n_windows": n,
        "sounds": sounds,
    }


# ======================================================================
#  FASE 4 — Modo Profesional: separar VOZ / DIÁLOGO del resto (Demucs)
#  Enfoque audiovisual (documentales, entrevistas, cine): lo útil es aislar la
#  voz del fondo, no desglosar instrumentos musicales. Demucs (htdemucs) separa
#  en voz/batería/bajo/otros; aquí se agrupa en 2 pistas:
#     * "voz"   -> la voz/diálogo
#     * "fondo" -> música + ambiente + todo lo que no es voz (suma del resto)
#  Cada pista se guarda como WAV 48 kHz estéreo en storage/<id>/stems/.
# ======================================================================
MAX_SEPARATE_SECONDS = 600  # tope de duración a separar (Demucs en CPU es lento)

# Nombres de pista -> etiqueta en español para la UI.
STEM_LABELS_ES = {
    "voz": "Voz / Diálogo",
    "fondo": "Música y fondo",
}

_demucs_cache: dict[str, object] = {}


def _get_demucs():
    """Carga perezosa y cacheada del separador Demucs (htdemucs)."""
    if "htdemucs" not in _demucs_cache:
        from demucs.api import Separator
        _demucs_cache["htdemucs"] = Separator(model="htdemucs", device=_get_device())
    return _demucs_cache["htdemucs"]


def _safe_stem_name(name: str) -> str:
    """Evita traversal: deja solo caracteres válidos de un nombre de pista."""
    return "".join(c for c in name if c.isalnum() or c == "_")


def _separate_audio(source: Path, out_dir: Path) -> list[str]:
    """
    Separa `source` en 2 pistas orientadas a audiovisual: 'voz' (diálogo) y
    'fondo' (música + ambiente = suma de batería+bajo+otros). WAV 48 kHz estéreo.
    """
    import torchaudio

    sep = _get_demucs()
    _origin, stems = sep.separate_audio_file(source)  # {vocals,drums,bass,other}
    sr = int(sep.samplerate)

    def to_48k_stereo(w):
        w = w if w.dim() == 2 else w.unsqueeze(0)   # (C, T)
        if sr != TARGET_SR:
            w = torchaudio.functional.resample(w, sr, TARGET_SR)
        if w.shape[0] == 1:
            w = w.repeat(2, 1)
        elif w.shape[0] > 2:
            w = w[:2]
        return w

    voz = to_48k_stereo(stems["vocals"])
    fondo = None
    for k in ("drums", "bass", "other"):        # todo lo que no es voz -> fondo
        if k not in stems:
            continue
        w = to_48k_stereo(stems[k])
        if fondo is None:
            fondo = w.clone()
        else:
            m = min(fondo.shape[1], w.shape[1])
            fondo = fondo[:, :m] + w[:, :m]

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_wav_CT(out_dir / "voz.wav", voz.detach().cpu().numpy(), TARGET_SR)
    _save_wav_CT(out_dir / "fondo.wav", fondo.detach().cpu().numpy(), TARGET_SR)
    return ["voz", "fondo"]


# --- Modelos del cuerpo de /mix -------------------------------------------
class TrackSetting(BaseModel):
    name: str
    enabled: bool = True
    gain: float = Field(default=1.0, ge=0.0, le=2.0)  # 0%..200%


class MixRequest(BaseModel):
    tracks: list[TrackSetting]


# ======================================================================
#  FASE 5 — Aislar CUALQUIER sonido descrito en lenguaje natural (AudioSep)
#  AudioSep vive en un entorno aislado (.venv-audiosep) y se invoca por
#  subproceso: su stack (lightning/transformers/CLAP) no debe mezclarse con el
#  de los motores de las fases anteriores. Cada sonido aislado se guarda como
#  una pista más (stems/iso_*.wav) que entra en el mezclador de la FASE 4.
# ======================================================================
AUDIOSEP_DIR = BASE_DIR / "third_party" / "AudioSep"
AUDIOSEP_PY = BASE_DIR / ".venv-audiosep" / "Scripts" / "python.exe"
AUDIOSEP_WORKER = BASE_DIR / "backend" / "audiosep_worker.py"
AUDIOSEP_CKPT = AUDIOSEP_DIR / "checkpoint" / "audiosep_base_4M_steps.ckpt"
AUDIOSEP_CLAP = AUDIOSEP_DIR / "checkpoint" / "music_speech_audioset_epoch_15_esc_89.98.pt"
# AudioSep procesa por bloques (use_chunk=True), así que soporta audios largos.
# En CPU es lento (~0.5x tiempo real + carga del modelo), por eso hay un tope
# generoso pero acotado. Con GPU (Colab) es mucho más rápido.
MAX_ISOLATE_SECONDS = 900  # 15 min


def _audiosep_available() -> bool:
    """True si el entorno aislado y ambos checkpoints están presentes."""
    return (
        AUDIOSEP_PY.exists()
        and AUDIOSEP_WORKER.exists()
        and AUDIOSEP_CKPT.exists()
        and AUDIOSEP_CLAP.exists()
        and AUDIOSEP_CKPT.stat().st_size > 1_000_000_000
        and AUDIOSEP_CLAP.stat().st_size > 1_000_000_000
    )


def _slugify(text: str) -> str:
    import unicodedata
    # Quita acentos (música -> musica) y deja solo [a-z0-9_] para rutas/URLs seguras.
    norm = unicodedata.normalize("NFKD", text or "")
    ascii_only = norm.encode("ascii", "ignore").decode("ascii").lower()
    s = "".join(c if (c.isascii() and c.isalnum()) else "_" for c in ascii_only)
    s = "_".join(filter(None, s.split("_")))
    return s[:40] or "sonido"


def _run_audiosep(source: Path, text: str, out_wav: Path) -> None:
    """Ejecuta el worker de AudioSep en el venv aislado (salida WAV mono 32 kHz)."""
    cmd = [str(AUDIOSEP_PY), str(AUDIOSEP_WORKER), str(source), text, str(out_wav)]
    proc = subprocess.run(
        cmd, cwd=str(AUDIOSEP_DIR), capture_output=True, text=True, timeout=1800
    )
    if proc.returncode != 0 or not out_wav.exists():
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-8:])
        raise RuntimeError(tail or "AudioSep no devolvió salida")


def _register_stem(sdir: Path, stem: dict) -> None:
    """Añade (o reemplaza) una pista en separation.json para que la UI la liste."""
    meta = sdir / "separation.json"
    if meta.exists():
        data = json.loads(meta.read_text(encoding="utf-8"))
    else:
        data = {"ok": True, "model": "isolated", "device": _get_device(), "stems": []}
    data.setdefault("stems", [])
    data["stems"] = [s for s in data["stems"] if s.get("name") != stem["name"]]
    data["stems"].append(stem)
    meta.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ======================================================================
#  Endpoints
# ======================================================================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado ({ext or 'sin extensión'}). "
                   f"Permitidos: {', '.join(sorted(ALLOWED_EXT))}",
        )

    session_id = str(uuid.uuid4())
    sdir = STORAGE_DIR / session_id
    sdir.mkdir(parents=True, exist_ok=True)

    original = sdir / f"original{ext}"
    with original.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    if original.stat().st_size == 0:
        shutil.rmtree(sdir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="El archivo está vacío")

    source_wav = sdir / "source.wav"
    _to_wav_48k_stereo(original, source_wav)  # 422 claro si está dañado

    duration = _wav_duration_seconds(source_wav)
    return {
        "id": session_id,
        "duration": duration,
        "filename": file.filename,
        "sample_rate": TARGET_SR,
        "channels": 2,
    }


@app.get("/info")
def info():
    """Reporta hardware (GPU) y si el aislamiento por IA (AudioSep) está disponible."""
    device = _get_device()
    return {
        "device": device,
        "cuda": device == "cuda",
        "audiosep_available": _audiosep_available(),
    }


@app.post("/clean/{session_id}")
def clean(session_id: str, quality: str = Form("rapida")):
    quality = (quality or "rapida").lower()
    if quality not in CLEANERS:
        raise HTTPException(
            status_code=400,
            detail=f"Calidad inválida '{quality}'. Usa 'rapida' o 'alta'.",
        )

    sdir = _session_dir(session_id)
    source = sdir / "source.wav"
    if not source.exists():
        raise HTTPException(status_code=404, detail="audio de origen no encontrado")

    duration = _wav_duration_seconds(source)
    limit = MAX_SECONDS[quality]
    if duration > limit:
        raise HTTPException(
            status_code=413,
            detail=(
                f"El audio dura {duration:.0f}s y supera el máximo de {limit}s "
                f"para la calidad '{quality}'. "
                + ("Prueba con la calidad 'Rápida' o recorta el audio."
                   if quality == "alta" else "Recorta el audio e inténtalo de nuevo.")
            ),
        )

    result = sdir / "result.wav"
    # invalida un MP3 previo de otra limpieza
    mp3 = sdir / "result.mp3"
    if mp3.exists():
        mp3.unlink()

    t0 = time.time()
    try:
        with _process_lock:
            CLEANERS[quality](source, result)
    except HTTPException:
        raise
    except MemoryError:
        raise HTTPException(
            status_code=507,
            detail="Sin memoria suficiente para procesar. Prueba un audio más corto "
                   "o la calidad 'Rápida'.",
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=422,
            detail=f"No se pudo limpiar el audio ({type(e).__name__}): {e}",
        )

    proc = round(time.time() - t0, 2)
    return {
        "ok": True,
        "quality": quality,
        "engine": "DeepFilterNet" if quality == "rapida" else "Resemble (denoiser)",
        "device": _get_device(),
        "duration": duration,
        "proc_seconds": proc,
    }


@app.post("/analyze/{session_id}")
def analyze(session_id: str):
    """
    FASE 3 — Detecta qué sonidos hay en el audio (Modo Profesional).
    Devuelve una lista de sonidos principales con su probabilidad (pico),
    su presencia (fracción de tiempo) y los tramos donde aparece cada uno.
    El resultado se cachea en analysis.json.
    """
    sdir = _session_dir(session_id)
    source = sdir / "source.wav"
    if not source.exists():
        raise HTTPException(status_code=404, detail="audio de origen no encontrado")

    duration = _wav_duration_seconds(source)
    if duration > MAX_ANALYZE_SECONDS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"El audio dura {duration:.0f}s y supera el máximo de "
                f"{MAX_ANALYZE_SECONDS}s para el análisis. Recórtalo e inténtalo de nuevo."
            ),
        )

    cache = sdir / "analysis.json"
    if cache.exists():  # análisis ya calculado para este audio
        return json.loads(cache.read_text(encoding="utf-8"))

    t0 = time.time()
    try:
        with _process_lock:  # serializa con la limpieza (ambos son CPU intensiva)
            result = _analyze_audio(source)
    except MemoryError:
        raise HTTPException(
            status_code=507,
            detail="Sin memoria suficiente para analizar. Prueba con un audio más corto.",
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=422,
            detail=f"No se pudo analizar el audio ({type(e).__name__}): {e}",
        )

    result["proc_seconds"] = round(time.time() - t0, 2)
    result["device"] = _get_device()
    result["model"] = "PANNs Cnn14 (AudioSet, 527 clases)"
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


@app.post("/separate/{session_id}")
def separate(session_id: str):
    """
    FASE 4 — Separa el audio en 2 pistas: 'voz' (diálogo) y 'fondo' (música +
    ambiente). Pensado para audiovisual (documentales, entrevistas, cine).
    Cachea el resultado en separation.json + storage/<id>/stems/voz|fondo.wav.
    """
    sdir = _session_dir(session_id)
    source = sdir / "source.wav"
    if not source.exists():
        raise HTTPException(status_code=404, detail="audio de origen no encontrado")

    duration = _wav_duration_seconds(source)
    if duration > MAX_SEPARATE_SECONDS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"El audio dura {duration:.0f}s y supera el máximo de "
                f"{MAX_SEPARATE_SECONDS}s para la separación. Recórtalo e inténtalo de nuevo."
            ),
        )

    stems_dir = sdir / "stems"
    voz_wav, fondo_wav = stems_dir / "voz.wav", stems_dir / "fondo.wav"

    t0 = time.time()
    if not (voz_wav.exists() and fondo_wav.exists()):  # (re)genera si no están
        try:
            with _process_lock:
                _separate_audio(source, stems_dir)
        except MemoryError:
            raise HTTPException(
                status_code=507,
                detail="Sin memoria suficiente para separar. Prueba con un audio más corto.",
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(
                status_code=422,
                detail=f"No se pudo separar el audio ({type(e).__name__}): {e}",
            )

    stems = [{"name": n, "label_es": STEM_LABELS_ES[n]} for n in ("voz", "fondo")]
    for s in stems:  # registra las pistas base (conserva las aisladas de AudioSep)
        _register_stem(sdir, s)
    for stale in ("mix.wav", "mix.mp3"):  # invalida una mezcla anterior
        p = sdir / stale
        if p.exists():
            p.unlink()

    return {
        "ok": True,
        "device": _get_device(),
        "duration": duration,
        "proc_seconds": round(time.time() - t0, 2),
        "stems": stems,
    }


@app.get("/stem/{session_id}/{name}")
def get_stem(session_id: str, name: str):
    """Sirve una pista separada (para escucharla en solitario)."""
    sdir = _session_dir(session_id)
    safe = _safe_stem_name(name)
    wav = sdir / "stems" / f"{safe}.wav"
    if not wav.exists():
        raise HTTPException(status_code=404, detail="pista no encontrada")
    return FileResponse(wav, media_type="audio/wav", filename=f"{safe}.wav")


@app.post("/mix/{session_id}")
def mix(session_id: str, req: MixRequest):
    """
    FASE 4 — Combina las pistas seleccionadas con su ganancia (0..2 = 0%..200%)
    y guarda la mezcla en mix.wav. Devuelve el pico y si hubo saturación.
    """
    import numpy as np

    sdir = _session_dir(session_id)
    stems_dir = sdir / "stems"
    if not stems_dir.is_dir():
        raise HTTPException(status_code=404, detail="primero separa las pistas")

    included = [t for t in req.tracks if t.enabled]
    if not included:
        raise HTTPException(status_code=400, detail="selecciona al menos una pista")

    mix_arr = None
    for t in included:
        safe = _safe_stem_name(t.name)
        p = stems_dir / f"{safe}.wav"
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"pista '{t.name}' no encontrada")
        data, _sr = _load_wav_CT(p)                 # (C, T) @ 48 kHz
        data = data * max(0.0, min(2.0, float(t.gain)))
        if mix_arr is None:
            mix_arr = np.zeros_like(data)
        n = min(mix_arr.shape[1], data.shape[1])    # alinea por si difieren
        mix_arr[:, :n] += data[:, :n]

    peak = float(np.max(np.abs(mix_arr))) if mix_arr is not None and mix_arr.size else 0.0
    _save_wav_CT(sdir / "mix.wav", mix_arr, TARGET_SR)  # recorta a [-1, 1]
    mp3 = sdir / "mix.mp3"
    if mp3.exists():  # invalida un MP3 de una mezcla anterior
        mp3.unlink()

    return {
        "ok": True,
        "peak": round(peak, 3),
        "clipped": peak > 1.0,
        "tracks": [t.name for t in included],
    }


@app.get("/mix-audio/{session_id}")
def get_mix_audio(session_id: str):
    """Sirve la mezcla final (para previsualizarla)."""
    sdir = _session_dir(session_id)
    wav = sdir / "mix.wav"
    if not wav.exists():
        raise HTTPException(status_code=404, detail="todavía no has creado la mezcla")
    return FileResponse(wav, media_type="audio/wav", filename="mezcla.wav")


@app.get("/export-mix/{session_id}")
def export_mix(session_id: str, format: str = "wav"):
    """Descarga la mezcla final en WAV o MP3."""
    fmt = (format or "wav").lower()
    if fmt not in ("wav", "mp3"):
        raise HTTPException(status_code=400, detail="Formato debe ser 'wav' o 'mp3'")

    sdir = _session_dir(session_id)
    mix_wav = sdir / "mix.wav"
    if not mix_wav.exists():
        raise HTTPException(status_code=404, detail="Primero crea la mezcla.")

    if fmt == "wav":
        return FileResponse(mix_wav, media_type="audio/wav", filename="mezcla.wav")

    mp3 = sdir / "mix.mp3"
    if not mp3.exists():
        cmd = [FFMPEG, "-y", "-i", str(mix_wav),
               "-codec:a", "libmp3lame", "-q:a", "2", str(mp3)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not mp3.exists():
            tail = "\n".join(proc.stderr.strip().splitlines()[-6:])
            raise HTTPException(status_code=500, detail=f"Error creando MP3:\n{tail}")
    return FileResponse(mp3, media_type="audio/mpeg", filename="mezcla.mp3")


@app.post("/isolate/{session_id}")
def isolate(session_id: str, query: str = Form(...), label: str = Form(None)):
    """
    FASE 5 — Aísla CUALQUIER sonido descrito en lenguaje natural (AudioSep).
    `query` es la descripción EN INGLÉS (p. ej. "applause", "engine noise").
    `label` es el nombre a mostrar (opcional; por defecto = query).
    El sonido aislado se añade como una pista más del mezclador (FASE 4).
    """
    if not _audiosep_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "El aislamiento por IA (AudioSep) no está disponible en este equipo "
                "(falta el entorno .venv-audiosep o los checkpoints). Usa el modo "
                "Google Colab: notebooks/AudioSep_LimpiarAudio.ipynb."
            ),
        )

    sdir = _session_dir(session_id)
    source = sdir / "source.wav"
    if not source.exists():
        raise HTTPException(status_code=404, detail="audio de origen no encontrado")

    q = (query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Indica una descripción del sonido.")
    label = (label or q).strip()

    duration = _wav_duration_seconds(source)
    if duration > MAX_ISOLATE_SECONDS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"El audio dura {duration:.0f}s y supera el máximo de {MAX_ISOLATE_SECONDS}s "
                f"para aislar en CPU. Recórtalo o usa el modo Google Colab (con GPU)."
            ),
        )

    stems_dir = sdir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    name = "iso_" + _slugify(label)
    tmp_out = sdir / f"{name}_32k.wav"

    t0 = time.time()
    try:
        with _process_lock:  # AudioSep es CPU intensiva; serializa con todo lo demás
            _run_audiosep(source, q, tmp_out)
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="AudioSep tardó demasiado (CPU). Prueba un audio más corto o usa Colab.",
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=422,
            detail=f"No se pudo aislar el sonido ({type(e).__name__}): {e}",
        )

    # Salida del worker: WAV mono 32 kHz -> 48 kHz estéreo (estándar del proyecto).
    import torch
    import torchaudio

    data, sr = _load_wav_CT(tmp_out)
    w = torch.from_numpy(data)
    if sr != TARGET_SR:
        w = torchaudio.functional.resample(w, sr, TARGET_SR)
    if w.shape[0] == 1:
        w = w.repeat(2, 1)
    _save_wav_CT(stems_dir / f"{name}.wav", w.numpy(), TARGET_SR)
    tmp_out.unlink(missing_ok=True)

    stem = {"name": name, "label_es": label, "query": q, "isolated": True}
    _register_stem(sdir, stem)
    for stale in ("mix.wav", "mix.mp3"):  # invalida la mezcla previa
        p = sdir / stale
        if p.exists():
            p.unlink()

    return {
        "ok": True,
        "proc_seconds": round(time.time() - t0, 2),
        "device": _get_device(),
        "stem": stem,
    }


@app.get("/audio/{session_id}")
def get_audio(session_id: str):
    """Audio ORIGINAL (para el A/B y la reproducción)."""
    sdir = _session_dir(session_id)
    wav = sdir / "source.wav"
    if not wav.exists():
        raise HTTPException(status_code=404, detail="audio no encontrado")
    return FileResponse(wav, media_type="audio/wav", filename="original.wav")


@app.get("/clean-audio/{session_id}")
def get_clean_audio(session_id: str):
    """Audio LIMPIO (para el A/B). 404 si aún no se ha limpiado."""
    sdir = _session_dir(session_id)
    wav = sdir / "result.wav"
    if not wav.exists():
        raise HTTPException(status_code=404, detail="todavía no has limpiado el audio")
    return FileResponse(wav, media_type="audio/wav", filename="limpio.wav")


@app.get("/download/{session_id}")
def download(session_id: str, format: str = "wav"):
    fmt = (format or "wav").lower()
    if fmt not in ("wav", "mp3"):
        raise HTTPException(status_code=400, detail="Formato debe ser 'wav' o 'mp3'")

    sdir = _session_dir(session_id)
    result = sdir / "result.wav"
    if not result.exists():
        raise HTTPException(
            status_code=404,
            detail="Primero limpia el audio antes de descargarlo.",
        )

    if fmt == "wav":
        return FileResponse(result, media_type="audio/wav", filename="limpio.wav")

    # MP3 bajo demanda (cacheado)
    mp3 = sdir / "result.mp3"
    if not mp3.exists():
        cmd = [FFMPEG, "-y", "-i", str(result),
               "-codec:a", "libmp3lame", "-q:a", "2", str(mp3)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not mp3.exists():
            tail = "\n".join(proc.stderr.strip().splitlines()[-6:])
            raise HTTPException(status_code=500, detail=f"Error creando MP3:\n{tail}")
    return FileResponse(mp3, media_type="audio/mpeg", filename="limpio.mp3")


@app.get("/", response_class=HTMLResponse)
def index():
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
