"""
Worker de aislamiento para LimpiarAudio (FASE 5).

Se ejecuta DENTRO del entorno aislado .venv-audiosep y con el directorio de
trabajo (cwd) fijado por el backend en la raíz del repo de AudioSep
(third_party/AudioSep), donde viven pipeline.py, utils.py, models/, config/ y
checkpoint/. Vive en backend/ (código versionado del proyecto) para no depender
del repo de AudioSep, que se clona aparte con setup-audiosep.ps1.

Uso:
    python audiosep_worker.py <entrada.wav> "<texto en inglés>" <salida.wav>

Escribe la salida como WAV mono 32 kHz (el backend la remuestrea a 48 kHz
estéreo y la añade como una pista más del mezclador de la FASE 4).
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
# El backend invoca este worker con cwd = raíz del repo de AudioSep; añadimos esa
# ruta a sys.path para poder importar pipeline/utils/models de AudioSep.
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from pipeline import build_audiosep, separate_audio  # noqa: E402

CONFIG = "config/audiosep_base.yaml"
CKPT = "checkpoint/audiosep_base_4M_steps.ckpt"


def main() -> int:
    if len(sys.argv) != 4:
        print("uso: audiosep_worker.py <in.wav> <texto> <out.wav>", file=sys.stderr)
        return 2
    in_wav, text, out_wav = sys.argv[1], sys.argv[2], sys.argv[3]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[isolate] device={device}", flush=True)

    model = build_audiosep(config_yaml=CONFIG, checkpoint_path=CKPT, device=device)
    # use_chunk=True: inferencia por bloques, más ligera en memoria (CPU) y admite
    # audios largos.
    separate_audio(model, in_wav, text, out_wav, device=device, use_chunk=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
