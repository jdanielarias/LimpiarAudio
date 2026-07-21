# 🎧 LimpiarAudio

Limpia y mejora el audio de grabaciones y videos. Sube un archivo, escucha la
onda, límpialo con un clic y compara **Original vs Limpio** antes de exportar.

## Estado

- **FASE 1** ✅ Subida (audio/video), conversión a WAV 48 kHz estéreo con ffmpeg,
  reproductor con forma de onda, dos modos (Rápido / Profesional).
- **FASE 2** ✅ **Modo Rápido**: limpieza automática de voz en un clic, con dos
  niveles de calidad, comparador A/B y export WAV/MP3.
- **FASE 3** ✅ **Modo Profesional (detección de sonidos)**: al subir un audio se
  analiza con **PANNs Cnn14** (AudioSet, 527 clases) recorriéndolo en ventanas de
  ~1 s. Devuelve una lista de los sonidos principales con su probabilidad (pico),
  su presencia (% del tiempo) y los **tramos de tiempo** donde aparece cada uno.
  Solo lista sonidos con **presencia sostenida** (≥ `MIN_ACTIVE_WINDOWS` ventanas
  por encima de `DETECT_THRESHOLD`), para evitar falsos positivos por picos
  aislados de 1 s (animales, disparos, etc.). Los nombres de las categorías más
  comunes se muestran traducidos al español.
- **FASE 4** ✅ **Modo Profesional (separar y mezclar)**: separa la **Voz / Diálogo**
  del **fondo** (música + ambiente) con **Demucs** (`htdemucs`; se suman batería,
  bajo y otros en una única pista "fondo"). Enfoque **audiovisual** (documentales,
  entrevistas, cine): cada pista es una fila con activar/desactivar, volumen 0–200%
  y botón para escucharla sola. La mezcla final se previsualiza y se exporta en WAV/MP3.
- **FASE 5** ✅ **Aislar cualquier sonido (AudioSep)**: junto a cada sonido de la
  lista de la FASE 3 hay un botón **«Aislar este sonido»** que usa **AudioSep**
  (separación guiada por texto). El sonido aislado se añade como **una pista más**
  del mezclador de la FASE 4. AudioSep corre en un **entorno aislado**
  (`.venv-audiosep`, invocado por subproceso) y, si no hay GPU, también en
  **Google Colab** (`notebooks/AudioSep_LimpiarAudio.ipynb`). Ver *Limitaciones* abajo.
- **FASE 6** ✅ **App de escritorio (Electron)**: `desktop/` empaqueta un lanzador
  que arranca el servidor local y abre la interfaz en su propia ventana. Genera un
  **instalador de Windows (NSIS)**. Ver *App de escritorio* abajo.

## Instalación y arranque

```powershell
.\setup.ps1     # crea el entorno (Python 3.11 + motores de audio)
.\run.ps1       # arranca el servidor en http://127.0.0.1:8000
```

## Niveles de calidad (Modo Rápido)

| Calidad | Motor | Velocidad (CPU) | Notas |
|--------|-------|-----------------|-------|
| ⚡ **Rápida** | DeepFilterNet 3 | ~20× tiempo real (RTF ≈ 0.05) | Funciona sin GPU. Denoiser que preserva la fase. |
| 💎 **Alta** | Resemble **denoiser** (fiel) | ~3× tiempo real (≈ 79 s / 4 min) | Quita más ruido **sin regenerar la voz**. Límite 5 min. |

Ambos niveles son **denoisers fieles**: quitan ruido conservando la voz. La calidad
alta usa el *denoiser* de Resemble (no su modo `enhance` generativo, que regeneraba
la voz con un vocoder y provocaba coloración robótica, amplificación de fondos y
alteración de la voz). Medido sobre una grabación real de 4 min: la voz se conserva
al **99 %** (correlación 0.99 con el original) y el ruido en los silencios baja de
−33.8 a **−37.5 dBFS**.

## API

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/upload` | Sube audio/video → WAV 48 kHz estéreo. Devuelve `id` y duración. |
| `POST` | `/clean/{id}` | Limpia. Campo `quality` = `rapida` \| `alta`. |
| `POST` | `/analyze/{id}` | Detecta los sonidos (PANNs Cnn14). Lista con probabilidad, presencia y tramos. Cacheado en `analysis.json`. |
| `POST` | `/separate/{id}` | Separa en 2 pistas: `voz` (diálogo) y `fondo` (música + ambiente) con Demucs. Cacheado en `separation.json` + `stems/`. |
| `GET`  | `/stem/{id}/{name}` | Sirve una pista suelta (para escucharla sola). |
| `POST` | `/mix/{id}` | Cuerpo JSON `{tracks:[{name,enabled,gain}]}` (gain 0–2). Combina y escribe `mix.wav`. |
| `GET`  | `/mix-audio/{id}` | Previsualización de la mezcla. |
| `GET`  | `/export-mix/{id}?format=wav\|mp3` | Descarga la mezcla final. |
| `POST` | `/isolate/{id}` | Aísla un sonido. `query` = descripción EN INGLÉS, `label` = nombre a mostrar. Añade la pista aislada. |
| `GET`  | `/info` | GPU disponible + `audiosep_available`. |
| `GET`  | `/audio/{id}` | Audio original (WAV). |
| `GET`  | `/clean-audio/{id}` | Audio limpio (WAV), para el A/B. |
| `GET`  | `/download/{id}?format=wav\|mp3` | Descarga el resultado. |
| `GET`  | `/info` | Indica si hay GPU disponible. |

Errores claros para: formato no soportado, archivo vacío o dañado, id inválido,
y **audio demasiado largo** (413) según el límite de cada calidad.

## Notas técnicas (Windows)

El backend corre en **Python 3.11** (DeepFilterNet y Resemble Enhance no soportan
3.13). Dos detalles resueltos en `backend/main.py`:

1. **deepspeed**: dependencia de resemble-enhance que no compila en Windows y solo
   se usa en entrenamiento. Se sustituye por un *stub* inyectado en `sys.modules`;
   la inferencia carga el modelo con `torch.load` y no lo necesita.
2. **PosixPath**: los `hparams.yaml` de Resemble venían con rutas `PosixPath`
   (guardadas en Linux). Se mapea `PosixPath → WindowsPath` al cargar.

ffmpeg se obtiene embebido vía `imageio-ffmpeg` (incluye libmp3lame para el MP3),
sin necesidad de instalarlo aparte.

## Estructura

```
backend/main.py        API FastAPI + limpieza + PANNs + Demucs + AudioSep
backend/labels_es.py   traducción al español de las etiquetas de AudioSet
frontend/index.html    UI (A/B, lista de sonidos, pistas, aislar, mezcla)
third_party/AudioSep/  repo de AudioSep (código + checkpoint/) para la FASE 5
notebooks/…            AudioSep_LimpiarAudio.ipynb (modo Google Colab, GPU)
desktop/               app de escritorio Electron (FASE 6): main.js, config.json…
storage/<id>/          original.* · source.wav · result.* · analysis.json · separation.json · stems/ · mix.*
setup.ps1 · run.ps1    instalación y arranque (fases 1-4)
setup-audiosep.ps1     instalación de AudioSep (FASE 5, opcional, entorno aislado)
```

El modelo Cnn14 y el CSV de etiquetas viven en `~/panns_data/` (los descarga
`setup.ps1`). La detección corre en CPU: ~5 s para un audio de 26 s. Los pesos de
Demucs se cachean en `~/.cache/huggingface/`; la separación en CPU tarda ~30 s
para 15 s de audio (htdemucs). AudioSep (FASE 5) usa su propio entorno
`.venv-audiosep` y sus checkpoints en `third_party/AudioSep/checkpoint/`
(~3.6 GB); el aislamiento en CPU tarda ~40 s por sonido.

## AudioSep (FASE 5): instalación y limitaciones

Instalación (opcional, pesada): `.\setup-audiosep.ps1` — crea el entorno aislado,
clona el repo y descarga los checkpoints. El backend lo detecta solo
(`GET /info` → `audiosep_available: true`). Sin GPU, para audios largos usa el
notebook de Colab.

**Cómo funciona:** AudioSep codifica tu descripción de texto (con CLAP, entrenado
en **inglés**) y separa del audio lo que “se parece” a esa descripción. Por eso la
app le pasa la **etiqueta en inglés** de PANNs (`Speech`, `Music`, `Applause`…).

**Funciona bien cuando:**
- El sonido es una **categoría acústica clara y presente** (voz, música, aplausos,
  motor, lluvia, ladrido, batería…).
- La descripción en inglés es **simple y concreta** (`dog barking`, `car engine`).
- El sonido objetivo es razonablemente **prominente** en la mezcla.

**Funciona regular o mal cuando:**
- **Fuentes muy solapadas o tímbricamente parecidas** (dos voces a la vez, dos
  instrumentos similares): puede mezclarlas o dejar restos.
- **Sonido objetivo muy flojo** respecto al resto: sale con poca energía.
- **Descripciones abstractas o muy específicas** (“la guitarra del estribillo”):
  CLAP no entiende matices tan finos.
- No es un “demucs perfecto”: espera **fugas** y algo de coloración; es
  *extracción por parecido semántico*, no una separación limpia garantizada.
- **Español u otros idiomas**: peor que en inglés (por eso usamos la etiqueta EN).
- **Calidad/*artefactos***: a 32 kHz y con posible reverberación residual.

**Rendimiento:** procesa por bloques (`use_chunk`), así que admite audios largos
(**límite 15 min** en la app). En **CPU** tarda ~1 s por cada 2 s de audio más la
carga del modelo (varios minutos en audios largos); en **GPU** (Colab T4) es casi
instantáneo → recomendado para audios largos o para probar muchas descripciones.

## App de escritorio (FASE 6)

`desktop/` es una app **Electron** que actúa de **lanzador**: al abrirla arranca el
servidor local (uvicorn) usando el entorno Python del proyecto y muestra la
interfaz (los dos modos) en su propia ventana; al cerrarla, detiene el servidor.

> **Alcance:** la app **no reempaqueta** el backend de ML (varios GB de PyTorch +
> modelos + el entorno aislado de AudioSep). Envuelve la instalación local creada
> por `setup.ps1` (y opcionalmente `setup-audiosep.ps1`). Por eso el instalador es
> pequeño y la primera vez sigue haciendo falta instalar el entorno con los scripts.
> La ruta del proyecto y del intérprete se fijan en `desktop/config.json` (o con las
> variables de entorno `LIMPIARAUDIO_HOME` / `LIMPIARAUDIO_PYTHON`).

### Probar en desarrollo
```powershell
cd desktop
npm install            # descarga Electron (~180 MB)
npm start              # abre la ventana y arranca el backend
```

### Distribuible portable (sin instalador, funciona ya)
```powershell
cd desktop
npm install                    # descarga Electron (~180 MB)
$env:CSC_IDENTITY_AUTO_DISCOVERY = "false"
npm run pack                   # genera desktop/dist/win-unpacked/ (app ejecutable)
```
Se obtiene `desktop/dist/win-unpacked/LimpiarAudio.exe` (app lista para usar) y,
comprimida, `desktop/dist/LimpiarAudio-1.0.0-portable-win-x64.zip` (~106 MB) que
puedes copiar y ejecutar sin instalar.

### Instalador NSIS (.exe)
```powershell
cd desktop
$env:CSC_IDENTITY_AUTO_DISCOVERY = "false"   # sin firma de código
npm run dist
```
Genera `desktop/dist/LimpiarAudio Setup 1.0.0.exe` (elige carpeta, crea accesos
directos en escritorio y menú inicio).

> **Requisito del entorno para NSIS:** electron-builder descarga su paquete
> `winCodeSign`, que contiene *symlinks* de macOS. En Windows, crear symlinks
> exige el privilegio `SeCreateSymbolicLinkPrivilege`, que una cuenta estándar
> **no** tiene por defecto. Si `npm run dist` falla con
> *«Cannot create symbolic link … El cliente no dispone de un privilegio requerido»*,
> haz **una** de estas dos cosas (una sola vez):
> 1. **Activa el Modo de desarrollador**: Configuración → Privacidad y seguridad →
>    Para desarrolladores → *Modo de desarrollador* = Activado; **o**
> 2. Ejecuta `npm run dist` desde una **terminal como Administrador**.
>
> No es un problema del proyecto (el `win-unpacked`/ZIP portable se generan sin
> ese privilegio); solo afecta al empaquetado NSIS.

**Configuración de rutas (opcional):** la app **detecta sola** la carpeta del
proyecto y el intérprete cuando el ejecutable está dentro de ella (build in-repo o
ZIP portable), así que normalmente **no necesitas configurar nada**. Solo si
instalas la app en otra ubicación (p. ej. el instalador NSIS en *Archivos de
programa*), copia `desktop/config.example.json` a `desktop/config.json` y ajusta
las rutas (**JSON válido, barras normales `/`**), o define las variables de
entorno `LIMPIARAUDIO_HOME` / `LIMPIARAUDIO_PYTHON`. El `config.json` real es local
(está en `.gitignore`); en el repo se versiona `config.example.json`.

Ajustes de empaquetado en `desktop/package.json` (bloque `build`) y lógica de
arranque en `desktop/main.js`. Requisitos de compilación: **Node.js** (probado con
v24) y **npm**. La app usa el Chromium que Electron incluye (no necesita WebView2).
