(function startLimpiarAudio() {
  "use strict";

  const {
    bestSpeechSource,
    clamp,
    endpointPath,
    finiteNumber,
    fmtTime,
    normalizedPercent,
    safeText,
  } = window.LimpiarAudioUtils;
  const byId = (id) => document.getElementById(id);
  const mutationMethods = new Set(["POST", "PUT", "PATCH", "DELETE"]);
  const allowedExtensions = new Set(["mp3", "wav", "m4a", "mp4", "mov"]);

  let sessionId = null;
  let quality = "rapida";
  let generation = 0;
  let wsLoaded = null;
  let wsA = null;
  let wsB = null;
  let previewAudio = new Audio();
  let previewKey = null;
  let lastSounds = [];
  let lastAnalyzeSource = "original";
  let availableStemNames = new Set();
  let artifacts = { clean: false, restore: false, mix: false, loudness: false };
  let hardware = { loaded: false, cuda: false };
  let activeOps = 0;
  const controllers = new Map();
  const timers = new Set();

  function createElement(tagName, className, text) {
    const element = document.createElement(tagName);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = String(text);
    return element;
  }

  function spinner() {
    const element = createElement("span", "spinner");
    element.setAttribute("aria-hidden", "true");
    return element;
  }

  function setStatus(element, text = "", options = {}) {
    const base = options.base || "status";
    element.className = `${base}${options.state ? ` ${options.state}` : ""}`;
    element.replaceChildren();
    if (options.busy) element.append(spinner());
    element.append(document.createTextNode(String(text)));
    element.setAttribute("aria-busy", options.busy ? "true" : "false");
  }

  function setButtonBusy(button, text, busy) {
    button.replaceChildren();
    if (busy) button.append(spinner());
    button.append(document.createTextNode(text));
    button.setAttribute("aria-busy", busy ? "true" : "false");
  }

  function updateGlobalStatus() {
    const dot = byId("global-dot");
    const text = byId("global-status-text");
    if (activeOps > 0) {
      dot.className = "dot busy";
      text.textContent = "Procesando…";
    } else {
      dot.className = "dot ok";
      text.textContent = "Listo";
    }
  }

  function beginOp() {
    activeOps += 1;
    updateGlobalStatus();
  }

  function endOp() {
    activeOps = Math.max(0, activeOps - 1);
    updateGlobalStatus();
  }

  function startElapsed(element, label, context, options = {}) {
    const started = Date.now();
    const update = () => {
      if (!isActive(context)) return;
      const elapsed = Math.floor((Date.now() - started) / 1000);
      setStatus(element, `${label} ${elapsed}s${options.suffix || ""}`, {
        base: options.base,
        busy: true,
      });
    };
    update();
    const timer = window.setInterval(update, options.interval || 500);
    timers.add(timer);
    return () => {
      window.clearInterval(timer);
      timers.delete(timer);
    };
  }

  function clearTimers() {
    timers.forEach((timer) => window.clearInterval(timer));
    timers.clear();
  }

  function abortRequests() {
    controllers.forEach((controller) => controller.abort());
    controllers.clear();
  }

  function contextSnapshot() {
    return { generation, sessionId };
  }

  function isActive(context) {
    return Boolean(
      context
      && context.generation === generation
      && context.sessionId === sessionId,
    );
  }

  function staleError() {
    return new DOMException("La operación pertenece a una sesión anterior.", "AbortError");
  }

  async function responseError(response) {
    let detail = response.statusText || `HTTP ${response.status}`;
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string") detail = payload.detail;
      else if (Array.isArray(payload.detail)) {
        detail = payload.detail.map((item) => safeText(item.msg, "Dato inválido")).join("; ");
      }
    } catch { /* la respuesta no era JSON */ }
    return new Error(detail);
  }

  async function requestJson(url, options = {}, requestKey = null, context = null) {
    if (context && !isActive(context)) throw staleError();
    if (requestKey && controllers.has(requestKey)) controllers.get(requestKey).abort();
    const controller = new AbortController();
    if (requestKey) controllers.set(requestKey, controller);
    const method = String(options.method || "GET").toUpperCase();
    const headers = new Headers(options.headers || {});
    headers.set("Accept", "application/json");
    if (mutationMethods.has(method)) headers.set("X-LimpiarAudio-Request", "1");
    try {
      const response = await fetch(url, {
        ...options,
        method,
        headers,
        signal: controller.signal,
        credentials: "same-origin",
        cache: "no-store",
        redirect: "error",
      });
      if (context && !isActive(context)) throw staleError();
      if (!response.ok) throw await responseError(response);
      const payload = await response.json();
      if (context && !isActive(context)) throw staleError();
      return payload;
    } finally {
      if (requestKey && controllers.get(requestKey) === controller) controllers.delete(requestKey);
    }
  }

  function isAbort(error) {
    return error && error.name === "AbortError";
  }

  function destroyWave(instance) {
    if (instance) {
      try { instance.destroy(); } catch { /* ya estaba destruido */ }
    }
    return null;
  }

  function stopPreview() {
    previewAudio.pause();
    previewAudio.removeAttribute("src");
    previewAudio.load();
    previewKey = null;
    clearPreviewUi();
  }

  function resetLink(id) {
    const link = byId(id);
    link.href = "#";
    link.setAttribute("aria-disabled", "true");
    link.removeAttribute("download");
  }

  function releaseSession(id) {
    if (!id) return;
    const headers = new Headers({
      Accept: "application/json",
      "X-LimpiarAudio-Request": "1",
    });
    // Es best-effort: la UI se reinicia de inmediato y el backend espera el
    // lock si todavía hay un modelo terminando una operación.
    void fetch(endpointPath("session", id), {
      method: "DELETE",
      headers,
      credentials: "same-origin",
      cache: "no-store",
      redirect: "error",
      keepalive: true,
    }).catch(() => {});
  }

  function resetAll() {
    const previousSessionId = sessionId;
    generation += 1;
    abortRequests();
    clearTimers();
    wsLoaded = destroyWave(wsLoaded);
    wsA = destroyWave(wsA);
    wsB = destroyWave(wsB);
    stopPreview();
    sessionId = null;
    releaseSession(previousSessionId);
    lastSounds = [];
    lastAnalyzeSource = "original";
    availableStemNames = new Set();
    artifacts = { clean: false, restore: false, mix: false, loudness: false };

    ["loaded", "analyze-panel", "restore-panel", "separate-panel",
      "transcribe-panel", "loudness-panel", "ab-section", "ab-slot", "mix-bar", "restore-bar",
      "sub-bar", "loudness-bar", "lufs-bar", "sub-preview"]
      .forEach((id) => byId(id).classList.add("hidden"));
    ["analyze-empty", "restore-empty", "separate-empty", "transcribe-empty", "loudness-empty", "ab-empty"]
      .forEach((id) => byId(id).classList.remove("hidden"));
    byId("ab-col-b").classList.add("hidden");
    byId("ab-col-a").classList.remove("hidden");
    document.querySelectorAll("#ab-toggle button").forEach((button) => {
      button.setAttribute("aria-pressed", button.dataset.ab === "a" ? "true" : "false");
    });
    byId("dropzone").classList.remove("compact");
    byId("analyze-btn").disabled = false;
    activeOps = 0;
    updateGlobalStatus();
    ["sounds-list", "stems", "sub-preview", "lufs-bar", "waveform", "wave-a", "wave-b"]
      .forEach((id) => byId(id).replaceChildren());

    setStatus(byId("app-status"), "", { base: "app-status" });
    setStatus(byId("clean-status"));
    setStatus(byId("analyze-status"), "", { base: "analyze-status" });
    setStatus(byId("separate-status"));
    setStatus(byId("restore-status"));
    setStatus(byId("transcribe-status"));
    setStatus(byId("loudness-status"));
    setStatus(byId("isolate-note"), "", { base: "mix-note" });
    setStatus(byId("mix-note"), "", { base: "mix-note" });

    byId("fileinput").value = "";
    byId("fname").textContent = "—";
    byId("fmeta").textContent = "";
    byId("pp-orig").disabled = true;
    byId("pp-orig").textContent = "▶ Reproducir";
    byId("time-orig").textContent = "0:00 / 0:00";
    byId("clean-btn").disabled = false;
    byId("separate-btn").disabled = false;
    byId("restore-btn").disabled = false;
    byId("transcribe-btn").disabled = false;
    byId("loudness-btn").disabled = false;
    byId("mix-play").textContent = "▶ Escuchar mezcla";
    byId("restore-play").textContent = "▶ Escuchar restaurado";
    byId("loudness-play").textContent = "▶ Escuchar";
    ["dl-wav", "dl-mp3", "mix-wav", "mix-mp3", "restore-wav", "restore-mp3",
      "sub-srt", "sub-vtt", "sub-txt", "loudness-wav", "loudness-mp3"]
      .forEach(resetLink);
    selectQuality("rapida");
    applyToolAvailability();
  }

  function selectQuality(selectedQuality) {
    quality = selectedQuality === "alta" ? "alta" : "rapida";
    document.querySelectorAll(".seg").forEach((segment) => {
      const selected = segment.dataset.q === quality;
      segment.classList.toggle("sel", selected);
      segment.setAttribute("aria-pressed", selected ? "true" : "false");
    });
    byId("alta-note").classList.toggle("show", quality === "alta");
  }

  const PRESETS = {
    entrevista: { label: "Entrevista", quality: "alta", restoreMode: "0", loudnessPreset: "podcast" },
    podcast: { label: "Podcast", quality: "rapida", loudnessPreset: "podcast" },
    cine: { label: "Cine / Doc", quality: "alta", loudnessPreset: "cine" },
    vozoff: { label: "Voz en off", quality: "alta", restoreMode: "0", loudnessPreset: "streaming" },
  };

  function applyPreset(key) {
    const preset = PRESETS[key];
    if (!preset) return;
    const applied = [];
    if (preset.quality) {
      selectQuality(preset.quality);
      applied.push("calidad");
    }
    if (preset.restoreMode !== undefined) {
      byId("restore-mode").value = preset.restoreMode;
      applied.push("restauración");
    }
    if (preset.loudnessPreset) {
      byId("loudness-preset").value = preset.loudnessPreset;
      applied.push("normalización");
    }
    document.querySelectorAll(".preset-btn").forEach((button) => {
      const selected = button.dataset.preset === key;
      button.classList.toggle("sel", selected);
      button.setAttribute("aria-pressed", selected ? "true" : "false");
    });
    setStatus(byId("app-status"), `Ajustes de «${preset.label}» aplicados: ${applied.join(", ")}.`, {
      base: "app-status",
    });
  }

  function makePlayer(container, url, playButton, timeElement, label) {
    if (!window.WaveSurfer || typeof window.WaveSurfer.create !== "function") {
      throw new Error("El reproductor de forma de onda no está disponible.");
    }
    const wave = window.WaveSurfer.create({
      container,
      url,
      waveColor: label === "clean" ? "#1c3a34" : "#3a4145",
      progressColor: label === "clean" ? "#22c7a9" : "#4fa69c",
      cursorColor: "#e9edec",
      height: 84,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
    });
    wave.on("ready", () => {
      playButton.disabled = false;
      if (timeElement) timeElement.textContent = `0:00 / ${fmtTime(wave.getDuration())}`;
    });
    wave.on("play", () => { playButton.textContent = playButton.textContent.replace("▶", "⏸"); });
    wave.on("pause", () => { playButton.textContent = playButton.textContent.replace("⏸", "▶"); });
    wave.on("finish", () => { playButton.textContent = playButton.textContent.replace("⏸", "▶"); });
    wave.on("error", (error) => {
      setStatus(byId("app-status"), `No se pudo cargar la vista de audio: ${safeText(error && error.message, String(error))}`, {
        base: "app-status", state: "err",
      });
    });
    if (timeElement) {
      wave.on("timeupdate", (time) => {
        timeElement.textContent = `${fmtTime(time)} / ${fmtTime(wave.getDuration())}`;
      });
    }
    return wave;
  }

  function updateHardwarePill() {
    const pill = byId("gpu-pill");
    pill.classList.remove("ok");
    const enhanceDevice = hardware.engines
      && hardware.engines.enhance
      && hardware.engines.enhance.worker_health
      && hardware.engines.enhance.worker_health.device;
    const usedDevice = enhanceDevice && enhanceDevice.used;
    if (!hardware.loaded) {
      pill.textContent = "comprobando hardware…";
    } else if (usedDevice === "cuda") {
      pill.textContent = "El motor usará GPU";
      pill.classList.add("ok");
    } else if (enhanceDevice && enhanceDevice.fallback_reason) {
      pill.textContent = `Usará CPU: ${safeText(enhanceDevice.fallback_reason)}`;
    } else {
      pill.textContent = "Sin GPU (usará CPU, será más lento)";
    }
  }

  function applyWhisperModels() {
    const whisper = hardware.engines && hardware.engines.whisper;
    const available = new Set(
      whisper && Array.isArray(whisper.available_models)
        ? whisper.available_models
        : (hardware.whisper_available ? [safeText(whisper && whisper.default_model, "small")] : []),
    );
    const select = byId("whisper-model");
    Array.from(select.options).forEach((option) => {
      if (!option.dataset.baseLabel) option.dataset.baseLabel = option.textContent;
      const installed = available.has(option.value);
      option.disabled = !installed;
      option.textContent = installed ? option.dataset.baseLabel : `${option.dataset.baseLabel} — no instalado`;
    });
    if (select.selectedOptions[0] && select.selectedOptions[0].disabled) {
      const firstAvailable = Array.from(select.options).find((option) => !option.disabled);
      if (firstAvailable) select.value = firstAvailable.value;
    }
  }

  function availabilityMessage(button, status, available, pending) {
    if (pending) {
      button.disabled = true;
      setStatus(status, "Comprobando disponibilidad…");
      status.dataset.availability = "true";
      return;
    }
    button.disabled = !available;
    if (!available) {
      setStatus(status, "Aviso: no disponible — ejecuta el instalador correspondiente y vuelve a abrir la app.");
      status.dataset.availability = "true";
    } else if (status.dataset.availability === "true") {
      setStatus(status);
      delete status.dataset.availability;
    }
  }

  function applyToolAvailability() {
    updateHardwarePill();
    const pending = !hardware.loaded;
    availabilityMessage(byId("restore-btn"), byId("restore-status"), hardware.restore_available, pending);
    availabilityMessage(byId("transcribe-btn"), byId("transcribe-status"), hardware.whisper_available, pending);
    const highQuality = document.querySelector('.seg[data-q="alta"]');
    highQuality.disabled = pending || !hardware.enhance_available;
    highQuality.setAttribute("aria-disabled", highQuality.disabled ? "true" : "false");
    if (quality === "alta" && highQuality.disabled) selectQuality("rapida");
    document.querySelectorAll(".isolate-btn").forEach((button) => {
      button.disabled = pending || !hardware.audiosep_available;
    });
    applyWhisperModels();
  }

  async function loadHardwareInfo() {
    try {
      const response = await fetch("/info", {
        headers: { Accept: "application/json" },
        credentials: "same-origin",
        cache: "no-store",
        redirect: "error",
      });
      if (!response.ok) throw await responseError(response);
      const data = await response.json();
      hardware = { ...data, loaded: true };
    } catch (error) {
      hardware = { loaded: true, cuda: false };
      setStatus(byId("app-status"), `No se pudo comprobar el estado de los motores: ${error.message}`, {
        base: "app-status", state: "err",
      });
    }
    applyToolAvailability();
  }

  async function uploadFile(file) {
    const extension = file.name.includes(".") ? file.name.split(".").pop().toLowerCase() : "";
    if (!allowedExtensions.has(extension)) {
      setStatus(byId("app-status"), "Formato no admitido. Usa MP3, WAV, M4A, MP4 o MOV.", {
        base: "app-status", state: "err",
      });
      byId("fileinput").value = "";
      return;
    }
    resetAll();
    const context = contextSnapshot();
    byId("loaded").classList.remove("hidden");
    byId("dropzone").classList.add("compact");
    byId("fname").textContent = file.name;
    byId("fmeta").textContent = `subiendo… (${(file.size / 1048576).toFixed(1)} MB)`;
    const form = new FormData();
    form.append("file", file);
    try {
      const data = await requestJson("/upload", { method: "POST", body: form }, "upload", context);
      sessionId = safeText(data.id);
      context.sessionId = sessionId;
      if (!sessionId || context.generation !== generation) throw staleError();
      const channels = finiteNumber(data.channels, 2);
      const channelLabel = channels === 1 ? "mono" : channels === 2 ? "estéreo" : `${channels} canales`;
      byId("fmeta").textContent = `${fmtTime(data.duration)} · ${(finiteNumber(data.sample_rate) / 1000).toFixed(1)} kHz · ${channelLabel}`;
      wsLoaded = makePlayer("#waveform", endpointPath("audio", sessionId), byId("pp-orig"), byId("time-orig"), "orig");
      byId("pp-orig").onclick = () => { if (wsLoaded) wsLoaded.playPause(); };

      ["analyze-panel", "restore-panel", "separate-panel", "transcribe-panel", "loudness-panel", "ab-section"]
        .forEach((id) => byId(id).classList.remove("hidden"));
      applyToolAvailability();
    } catch (error) {
      if (isAbort(error)) return;
      if (context.generation !== generation) return;
      byId("fmeta").textContent = "";
      byId("loaded").classList.add("hidden");
      byId("dropzone").classList.remove("compact");
      setStatus(byId("app-status"), `No se pudo subir: ${error.message}`, {
        base: "app-status", state: "err",
      });
    } finally {
      byId("fileinput").value = "";
    }
  }

  async function cleanAudio() {
    if (!sessionId) return;
    const context = contextSnapshot();
    const button = byId("clean-btn");
    button.disabled = true;
    beginOp();
    const stopElapsed = startElapsed(byId("clean-status"), `Procesando (${quality})…`, context, {
      suffix: quality === "alta" ? " — puede tardar varios minutos" : "",
    });
    try {
      const form = new FormData();
      form.append("quality", quality);
      form.append("source", "original");
      const data = await requestJson(endpointPath("clean", sessionId), { method: "POST", body: form }, "clean", context);
      artifacts.clean = true;
      setStatus(byId("clean-status"), `Listo con ${safeText(data.engine, "el motor de limpieza")} en ${finiteNumber(data.proc_seconds)}s (${safeText(data.device, "cpu").toUpperCase()}).`, {
        state: "ok",
      });
      showComparison(context);
    } catch (error) {
      if (!isAbort(error) && isActive(context)) setStatus(byId("clean-status"), `Error: ${error.message}`, { state: "err" });
    } finally {
      stopElapsed();
      endOp();
      if (isActive(context)) button.disabled = false;
    }
  }

  async function analyzeSounds() {
    if (!sessionId) return;
    const context = contextSnapshot();
    const button = byId("analyze-btn");
    button.disabled = true;
    byId("analyze-empty").classList.add("hidden");
    const list = byId("sounds-list");
    list.replaceChildren();
    lastAnalyzeSource = "original";
    const form = new FormData();
    form.append("source", lastAnalyzeSource);
    beginOp();
    const stopElapsed = startElapsed(byId("analyze-status"), "Analizando los sonidos del audio…", context, {
      base: "analyze-status",
      interval: 350,
    });
    try {
      const data = await requestJson(endpointPath("analyze", sessionId), { method: "POST", body: form }, "analyze", context);
      lastAnalyzeSource = safeText(data.source, lastAnalyzeSource);
      renderSounds(data, context);
    } catch (error) {
      if (!isAbort(error) && isActive(context)) {
        setStatus(byId("analyze-status"), `Error: ${error.message}`, { base: "analyze-status", state: "err" });
      }
    } finally {
      stopElapsed();
      endOp();
      if (isActive(context)) button.disabled = false;
    }
  }

  function renderSounds(data, context) {
    if (!isActive(context)) return;
    const list = byId("sounds-list");
    list.replaceChildren();
    lastSounds = Array.isArray(data.sounds) ? data.sounds : [];
    if (!lastSounds.length) {
      setStatus(byId("analyze-status"), "No se detectaron sonidos destacados en este audio.", { base: "analyze-status" });
      return;
    }
    setStatus(
      byId("analyze-status"),
      `${lastSounds.length} sonidos detectados · ${finiteNumber(data.proc_seconds)}s con PANNs Cnn14 (AudioSet).`,
      { base: "analyze-status" },
    );

    lastSounds.forEach((sound, index) => {
      const labelEs = safeText(sound.label_es, "Sonido");
      const labelEn = safeText(sound.label);
      const confidence = normalizedPercent(sound.confidence);
      const presence = normalizedPercent(sound.presence);
      const card = createElement("article", "sound");
      const top = createElement("div", "top");
      const name = createElement("div", "nm", labelEs);
      if (labelEn && labelEn !== labelEs) name.append(createElement("span", "en", labelEn));
      top.append(name, createElement("div", "pct", `${confidence}%`));
      const bar = createElement("div", "bar");
      const progress = createElement("progress");
      progress.max = 100;
      progress.value = confidence;
      progress.setAttribute("aria-label", `Confianza: ${confidence}%`);
      bar.append(progress);
      const segments = Array.isArray(sound.segments) ? sound.segments : [];
      const subtitle = createElement(
        "div", "sub",
        `Presente el ${presence}% del tiempo${segments.length ? " · aparece en:" : ""}`,
      );
      const chips = createElement("div", "segs");
      segments.forEach((segment) => {
        chips.append(createElement("span", "seg-chip", `${fmtTime(segment.start)}–${fmtTime(segment.end)}`));
      });
      const isolate = createElement("button", "btn isolate-btn", "Aislar este sonido");
      isolate.type = "button";
      isolate.disabled = !hardware.loaded || !hardware.audiosep_available;
      isolate.addEventListener("click", () => void isolateSound(index, isolate));
      card.append(top, bar, subtitle, chips, isolate);
      list.append(card);
    });

    const note = byId("isolate-note");
    if (!hardware.loaded) {
      setStatus(note, "Comprobando disponibilidad de AudioSep…", { base: "mix-note" });
    } else if (!hardware.audiosep_available) {
      setStatus(note, "Aviso: el aislamiento por IA no está instalado en este equipo.", { base: "mix-note", state: "warn" });
    } else {
      const audioSepDevice = hardware.engines
        && hardware.engines.audiosep
        && hardware.engines.audiosep.worker_health
        && hardware.engines.audiosep.worker_health.device;
      if (audioSepDevice && audioSepDevice.used === "cpu") {
        const reason = safeText(audioSepDevice.fallback_reason);
        setStatus(note, `AudioSep usará CPU; puede tardar varios minutos.${reason ? ` ${reason}.` : ""}`, { base: "mix-note" });
      } else {
        setStatus(note, "", { base: "mix-note" });
      }
    }
  }

  async function isolateSound(index, button) {
    if (!sessionId || !lastSounds[index] || !hardware.audiosep_available) return;
    const context = contextSnapshot();
    const sound = lastSounds[index];
    const originalLabel = button.textContent;
    button.disabled = true;
    const started = Date.now();
    const update = () => {
      if (!isActive(context)) return;
      const elapsed = Math.floor((Date.now() - started) / 1000);
      setButtonBusy(button, `Aislando «${safeText(sound.label_es, "sonido")}»… ${elapsed}s`, true);
    };
    update();
    const timer = window.setInterval(update, 400);
    timers.add(timer);
    beginOp();
    let succeeded = false;
    try {
      const form = new FormData();
      form.append("query", safeText(sound.label));
      form.append("label", safeText(sound.label_es, "Sonido aislado"));
      form.append("source", lastAnalyzeSource);
      const data = await requestJson(endpointPath("isolate", sessionId), { method: "POST", body: form }, `isolate:${index}`, context);
      addStem(data.stem, true);
      setButtonBusy(button, `Aislado (${finiteNumber(data.proc_seconds)}s) — añadido a las pistas`, false);
      succeeded = true;
    } catch (error) {
      if (!isAbort(error) && isActive(context)) {
        setStatus(byId("isolate-note"), `Error al aislar: ${error.message}`, { base: "mix-note", state: "warn" });
      }
    } finally {
      window.clearInterval(timer);
      timers.delete(timer);
      endOp();
      if (isActive(context) && !succeeded) {
        setButtonBusy(button, originalLabel, false);
        button.disabled = false;
      }
    }
  }

  function clearPreviewUi() {
    document.querySelectorAll(".stem .solo.playing").forEach((button) => {
      button.classList.remove("playing");
      button.textContent = "▶ Sola";
    });
    byId("mix-play").textContent = "▶ Escuchar mezcla";
    byId("restore-play").textContent = "▶ Escuchar restaurado";
    byId("loudness-play").textContent = "▶ Escuchar";
    previewKey = null;
  }

  function playPreview(url, key, onStart) {
    if (previewKey === key && !previewAudio.paused) {
      previewAudio.pause();
      clearPreviewUi();
      return;
    }
    previewAudio.pause();
    clearPreviewUi();
    previewAudio.src = url;
    previewKey = key;
    previewAudio.play().then(() => {
      if (onStart) onStart();
    }).catch((error) => {
      clearPreviewUi();
      setStatus(byId("app-status"), `No se pudo reproducir el audio: ${error.message}`, {
        base: "app-status", state: "err",
      });
    });
  }

  function createStemRow(stem, isolated) {
    const name = safeText(stem && stem.name);
    if (!name) return null;
    const label = safeText(stem.label_es, name);
    const row = createElement("div", "stem");
    row.dataset.stem = name;
    const checkbox = createElement("input", "chk");
    checkbox.type = "checkbox";
    checkbox.checked = true;
    checkbox.setAttribute("aria-label", `Incluir pista ${label}`);
    const stemName = createElement("div", "snm", label);
    const sublabel = isolated
      ? `IA · «${safeText(stem.query, label)}»`
      : (label !== name ? name : "");
    if (sublabel) stemName.append(createElement("span", "en", sublabel));
    if (isolated) stemName.append(createElement("span", "tag-iso", "aislado"));
    const range = createElement("input");
    range.type = "range";
    range.min = "0";
    range.max = "200";
    range.step = "5";
    range.value = "100";
    range.setAttribute("aria-label", `Volumen de ${label}`);
    const volume = createElement("output", "vol", "100%");
    const solo = createElement("button", "btn solo", "▶ Sola");
    solo.type = "button";
    checkbox.addEventListener("change", () => row.classList.toggle("off", !checkbox.checked));
    range.addEventListener("input", () => { volume.textContent = `${range.value}%`; });
    solo.addEventListener("click", () => {
      if (!sessionId) return;
      playPreview(endpointPath("stem", sessionId, name), `stem:${name}`, () => {
        solo.classList.add("playing");
        solo.textContent = "⏸ Sola";
      });
    });
    row.append(checkbox, stemName, range, volume, solo);
    return row;
  }

  function showMixArea() {
    byId("separate-panel").classList.remove("hidden");
    byId("mix-bar").classList.remove("hidden");
    setStatus(byId("mix-note"), "Ajusta cada pista y pulsa «Escuchar mezcla» o «Exportar».", { base: "mix-note" });
  }

  function renderStems(data) {
    clearPreviewUi();
    availableStemNames = new Set();
    const box = byId("stems");
    box.replaceChildren();
    const stems = Array.isArray(data.stems) ? data.stems : [];
    stems.forEach((stem) => {
      const row = createStemRow(stem, Boolean(stem.isolated || stem.query));
      if (!row) return;
      availableStemNames.add(row.dataset.stem);
      box.append(row);
    });
    showMixArea();
  }

  function addStem(stem, isolated) {
    const row = createStemRow(stem, isolated);
    if (!row) return;
    const box = byId("stems");
    Array.from(box.children).forEach((candidate) => {
      if (candidate.dataset.stem === row.dataset.stem) candidate.remove();
    });
    availableStemNames.add(row.dataset.stem);
    box.append(row);
    showMixArea();
    row.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  async function separateAudio() {
    if (!sessionId) return;
    const context = contextSnapshot();
    const button = byId("separate-btn");
    button.disabled = true;
    byId("separate-empty").classList.add("hidden");
    const source = artifacts.restore ? "restore" : artifacts.clean ? "clean" : "original";
    beginOp();
    const stopElapsed = startElapsed(byId("separate-status"), "Separando voz y fondo…", context, {
      suffix: " — Demucs puede tardar en CPU",
    });
    try {
      const form = new FormData();
      form.append("source", source);
      const data = await requestJson(endpointPath("separate", sessionId), { method: "POST", body: form }, "separate", context);
      setStatus(byId("separate-status"), `Voz y fondo separados en ${finiteNumber(data.proc_seconds)}s (${safeText(data.device, "cpu").toUpperCase()}) desde ${safeText(data.source, source)}.`, {
        state: "ok",
      });
      renderStems(data);
    } catch (error) {
      if (!isAbort(error) && isActive(context)) setStatus(byId("separate-status"), `Error: ${error.message}`, { state: "err" });
    } finally {
      stopElapsed();
      endOp();
      if (isActive(context)) button.disabled = false;
    }
  }

  function collectTracks() {
    return Array.from(document.querySelectorAll("#stems .stem")).map((row) => ({
      name: row.dataset.stem,
      enabled: row.querySelector(".chk").checked,
      gain: clamp(row.querySelector('input[type="range"]').value / 100, 0, 2),
    }));
  }

  async function mixNow(context) {
    const data = await requestJson(endpointPath("mix", sessionId), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tracks: collectTracks() }),
    }, "mix", context);
    artifacts.mix = true;
    return data;
  }

  function showMixNote(result, extra = "") {
    const clipped = Boolean(result && result.clipped);
    const prefix = clipped ? "Aviso: la mezcla satura. Baja algún volumen. " : "";
    const message = extra || (result ? `Mezcla lista (pico ${Math.round(finiteNumber(result.peak) * 100)}%).` : "");
    setStatus(byId("mix-note"), prefix + message, { base: "mix-note", state: clipped ? "warn" : "" });
  }

  async function previewMix() {
    if (!sessionId) return;
    if (previewKey === "mix" && !previewAudio.paused) {
      previewAudio.pause();
      clearPreviewUi();
      return;
    }
    const context = contextSnapshot();
    setStatus(byId("mix-note"), "Preparando mezcla…", { base: "mix-note", busy: true });
    beginOp();
    try {
      const result = await mixNow(context);
      showMixNote(result);
      playPreview(`${endpointPath("mix-audio", sessionId)}?t=${Date.now()}`, "mix", () => {
        byId("mix-play").textContent = "⏸ Escuchar mezcla";
      });
    } catch (error) {
      if (!isAbort(error) && isActive(context)) setStatus(byId("mix-note"), `Error: ${error.message}`, { base: "mix-note", state: "warn" });
    } finally {
      endOp();
    }
  }

  function triggerDownload(url, filename) {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
  }

  async function exportMix(format) {
    if (!sessionId) return;
    const context = contextSnapshot();
    setStatus(byId("mix-note"), `Preparando ${format.toUpperCase()}…`, { base: "mix-note", busy: true });
    beginOp();
    try {
      const result = await mixNow(context);
      triggerDownload(`${endpointPath("export-mix", sessionId)}?format=${format}&t=${Date.now()}`, `mezcla.${format}`);
      showMixNote(result, `Descargando mezcla.${format}…`);
    } catch (error) {
      if (!isAbort(error) && isActive(context)) setStatus(byId("mix-note"), `Error: ${error.message}`, { base: "mix-note", state: "warn" });
    } finally {
      endOp();
    }
  }

  async function restoreAudio() {
    if (!sessionId) return;
    const context = contextSnapshot();
    const button = byId("restore-btn");
    button.disabled = true;
    byId("restore-empty").classList.add("hidden");
    beginOp();
    const stopElapsed = startElapsed(byId("restore-status"), "Restaurando…", context, {
      suffix: " — puede tardar en CPU",
    });
    try {
      const form = new FormData();
      form.append("mode", byId("restore-mode").value);
      form.append("source", artifacts.clean ? "clean" : "original");
      const data = await requestJson(endpointPath("restore", sessionId), { method: "POST", body: form }, "restore", context);
      artifacts.restore = true;
      setStatus(byId("restore-status"), `Voz restaurada en ${finiteNumber(data.proc_seconds)}s (${safeText(data.device, "cpu").toUpperCase()}) desde ${safeText(data.source, "original")}.`, {
        state: "ok",
      });
      byId("restore-bar").classList.remove("hidden");
      setDownloadLink("restore-wav", `${endpointPath("export-restore", sessionId)}?format=wav`, "restaurado.wav");
      setDownloadLink("restore-mp3", `${endpointPath("export-restore", sessionId)}?format=mp3`, "restaurado.mp3");
    } catch (error) {
      if (!isAbort(error) && isActive(context)) setStatus(byId("restore-status"), `Error: ${error.message}`, { state: "err" });
    } finally {
      stopElapsed();
      endOp();
      if (isActive(context)) button.disabled = !hardware.restore_available;
    }
  }

  async function transcribeAudio() {
    if (!sessionId) return;
    const context = contextSnapshot();
    const button = byId("transcribe-btn");
    const model = byId("whisper-model").value;
    const source = bestSpeechSource(availableStemNames, artifacts);
    button.disabled = true;
    byId("transcribe-empty").classList.add("hidden");
    beginOp();
    const stopElapsed = startElapsed(byId("transcribe-status"), `Transcribiendo (${model})…`, context, {
      suffix: " — puede tardar en CPU",
    });
    try {
      const form = new FormData();
      form.append("model", model);
      form.append("language", byId("whisper-lang").value);
      form.append("source", source);
      const data = await requestJson(endpointPath("transcribe", sessionId), { method: "POST", body: form }, "transcribe", context);
      setStatus(byId("transcribe-status"), `${finiteNumber(data.n_segments)} segmentos · idioma ${safeText(data.language, "desconocido")} · ${finiteNumber(data.proc_seconds)}s · fuente ${safeText(data.source, source)}.`, {
        state: "ok",
      });
      renderTranscript(data);
    } catch (error) {
      if (!isAbort(error) && isActive(context)) setStatus(byId("transcribe-status"), `Error: ${error.message}`, { state: "err" });
    } finally {
      stopElapsed();
      endOp();
      if (isActive(context)) button.disabled = !hardware.whisper_available;
    }
  }

  function renderTranscript(data) {
    const preview = byId("sub-preview");
    preview.replaceChildren();
    preview.classList.remove("hidden");
    const segments = Array.isArray(data.segments) ? data.segments : [];
    segments.forEach((segment) => {
      const row = createElement("div");
      row.append(
        createElement("span", "cue", fmtTime(segment.start)),
        document.createTextNode(safeText(segment.text)),
      );
      preview.append(row);
    });
    if (finiteNumber(data.n_segments) > segments.length) {
      preview.append(createElement("div", "cue", "… descarga el archivo para verlo completo"));
    }
    byId("sub-bar").classList.remove("hidden");
    setDownloadLink("sub-srt", `${endpointPath("subtitles", sessionId)}?format=srt`, "subtitulos.srt");
    setDownloadLink("sub-vtt", `${endpointPath("subtitles", sessionId)}?format=vtt`, "subtitulos.vtt");
    setDownloadLink("sub-txt", `${endpointPath("subtitles", sessionId)}?format=txt`, "subtitulos.txt");
  }

  async function normalizeLoudness() {
    if (!sessionId) return;
    const context = contextSnapshot();
    const button = byId("loudness-btn");
    button.disabled = true;
    byId("loudness-empty").classList.add("hidden");
    beginOp();
    const stopElapsed = startElapsed(byId("loudness-status"), "Normalizando…", context);
    try {
      const form = new FormData();
      form.append("preset", byId("loudness-preset").value);
      form.append("source", "auto");
      const data = await requestJson(endpointPath("loudness", sessionId), { method: "POST", body: form }, "loudness", context);
      artifacts.loudness = true;
      setStatus(byId("loudness-status"), `${safeText(data.preset_label, "Normalizado")} · fuente ${safeText(data.source, "auto")} · ${finiteNumber(data.proc_seconds)}s.`, {
        state: "ok",
      });
      renderLoudness(data);
    } catch (error) {
      if (!isAbort(error) && isActive(context)) setStatus(byId("loudness-status"), `Error: ${error.message}`, { state: "err" });
    } finally {
      stopElapsed();
      endOp();
      if (isActive(context)) button.disabled = false;
    }
  }

  function labeledMetric(label, value, emphasized) {
    const container = createElement("span");
    container.append(document.createTextNode(`${label}: `));
    container.append(createElement(emphasized ? "b" : "span", "", value));
    return container;
  }

  function renderLoudness(data) {
    const metrics = byId("lufs-bar");
    metrics.replaceChildren(
      labeledMetric("Entrada", `${finiteNumber(data.input_lufs)} LUFS`, true),
      labeledMetric("Salida", `${finiteNumber(data.output_lufs)} LUFS`, true),
      labeledMetric("Objetivo", `${finiteNumber(data.target_lufs)} LUFS`, false),
    );
    metrics.classList.remove("hidden");
    byId("loudness-bar").classList.remove("hidden");
    setDownloadLink("loudness-wav", `${endpointPath("export-loudness", sessionId)}?format=wav`, "normalizado.wav");
    setDownloadLink("loudness-mp3", `${endpointPath("export-loudness", sessionId)}?format=mp3`, "normalizado.mp3");
  }

  function setDownloadLink(id, url, filename) {
    const link = byId(id);
    link.href = url;
    link.download = filename;
    link.removeAttribute("aria-disabled");
  }

  function showComparison(context) {
    if (!isActive(context)) return;
    byId("ab-section").classList.remove("hidden");
    byId("ab-empty").classList.add("hidden");
    byId("ab-slot").classList.remove("hidden");
    wsA = destroyWave(wsA);
    wsB = destroyWave(wsB);
    const cacheBust = `?t=${Date.now()}`;
    wsA = makePlayer("#wave-a", endpointPath("audio", sessionId), byId("pp-a"), null, "orig");
    wsB = makePlayer("#wave-b", endpointPath("clean-audio", sessionId) + cacheBust, byId("pp-b"), null, "clean");
    byId("pp-a").textContent = "▶ Original";
    byId("pp-b").textContent = "▶ Limpio";
    byId("pp-a").onclick = () => { if (wsB) wsB.pause(); if (wsA) wsA.playPause(); };
    byId("pp-b").onclick = () => { if (wsA) wsA.pause(); if (wsB) wsB.playPause(); };
    setDownloadLink("dl-wav", `${endpointPath("download", sessionId)}?format=wav`, "limpio.wav");
    setDownloadLink("dl-mp3", `${endpointPath("download", sessionId)}?format=mp3`, "limpio.mp3");
    byId("ab-section").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function wireEvents() {
    document.querySelectorAll(".preset-btn").forEach((button) => {
      button.addEventListener("click", () => applyPreset(button.dataset.preset));
    });
    document.querySelectorAll("#ab-toggle button").forEach((button) => {
      button.addEventListener("click", () => {
        const which = button.dataset.ab;
        document.querySelectorAll("#ab-toggle button").forEach((candidate) => {
          candidate.setAttribute("aria-pressed", candidate === button ? "true" : "false");
        });
        byId("ab-col-a").classList.toggle("hidden", which !== "a");
        byId("ab-col-b").classList.toggle("hidden", which !== "b");
        if (which === "a" && wsB) wsB.pause();
        if (which === "b" && wsA) wsA.pause();
      });
    });
    window.addEventListener("pagehide", () => {
      const previousSessionId = sessionId;
      sessionId = null;
      releaseSession(previousSessionId);
    });
    document.querySelectorAll(".seg").forEach((segment) => {
      segment.addEventListener("click", () => { if (!segment.disabled) selectQuality(segment.dataset.q); });
    });

    const dropzone = byId("dropzone");
    dropzone.addEventListener("click", () => byId("fileinput").click());
    byId("fileinput").addEventListener("change", (event) => {
      if (event.target.files[0]) void uploadFile(event.target.files[0]);
    });
    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("drag");
      });
    });
    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove("drag");
      });
    });
    dropzone.addEventListener("drop", (event) => {
      const file = event.dataTransfer.files[0];
      if (file) void uploadFile(file);
    });

    byId("clean-btn").addEventListener("click", () => void cleanAudio());
    byId("analyze-btn").addEventListener("click", () => void analyzeSounds());
    byId("separate-btn").addEventListener("click", () => void separateAudio());
    byId("mix-play").addEventListener("click", () => void previewMix());
    byId("mix-wav").addEventListener("click", (event) => { event.preventDefault(); void exportMix("wav"); });
    byId("mix-mp3").addEventListener("click", (event) => { event.preventDefault(); void exportMix("mp3"); });
    byId("restore-btn").addEventListener("click", () => void restoreAudio());
    byId("restore-play").addEventListener("click", () => {
      if (sessionId) playPreview(`${endpointPath("restore-audio", sessionId)}?t=${Date.now()}`, "restore", () => {
        byId("restore-play").textContent = "⏸ Escuchar restaurado";
      });
    });
    byId("transcribe-btn").addEventListener("click", () => void transcribeAudio());
    byId("loudness-btn").addEventListener("click", () => void normalizeLoudness());
    byId("loudness-play").addEventListener("click", () => {
      if (sessionId) playPreview(`${endpointPath("loudness-audio", sessionId)}?t=${Date.now()}`, "loudness", () => {
        byId("loudness-play").textContent = "⏸ Escuchar";
      });
    });
    previewAudio.addEventListener("ended", clearPreviewUi);
    previewAudio.addEventListener("error", clearPreviewUi);

    if (window.limpiaraudioDesktop && typeof window.limpiaraudioDesktop.selectProject === "function") {
      const projectButton = byId("project-btn");
      projectButton.classList.remove("hidden");
      projectButton.addEventListener("click", async () => {
        if (!window.confirm("La aplicación se reiniciará después de cambiar la carpeta. ¿Continuar?")) return;
        projectButton.disabled = true;
        try {
          await window.limpiaraudioDesktop.selectProject();
        } catch (error) {
          projectButton.disabled = false;
          setStatus(byId("app-status"), `No se pudo cambiar la carpeta: ${error.message}`, {
            base: "app-status", state: "err",
          });
        }
      });
    }
  }

  wireEvents();
  resetAll();
  void loadHardwareInfo();
}());
