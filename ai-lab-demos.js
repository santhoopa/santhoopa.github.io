/* AI Lab — in-browser ML demos. All models run locally (WASM/WebGPU); nothing leaves the tab. */
(function () {
'use strict';
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];

const TJS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.1';
const MP  = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
let tjsMod = null;
const loadTJS = () => (tjsMod ??= import(TJS));

/* ---------- shared card plumbing ---------- */
function card(prefix) {
  return {
    st:   $('#' + prefix + '-st'),
    gate: $('#' + prefix + '-gate'),
    init: $('#' + prefix + '-init'),
    prog: $('#' + prefix + '-prog'),
    bar:  $('#' + prefix + '-prog .bar i'),
    lbl:  $('#' + prefix + '-prog .lbl'),
    body: $('#' + prefix + '-body'),
    err:  $('#' + prefix + '-err'),
    errMsg: $('#' + prefix + '-err .msg')
  };
}
function setStatus(u, txt, cls) {
  u.st.textContent = txt;
  u.st.className = 'st ' + (cls || '');
}
function beginLoad(u) {
  u.init.disabled = true;
  u.prog.hidden = false;
  setStatus(u, 'LOADING', 'busy');
}
function showBody(u) {
  u.gate.hidden = true;
  u.body.hidden = false;
  setStatus(u, 'READY', 'ok');
}
function fail(u, e) {
  console.warn(e);
  u.gate.hidden = true;
  u.body.hidden = true;
  u.err.hidden = false;
  u.errMsg.textContent = (e && e.message) ? String(e.message).slice(0, 160) : 'Unknown error';
  setStatus(u, 'FAILED', 'bad');
}
/* classic <script> loader for UMD libs (face-api.js, tfjs), cached by URL */
const __scriptCache = new Map();
function loadScript(src) {
  if (__scriptCache.has(src)) return __scriptCache.get(src);
  const p = new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error('failed to load ' + src));
    document.head.appendChild(s);
  });
  __scriptCache.set(src, p);
  return p;
}
/* transformers.js download progress → aggregate bar */
function tjsProgress(u) {
  const files = new Map();
  return p => {
    if (p.status === 'progress' && p.total) files.set(p.file, { loaded: p.loaded, total: p.total });
    if (p.status === 'done' && files.has(p.file)) { const f = files.get(p.file); f.loaded = f.total; }
    let loaded = 0, total = 0;
    files.forEach(f => { loaded += f.loaded; total += f.total; });
    if (total > 0) {
      const pct = Math.min(100, Math.round(loaded / total * 100));
      u.bar.style.width = pct + '%';
      u.lbl.textContent = 'downloading ' + (p.file || '') + ' — ' + pct + '% (' + (loaded / 1048576).toFixed(1) + ' MB)';
    } else {
      u.lbl.textContent = 'fetching model manifest…';
    }
  };
}

/* ================================================================
   01 · WHISPER TINY — speech → text
   ================================================================ */
const asr = card('asr');
let asrPipe = null, mediaRec = null, recChunks = [], recTimer = null, recStart = 0, vuRAF = 0;

asr.init.addEventListener('click', async () => {
  beginLoad(asr);
  try {
    const { pipeline } = await loadTJS();
    asrPipe = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en', {
      dtype: 'q8', progress_callback: tjsProgress(asr)
    });
    showBody(asr);
  } catch (e) { fail(asr, e); }
});

const recBtn = $('#asr-rec');
function stopVU() { cancelAnimationFrame(vuRAF); }
function drawVU(analyser) {
  const cv = $('#asr-vu'), ctx = cv.getContext('2d');
  const data = new Uint8Array(analyser.frequencyBinCount);
  const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
  (function loop() {
    vuRAF = requestAnimationFrame(loop);
    analyser.getByteFrequencyData(data);
    ctx.clearRect(0, 0, cv.width, cv.height);
    const bars = 48, step = Math.floor(data.length / bars);
    for (let i = 0; i < bars; i++) {
      const v = data[i * step] / 255;
      const h = Math.max(2, v * cv.height);
      ctx.fillStyle = accent;
      ctx.globalAlpha = 0.35 + v * 0.65;
      ctx.fillRect(i * (cv.width / bars), (cv.height - h) / 2, cv.width / bars - 2, h);
    }
  })();
}
recBtn.addEventListener('click', async () => {
  if (mediaRec && mediaRec.state === 'recording') { mediaRec.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const actx = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = actx.createAnalyser();
    analyser.fftSize = 256;
    actx.createMediaStreamSource(stream).connect(analyser);
    drawVU(analyser);
    recChunks = [];
    mediaRec = new MediaRecorder(stream);
    mediaRec.ondataavailable = e => recChunks.push(e.data);
    mediaRec.onstop = async () => {
      clearInterval(recTimer); stopVU();
      stream.getTracks().forEach(t => t.stop()); actx.close();
      recBtn.classList.remove('recording');
      recBtn.querySelector('span').textContent = 'Record';
      $('#asr-out').textContent = '…transcribing on-device…';
      setStatus(asr, 'INFER', 'busy');
      try {
        const blob = new Blob(recChunks);
        const dctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        const buf = await dctx.decodeAudioData(await blob.arrayBuffer());
        const audio = buf.getChannelData(0);
        dctx.close();
        const t0 = performance.now();
        const res = await asrPipe(audio);
        const ms = Math.round(performance.now() - t0);
        $('#asr-out').textContent = (res.text || '').trim() || '(silence — nothing detected)';
        $('#asr-meta').textContent = buf.duration.toFixed(1) + 's audio → transcribed in ' + ms + 'ms · on-device';
      } catch (e) {
        $('#asr-out').textContent = 'Transcription failed: ' + e.message;
      }
      setStatus(asr, 'READY', 'ok');
    };
    mediaRec.start();
    recStart = Date.now();
    recBtn.classList.add('recording');
    recBtn.querySelector('span').textContent = 'Stop';
    setStatus(asr, 'REC ●', 'bad');
    recTimer = setInterval(() => {
      const s = (Date.now() - recStart) / 1000;
      $('#asr-timer').textContent = s.toFixed(1) + 's / 30s';
      if (s >= 30 && mediaRec.state === 'recording') mediaRec.stop();
    }, 100);
  } catch (e) {
    $('#asr-out').textContent = 'Microphone unavailable — ' + e.message;
  }
});

/* ================================================================
   02 · GESTURE LINK — hand tracking + rock·paper·scissors
   ================================================================ */
const ges = card('ges');
let recognizer = null, gesStream = null, gesRAF = 0, lastVideoTime = -1, mpMod = null;
let currentGesture = null;

ges.init.addEventListener('click', async () => {
  beginLoad(ges);
  ges.bar.classList.add('indet');
  ges.lbl.textContent = 'downloading gesture model ~8 MB…';
  try {
    mpMod = await import(MP);
    const fs = await mpMod.FilesetResolver.forVisionTasks(MP + '/wasm');
    const opts = delegate => ({
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task',
        delegate
      },
      runningMode: 'VIDEO', numHands: 1
    });
    try { recognizer = await mpMod.GestureRecognizer.createFromOptions(fs, opts('GPU')); }
    catch (_) { recognizer = await mpMod.GestureRecognizer.createFromOptions(fs, opts('CPU')); }
    ges.bar.classList.remove('indet');
    showBody(ges);
  } catch (e) { fail(ges, e); }
});

const camBtn = $('#ges-cam');
camBtn.addEventListener('click', async () => {
  if (gesStream) { stopCam(); return; }
  try {
    gesStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    const video = $('#ges-video');
    video.srcObject = gesStream;
    await video.play();
    const cv = $('#ges-canvas');
    cv.width = video.videoWidth; cv.height = video.videoHeight;
    camBtn.querySelector('span').textContent = 'Stop camera';
    $('#ges-stage').classList.add('live');
    setStatus(ges, 'LIVE ●', 'ok');
    gesLoop();
  } catch (e) {
    $('#ges-label').textContent = 'camera unavailable — ' + e.message;
  }
});
function stopCam() {
  cancelAnimationFrame(gesRAF);
  if (gesStream) gesStream.getTracks().forEach(t => t.stop());
  gesStream = null;
  currentGesture = null;
  $('#ges-stage').classList.remove('live');
  camBtn.querySelector('span').textContent = 'Start camera';
  $('#ges-label').textContent = '— camera off —';
  $$('#ges-strip span').forEach(s => s.classList.remove('on'));
  setStatus(ges, 'READY', 'ok');
}
function gesLoop() {
  const video = $('#ges-video'), cv = $('#ges-canvas'), ctx = cv.getContext('2d');
  const du = new mpMod.DrawingUtils(ctx);
  const accent = () => getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
  (function loop() {
    gesRAF = requestAnimationFrame(loop);
    if (!gesStream || video.readyState < 2) return;
    if (video.currentTime === lastVideoTime) return;
    lastVideoTime = video.currentTime;
    const res = recognizer.recognizeForVideo(video, performance.now());
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (res.landmarks && res.landmarks.length) {
      for (const lm of res.landmarks) {
        du.drawConnectors(lm, mpMod.GestureRecognizer.HAND_CONNECTIONS, { color: accent(), lineWidth: 2 });
        du.drawLandmarks(lm, { color: '#ffffff', lineWidth: 1, radius: 2.5 });
      }
    }
    const g = res.gestures && res.gestures[0] && res.gestures[0][0];
    if (g && g.categoryName && g.categoryName !== 'None') {
      currentGesture = g.categoryName;
      $('#ges-label').textContent = g.categoryName.replace(/_/g, ' ') + ' · ' + Math.round(g.score * 100) + '%';
    } else {
      currentGesture = null;
      $('#ges-label').textContent = 'show a hand to the camera…';
    }
    $$('#ges-strip span').forEach(s => s.classList.toggle('on', s.dataset.g === currentGesture));
  })();
}

/* ----- rock · paper · scissors ----- */
const RPS_MAP = { Closed_Fist: 'rock', Open_Palm: 'paper', Victory: 'scissors' };
const RPS_EMO = { Closed_Fist: '✊️', Open_Palm: '✋️', Victory: '✌️' };
const RPS_BEATS = { Closed_Fist: 'Victory', Victory: 'Open_Palm', Open_Palm: 'Closed_Fist' };
let rpsScore = { you: 0, bot: 0 }, counting = false;

/* tiny WebAudio beep — no audio files needed */
let beepCtx = null;
function beep(freq, dur, type, vol) {
  try {
    beepCtx = beepCtx || new (window.AudioContext || window.webkitAudioContext)();
    const osc = beepCtx.createOscillator(), gain = beepCtx.createGain();
    osc.type = type || 'square';
    osc.frequency.value = freq || 660;
    gain.gain.value = vol != null ? vol : 0.05;
    osc.connect(gain); gain.connect(beepCtx.destination);
    osc.start();
    gain.gain.exponentialRampToValueAtTime(0.0001, beepCtx.currentTime + (dur || 0.09));
    osc.stop(beepCtx.currentTime + (dur || 0.09) + 0.02);
  } catch (_) {}
}

$('#rps-play').addEventListener('click', () => {
  if (counting) return;
  if (!gesStream) { $('#rps-result').textContent = 'start the camera first'; return; }
  counting = true;
  let n = 3;
  const cd = $('#ges-count');
  cd.textContent = n; cd.classList.add('on');
  beep(660, 0.09, 'square', 0.05);
  $('#rps-result').textContent = 'get your hand ready…';
  const iv = setInterval(() => {
    n--;
    if (n > 0) { cd.textContent = n; beep(660, 0.09, 'square', 0.05); return; }
    clearInterval(iv);
    cd.classList.remove('on');
    counting = false;
    beep(1046, 0.13, 'square', 0.06);
    resolveRound();
  }, 800);
});
function resolveRound() {
  const you = currentGesture;
  if (!you || !RPS_MAP[you]) {
    $('#rps-you').textContent = '?';
    $('#rps-bot').textContent = '—';
    $('#rps-result').textContent = 'no valid gesture — show ✊️ ✋️ or ✌️';
    return;
  }
  const opts = Object.keys(RPS_MAP);
  const bot = opts[Math.floor(Math.random() * 3)];
  $('#rps-you').textContent = RPS_EMO[you];
  $('#rps-bot').textContent = RPS_EMO[bot];
  let msg;
  if (you === bot) { msg = 'draw — ' + RPS_MAP[you] + ' vs ' + RPS_MAP[bot]; beep(500, 0.14, 'triangle', 0.05); }
  else if (RPS_BEATS[you] === bot) { rpsScore.you++; msg = RPS_MAP[you] + ' beats ' + RPS_MAP[bot] + ' — you win'; beep(1200, 0.18, 'square', 0.06); }
  else { rpsScore.bot++; msg = RPS_MAP[bot] + ' beats ' + RPS_MAP[you] + ' — model wins'; beep(180, 0.22, 'sawtooth', 0.06); }
  $('#rps-result').textContent = msg;
  $('#rps-score').textContent = 'YOU ' + rpsScore.you + ' — ' + rpsScore.bot + ' MODEL';
}

/* ================================================================
   03 · OBJECT DETECTION — EfficientDet-Lite0 on the webcam
   ================================================================ */
const det = card('det');
let detector = null, detStream = null, detRAF = 0, detLastTime = -1, detListAt = 0;
let fpsCount = 0, fpsAt = 0;

det.init.addEventListener('click', async () => {
  beginLoad(det);
  det.bar.classList.add('indet');
  det.lbl.textContent = 'downloading detector ~5 MB…';
  try {
    mpMod = mpMod || await import(MP);
    const fs = await mpMod.FilesetResolver.forVisionTasks(MP + '/wasm');
    const opts = delegate => ({
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
        delegate
      },
      runningMode: 'VIDEO', scoreThreshold: 0.45, maxResults: 6
    });
    try { detector = await mpMod.ObjectDetector.createFromOptions(fs, opts('GPU')); }
    catch (_) { detector = await mpMod.ObjectDetector.createFromOptions(fs, opts('CPU')); }
    det.bar.classList.remove('indet');
    showBody(det);
  } catch (e) { fail(det, e); }
});

const detCamBtn = $('#det-cam');
detCamBtn.addEventListener('click', async () => {
  if (detStream) { stopDetCam(); return; }
  try {
    detStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    const video = $('#det-video');
    video.srcObject = detStream;
    await video.play();
    const cv = $('#det-canvas');
    cv.width = video.videoWidth; cv.height = video.videoHeight;
    detCamBtn.querySelector('span').textContent = 'Stop camera';
    $('#det-stage').classList.add('live');
    setStatus(det, 'LIVE ●', 'ok');
    detLoop();
  } catch (e) {
    $('#det-list').innerHTML = '<div class="det-none">camera unavailable — ' + e.message + '</div>';
  }
});
function stopDetCam() {
  cancelAnimationFrame(detRAF);
  if (detStream) detStream.getTracks().forEach(t => t.stop());
  detStream = null;
  $('#det-stage').classList.remove('live');
  detCamBtn.querySelector('span').textContent = 'Start camera';
  $('#det-list').innerHTML = '<div class="det-none">— camera off —</div>';
  $('#det-fps').textContent = '';
  setStatus(det, 'READY', 'ok');
}
function detLoop() {
  const video = $('#det-video'), cv = $('#det-canvas'), ctx = cv.getContext('2d');
  (function loop() {
    detRAF = requestAnimationFrame(loop);
    if (!detStream || video.readyState < 2) return;
    if (video.currentTime === detLastTime) return;
    detLastTime = video.currentTime;
    const res = detector.detectForVideo(video, performance.now());
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.font = '600 13px "JetBrains Mono", monospace';
    ctx.lineWidth = 2;
    const dets = res.detections || [];
    for (const d of dets) {
      const b = d.boundingBox, c = d.categories[0];
      ctx.strokeStyle = accent;
      ctx.strokeRect(b.originX, b.originY, b.width, b.height);
      const label = c.categoryName + ' ' + Math.round(c.score * 100) + '%';
      const tw = ctx.measureText(label).width + 12;
      const ly = Math.max(0, b.originY - 20);
      ctx.fillStyle = accent;
      ctx.fillRect(b.originX - 1, ly, tw, 20);
      ctx.fillStyle = '#000';
      ctx.fillText(label, b.originX + 5, ly + 14);
    }
    /* fps */
    fpsCount++;
    const now = performance.now();
    if (now - fpsAt > 1000) {
      $('#det-fps').textContent = fpsCount + ' fps · on-device';
      fpsCount = 0; fpsAt = now;
    }
    /* side list, throttled */
    if (now - detListAt > 180) {
      detListAt = now;
      const listEl = $('#det-list');
      if (!dets.length) {
        listEl.innerHTML = '<div class="det-none">nothing detected — point the camera at objects</div>';
      } else {
        const agg = new Map();
        for (const d of dets) {
          const c = d.categories[0];
          const e = agg.get(c.categoryName) || { n: 0, s: 0 };
          e.n++; e.s = Math.max(e.s, c.score);
          agg.set(c.categoryName, e);
        }
        listEl.innerHTML = '';
        [...agg.entries()].sort((a, b) => b[1].s - a[1].s).forEach(([name, e]) => {
          const row = document.createElement('div');
          row.className = 'det-row';
          row.innerHTML =
            '<div class="det-row-top"><span class="nm"></span><span class="sc">' + (e.n > 1 ? '×' + e.n + ' · ' : '') + Math.round(e.s * 100) + '%</span></div>' +
            '<div class="hit-bar"><i style="width:' + Math.round(e.s * 100) + '%"></i></div>';
          row.querySelector('.nm').textContent = name;
          listEl.appendChild(row);
        });
      }
    }
  })();
}

/* ================================================================
   04 · FACE READER — face-api.js (tiny detector + expression + age/gender)
   ================================================================ */
const face = card('face');
let faceStream = null, faceRAF = 0;
const FACE_EMOJI = { neutral: '😐', happy: '😄', sad: '😢', angry: '😠', fearful: '😨', disgusted: '🤢', surprised: '😲' };

face.init.addEventListener('click', async () => {
  beginLoad(face);
  face.bar.classList.add('indet');
  face.lbl.textContent = 'loading face-api.js + models ~1.5 MB…';
  try {
    await loadScript('https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js');
    const MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights';
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
      faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL)
    ]);
    face.bar.classList.remove('indet');
    showBody(face);
  } catch (e) { fail(face, e); }
});

const faceCamBtn = $('#face-cam');
faceCamBtn.addEventListener('click', async () => {
  if (faceStream) { stopFaceCam(); return; }
  try {
    faceStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    const video = $('#face-video');
    video.srcObject = faceStream;
    await video.play();
    const cv = $('#face-canvas');
    cv.width = video.videoWidth; cv.height = video.videoHeight;
    faceCamBtn.querySelector('span').textContent = 'Stop camera';
    $('#face-stage').classList.add('live');
    setStatus(face, 'LIVE ●', 'ok');
    faceLoop();
  } catch (e) {
    $('#face-readout').innerHTML = '<div class="det-none">camera unavailable — ' + e.message + '</div>';
  }
});
function stopFaceCam() {
  cancelAnimationFrame(faceRAF);
  if (faceStream) faceStream.getTracks().forEach(t => t.stop());
  faceStream = null;
  $('#face-stage').classList.remove('live');
  faceCamBtn.querySelector('span').textContent = 'Start camera';
  $('#face-readout').innerHTML = '<div class="det-none">— camera off —</div>';
  setStatus(face, 'READY', 'ok');
}
function faceLoop() {
  const video = $('#face-video'), cv = $('#face-canvas'), ctx = cv.getContext('2d');
  const opts = new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 });
  let busy = false;
  (function loop() {
    faceRAF = requestAnimationFrame(loop);
    if (!faceStream || video.readyState < 2 || busy) return;
    busy = true;
    faceapi.detectSingleFace(video, opts).withFaceExpressions().withAgeAndGender()
      .then(res => {
        busy = false;
        if (!faceStream) return;
        const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
        ctx.clearRect(0, 0, cv.width, cv.height);
        if (!res) {
          $('#face-readout').innerHTML = '<div class="det-none">show your face to the camera…</div>';
          return;
        }
        const b = res.detection.box;
        ctx.strokeStyle = accent; ctx.lineWidth = 2;
        ctx.strokeRect(b.x, b.y, b.width, b.height);
        const top = Object.entries(res.expressions).sort((a, b) => b[1] - a[1])[0];
        const emoji = FACE_EMOJI[top[0]] || '🙂';
        const age = Math.round(res.age);
        const genderLabel = res.genderProbability > 0.6 ? res.gender : 'uncertain';
        $('#face-readout').innerHTML =
          '<div class="face-emoji">' + emoji + '</div>' +
          '<div class="face-lines">' +
            '<div class="face-line"><span class="k">expression</span><span class="v">' + top[0] + ' · ' + Math.round(top[1] * 100) + '%</span></div>' +
            '<div class="face-line"><span class="k">age (est.)</span><span class="v">~' + age + ' yrs</span></div>' +
            '<div class="face-line"><span class="k">gender (est.)</span><span class="v">' + genderLabel + ' · ' + Math.round(res.genderProbability * 100) + '%</span></div>' +
          '</div>';
      })
      .catch(() => { busy = false; });
  })();
}

/* ================================================================
   05 · CARTOONIZE — White-box CartoonGAN (TensorFlow.js)
   ================================================================ */
const cart = card('cart');
let cartModel = null, cartRetried = false, cartRawImageData = null;

/* posterize + saturation boost + Sobel-edge ink overlay, layered on top of the raw GAN output */
function cartApplyStrength(strengthPct) {
  if (!cartRawImageData) return;
  const s = Math.max(0, Math.min(100, strengthPct)) / 100;
  const w = cartRawImageData.width, h = cartRawImageData.height;
  const src = cartRawImageData.data;
  const lum = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    lum[i] = 0.299 * src[i * 4] + 0.587 * src[i * 4 + 1] + 0.114 * src[i * 4 + 2];
  }
  const out = new Uint8ClampedArray(src.length);
  const levels = Math.max(3, Math.round(10 - s * 6));
  const step = 255 / (levels - 1);
  const satMul = 1 + s * 0.9;
  for (let i = 0; i < w * h; i++) {
    const p = i * 4;
    const l = lum[i];
    let r = l + (src[p] - l) * satMul;
    let g = l + (src[p + 1] - l) * satMul;
    let b = l + (src[p + 2] - l) * satMul;
    r = Math.round(Math.round(r / step) * step);
    g = Math.round(Math.round(g / step) * step);
    b = Math.round(Math.round(b / step) * step);
    out[p] = r; out[p + 1] = g; out[p + 2] = b; out[p + 3] = 255;
  }
  if (s > 0.04) {
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = y * w + x;
        const gx = -lum[idx - w - 1] - 2 * lum[idx - 1] - lum[idx + w - 1] + lum[idx - w + 1] + 2 * lum[idx + 1] + lum[idx + w + 1];
        const gy = -lum[idx - w - 1] - 2 * lum[idx - w] - lum[idx - w + 1] + lum[idx + w - 1] + 2 * lum[idx + w] + lum[idx + w + 1];
        const mag = Math.sqrt(gx * gx + gy * gy);
        if (mag > 90) {
          const darken = Math.min(1, (mag - 90) / 120) * s;
          const p = idx * 4;
          out[p] *= (1 - darken); out[p + 1] *= (1 - darken); out[p + 2] *= (1 - darken);
        }
      }
    }
  }
  const cv = $('#cart-canvas');
  cv.getContext('2d').putImageData(new ImageData(out, w, h), 0, 0);
  $('#cart-dl').href = cv.toDataURL('image/jpeg', 0.92);
}
$('#cart-strength').addEventListener('input', e => {
  $('#cart-strength-val').textContent = e.target.value + '%';
  cartApplyStrength(+e.target.value);
});

cart.init.addEventListener('click', async () => {
  beginLoad(cart);
  cart.bar.classList.add('indet');
  cart.lbl.textContent = 'loading TensorFlow.js + model ~1.5 MB…';
  try {
    await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');
    // Prefer WASM: the model's op mix (dynamic Fill/resize + depthwise ops) is the
    // combination the upstream demo itself ships on WASM — some WebGL drivers choke on it.
    let backendUsed = 'cpu';
    try {
      await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/tf-backend-wasm.min.js');
      tf.wasm.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/');
      await tf.setBackend('wasm');
      backendUsed = 'wasm';
    } catch (_) {
      try { await tf.setBackend('webgl'); backendUsed = 'webgl'; }
      catch (_2) { await tf.setBackend('cpu'); }
    }
    await tf.ready();
    cartModel = await tf.loadGraphModel('https://cdn.jsdelivr.net/gh/pratapvardhan/cartoonizer-with-tfjs@master/models/CartoonGAN/web-uint8/model.json');
    cart.bar.classList.remove('indet');
    showBody(cart);
    console.info('[cartoonize] backend:', backendUsed);
  } catch (e) { fail(cart, e); }
});

async function cartRun(imgEl) {
  const statusEl = $('#cart-status');
  statusEl.hidden = false;
  statusEl.textContent = 'stylizing on-device… (~5–10s on CPU)';
  $('#cart-dl').hidden = true;
  setStatus(cart, 'INFER', 'busy');
  let img, padded, resized, normalized, result, squeezed, out;
  try {
    img = tf.browser.fromPixels(imgEl);
    const dim0 = img.shape[0], dim1 = img.shape[1];
    const pad = (dim0 > dim1) ? [[0, 0], [dim0 - dim1, 0], [0, 0]] : [[dim1 - dim0, 0], [0, 0], [0, 0]];
    padded = img.pad(pad);
    const size = 256;
    resized = tf.image.resizeBilinear(padded, [size, size]).reshape([1, size, size, 3]);
    const offset = tf.scalar(127.5);
    normalized = resized.sub(offset).div(offset);
    const t0 = performance.now();
    result = await cartModel.predict({ 'input_photo:0': normalized });
    const ms = Math.round(performance.now() - t0);
    squeezed = result.squeeze().sub(tf.scalar(-1)).div(tf.scalar(2)).clipByValue(0, 1);
    const padAmt = Math.round(Math.abs(dim0 - dim1) / Math.max(dim0, dim1) * size);
    const slice = (dim0 > dim1) ? [0, padAmt, 0] : [padAmt, 0, 0];
    out = squeezed.slice(slice);
    const cv = $('#cart-canvas');
    cv.width = out.shape[1]; cv.height = out.shape[0];
    await tf.browser.toPixels(out, cv);
    cartRawImageData = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height);
    $('#cart-strength-row').hidden = false;
    cartApplyStrength(+$('#cart-strength').value);
    statusEl.hidden = true;
    $('#cart-meta').textContent = 'stylized in ' + ms + 'ms · on-device · 256px working resolution';
    $('#cart-dl').hidden = false;
    setStatus(cart, 'READY', 'ok');
  } catch (e) {
    console.warn('[cartoonize] inference failed', e);
    // one automatic retry on the most compatible backend before giving up
    if (!cartRetried && tf.getBackend() !== 'cpu') {
      cartRetried = true;
      try {
        await tf.setBackend('cpu');
        await tf.ready();
        [img, padded, resized, normalized, result, squeezed, out].forEach(t => { try { t && t.dispose(); } catch (_) {} });
        return cartRun(imgEl);
      } catch (_) {}
    }
    statusEl.textContent = 'failed: ' + (e && e.message ? e.message : 'inference error') + ' — try a smaller photo or a different browser';
    setStatus(cart, 'READY', 'ok');
  } finally {
    [img, padded, resized, normalized, result, squeezed, out].forEach(t => { try { t && t.dispose(); } catch (_) {} });
  }
}

const cartDrop = $('#cart-drop'), cartFile = $('#cart-file'), cartPreview = $('#cart-preview');
cartDrop.addEventListener('click', () => cartFile.click());
cartDrop.addEventListener('dragover', e => { e.preventDefault(); cartDrop.classList.add('over'); });
cartDrop.addEventListener('dragleave', () => cartDrop.classList.remove('over'));
cartDrop.addEventListener('drop', e => {
  e.preventDefault();
  cartDrop.classList.remove('over');
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) handleCartFile(f);
});
cartFile.addEventListener('change', e => { if (e.target.files[0]) handleCartFile(e.target.files[0]); });
function handleCartFile(file) {
  if (!cartModel) return;
  const reader = new FileReader();
  reader.onload = ev => {
    cartPreview.src = ev.target.result;
    cartPreview.hidden = false;
    $('#cart-drop-inner').hidden = true;
    cartPreview.onload = () => cartRun(cartPreview);
  };
  reader.readAsDataURL(file);
}

/* ================================================================
   HUD nav — active-section scrollspy + scramble hover (matches index.html)
   ================================================================ */
const labNavLinks = $$('.hud .c a');
const labSecEls = $$('[id^="demo-"]');
function labNavActive() {
  const line = scrollY + innerHeight * .35;
  let cur = null;
  for (const s of labSecEls) if (s.offsetTop <= line) cur = s.id;
  labNavLinks.forEach(a => a.classList.toggle('active', a.dataset.sec === cur));
}
addEventListener('scroll', labNavActive, { passive: true });
labNavActive();

const LAB_GLYPHS = '!<>-_\\/[]{}—=+*^?#';
function labScramble(el) {
  const orig = el.dataset.text || el.textContent;
  el.dataset.text = orig;
  let frame = 0;
  cancelAnimationFrame(el._raf);
  (function run() {
    frame++;
    el.textContent = [...orig].map((c, i) =>
      c === ' ' ? ' ' : (i < frame / 2.2 ? c : LAB_GLYPHS[(Math.random() * LAB_GLYPHS.length) | 0])
    ).join('');
    if (frame / 2.2 < orig.length) el._raf = requestAnimationFrame(run);
    else el.textContent = orig;
  })();
}
if (!matchMedia('(hover: none)').matches) {
  $$('.scr').forEach(el => el.addEventListener('mouseenter', () => labScramble(el)));
}
})();
