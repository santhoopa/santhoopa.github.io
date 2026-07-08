/* AI Lab — three in-browser ML demos. All models run locally (WASM/WebGPU); nothing leaves the tab. */
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
   01 · SEMANTIC RECALL — retrieve (MiniLM) → read (DistilBERT-QA)
   ================================================================ */
const rag = card('rag');
let ragPipe = null, qaPipe = null, ragIndex = null;

rag.init.addEventListener('click', async () => {
  beginLoad(rag);
  try {
    const { pipeline } = await loadTJS();
    const ph = tjsProgress(rag);
    ragPipe = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
      dtype: 'q8', progress_callback: ph
    });
    qaPipe = await pipeline('question-answering', 'Xenova/distilbert-base-cased-distilled-squad', {
      dtype: 'q8', progress_callback: ph
    });
    rag.lbl.textContent = 'embedding corpus…';
    rag.bar.style.width = '100%';
    const texts = window.LAB_CORPUS.map(c => c.text);
    const out = await ragPipe(texts, { pooling: 'mean', normalize: true });
    const arr = out.tolist();
    ragIndex = arr.map((v, i) => ({ vec: v, ...window.LAB_CORPUS[i] }));
    showBody(rag);
    $('#rag-dims').textContent = ragIndex.length + ' chunks · 384-dim retriever + extractive reader';
  } catch (e) { fail(rag, e); }
});

function cosine(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

async function ragAsk(q) {
  if (!ragPipe || !qaPipe || !q.trim()) return;
  const ansEl = $('#rag-answer'), resEl = $('#rag-results');
  ansEl.innerHTML = '<div class="thinking">embed query → search index → read answer span…</div>';
  resEl.innerHTML = '';
  setStatus(rag, 'INFER', 'busy');
  try {
    const out = await ragPipe(q, { pooling: 'mean', normalize: true });
    const qv = out.tolist()[0];
    const ranked = ragIndex.map(c => ({ ...c, score: cosine(qv, c.vec) }))
      .sort((a, b) => b.score - a.score).slice(0, 3);

    /* reader: run extractive QA over each retrieved chunk, keep the best span */
    let best = null;
    for (const r of ranked) {
      try {
        const res = await qaPipe(q, r.text);
        if (res && res.answer && (!best || res.score > best.score)) best = { answer: res.answer, score: res.score, src: r };
      } catch (_) {}
    }

    ansEl.innerHTML = '';
    const panel = document.createElement('div');
    if (best && best.score > 0.05) {
      panel.className = 'rag-ans';
      panel.innerHTML =
        '<div class="ans-k">// ANSWER — extracted from [<span class="ans-tag"></span>]</div>' +
        '<p class="ans-text"></p>' +
        '<div class="ans-conf">reader confidence <b></b>%</div>';
      panel.querySelector('.ans-tag').textContent = best.src.tag;
      panel.querySelector('.ans-text').textContent = best.answer;
      panel.querySelector('.ans-conf b').textContent = Math.round(best.score * 100);
    } else {
      panel.className = 'rag-ans low';
      panel.innerHTML =
        '<div class="ans-k">// LOW CONFIDENCE — no clear answer span found</div>' +
        '<p class="ans-text soft">The reader couldn\'t extract a confident answer. Closest retrieved context is below — try rephrasing as a direct question.</p>';
    }
    ansEl.appendChild(panel);

    /* sources, collapsed */
    const det = document.createElement('details');
    det.className = 'rag-src';
    const sum = document.createElement('summary');
    sum.textContent = '// SOURCES — top ' + ranked.length + ' of ' + ragIndex.length + ' chunks, by cosine similarity';
    det.appendChild(sum);
    ranked.forEach((r, i) => {
      const pct = Math.max(0, Math.round(r.score * 100));
      const d = document.createElement('div');
      d.className = 'rag-hit' + (i === 0 ? ' top' : '');
      d.innerHTML =
        '<div class="hit-meta"><span class="hit-tag"></span><span class="hit-score">' + r.score.toFixed(3) + '</span></div>' +
        '<div class="hit-bar"><i style="width:' + pct + '%"></i></div>' +
        '<p class="hit-text"></p>';
      d.querySelector('.hit-tag').textContent = '[' + r.tag + ']';
      d.querySelector('.hit-text').textContent = r.text;
      det.appendChild(d);
    });
    if (best && best.score > 0.05) det.open = false; else det.open = true;
    resEl.appendChild(det);
    setStatus(rag, 'READY', 'ok');
  } catch (e) {
    ansEl.innerHTML = '<div class="thinking">query failed — try again</div>';
    setStatus(rag, 'READY', 'ok');
  }
}
$('#rag-ask').addEventListener('click', () => ragAsk($('#rag-q').value));
$('#rag-q').addEventListener('keydown', e => { if (e.key === 'Enter') ragAsk(e.target.value); });
$$('#rag-chips button').forEach(b => b.addEventListener('click', () => { $('#rag-q').value = b.textContent; ragAsk(b.textContent); }));

/* ================================================================
   02 · WHISPER TINY — speech → text
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
   03 · GESTURE LINK — hand tracking + rock·paper·scissors
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
const RPS_EMO = { Closed_Fist: '✊', Open_Palm: '✋', Victory: '✌' };
const RPS_BEATS = { Closed_Fist: 'Victory', Victory: 'Open_Palm', Open_Palm: 'Closed_Fist' };
let rpsScore = { you: 0, bot: 0 }, counting = false;

$('#rps-play').addEventListener('click', () => {
  if (counting) return;
  if (!gesStream) { $('#rps-result').textContent = 'start the camera first'; return; }
  counting = true;
  let n = 3;
  const cd = $('#ges-count');
  cd.textContent = n; cd.classList.add('on');
  $('#rps-result').textContent = 'get your hand ready…';
  const iv = setInterval(() => {
    n--;
    if (n > 0) { cd.textContent = n; return; }
    clearInterval(iv);
    cd.classList.remove('on');
    counting = false;
    resolveRound();
  }, 800);
});
function resolveRound() {
  const you = currentGesture;
  if (!you || !RPS_MAP[you]) {
    $('#rps-you').textContent = '?';
    $('#rps-bot').textContent = '—';
    $('#rps-result').textContent = 'no valid gesture — show ✊ ✋ or ✌';
    return;
  }
  const opts = Object.keys(RPS_MAP);
  const bot = opts[Math.floor(Math.random() * 3)];
  $('#rps-you').textContent = RPS_EMO[you];
  $('#rps-bot').textContent = RPS_EMO[bot];
  let msg;
  if (you === bot) msg = 'draw — ' + RPS_MAP[you] + ' vs ' + RPS_MAP[bot];
  else if (RPS_BEATS[you] === bot) { rpsScore.you++; msg = RPS_MAP[you] + ' beats ' + RPS_MAP[bot] + ' — you win'; }
  else { rpsScore.bot++; msg = RPS_MAP[bot] + ' beats ' + RPS_MAP[you] + ' — model wins'; }
  $('#rps-result').textContent = msg;
  $('#rps-score').textContent = 'YOU ' + rpsScore.you + ' — ' + rpsScore.bot + ' MODEL';
}
})();
