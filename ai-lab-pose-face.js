/* AI Lab — E/05 Rep counter (Pose Landmarker) + E/06 Puppet face (Face Landmarker).
   All models run locally; nothing leaves the tab. */
(function () {
'use strict';
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const MP = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
let mpMod = null;

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
function setStatus(u, txt, cls) { u.st.textContent = txt; u.st.className = 'st ' + (cls || ''); }
function beginLoad(u) { u.init.disabled = true; u.prog.hidden = false; setStatus(u, 'LOADING', 'busy'); }
function showBody(u) { u.gate.hidden = true; u.body.hidden = false; setStatus(u, 'READY', 'ok'); }
function fail(u, e) {
  console.warn(e);
  u.gate.hidden = true; u.body.hidden = true; u.err.hidden = false;
  u.errMsg.textContent = (e && e.message) ? String(e.message).slice(0, 160) : 'Unknown error';
  setStatus(u, 'FAILED', 'bad');
}
const accent = () => getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
let beepCtx2 = null;
function beep2(freq, dur, type, vol) {
  try {
    beepCtx2 = beepCtx2 || new (window.AudioContext || window.webkitAudioContext)();
    const osc = beepCtx2.createOscillator(), gain = beepCtx2.createGain();
    osc.type = type || 'square';
    osc.frequency.value = freq || 660;
    gain.gain.value = vol != null ? vol : 0.05;
    osc.connect(gain); gain.connect(beepCtx2.destination);
    osc.start();
    gain.gain.exponentialRampToValueAtTime(0.0001, beepCtx2.currentTime + (dur || 0.09));
    osc.stop(beepCtx2.currentTime + (dur || 0.09) + 0.02);
  } catch (_) {}
}

/* ================================================================
   E/05 · REP COUNTER — Pose Landmarker Lite
   ================================================================ */
const rep = card('rep');
let poseLm = null, repStream = null, repRAF = 0, repLastTime = -1;
let repEx = 'squat', repCount = 0, repPhase = 'up';

rep.init.addEventListener('click', async () => {
  beginLoad(rep);
  rep.bar.classList.add('indet');
  rep.lbl.textContent = 'downloading pose model ~5.5 MB…';
  try {
    mpMod = mpMod || await import(MP);
    const fs = await mpMod.FilesetResolver.forVisionTasks(MP + '/wasm');
    const opts = delegate => ({
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        delegate
      },
      runningMode: 'VIDEO', numPoses: 1
    });
    try { poseLm = await mpMod.PoseLandmarker.createFromOptions(fs, opts('GPU')); }
    catch (_) { poseLm = await mpMod.PoseLandmarker.createFromOptions(fs, opts('CPU')); }
    rep.bar.classList.remove('indet');
    showBody(rep);
  } catch (e) { fail(rep, e); }
});

$$('#rep-select button').forEach(b => b.addEventListener('click', () => {
  repEx = b.dataset.ex;
  $$('#rep-select button').forEach(x => x.classList.toggle('on', x === b));
  repCount = 0; repPhase = 'up';
  $('#rep-count').textContent = '0';
  $('#rep-phase').textContent = '—';
}));
$('#rep-reset').addEventListener('click', () => {
  repCount = 0; repPhase = 'up';
  $('#rep-count').textContent = '0';
});

const repCamBtn = $('#rep-cam');
repCamBtn.addEventListener('click', async () => {
  if (repStream) { stopRepCam(); return; }
  try {
    repStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    const video = $('#rep-video');
    video.srcObject = repStream;
    await video.play();
    const cv = $('#rep-canvas');
    cv.width = video.videoWidth; cv.height = video.videoHeight;
    repCamBtn.querySelector('span').textContent = 'Stop camera';
    $('#rep-stage').classList.add('live');
    setStatus(rep, 'LIVE ●', 'ok');
    repLoop();
  } catch (e) {
    $('#rep-hint').textContent = 'camera unavailable — ' + e.message;
  }
});
function stopRepCam() {
  cancelAnimationFrame(repRAF);
  if (repStream) repStream.getTracks().forEach(t => t.stop());
  repStream = null;
  $('#rep-stage').classList.remove('live');
  repCamBtn.querySelector('span').textContent = 'Start camera';
  $('#rep-hint').textContent = 'start the camera, step back so your body is in frame';
  $('#rep-phase').textContent = '—';
  $('#rep-depth').style.width = '0%';
  setStatus(rep, 'READY', 'ok');
}

function angleAt(a, b, c) {
  const v1 = { x: a.x - b.x, y: a.y - b.y }, v2 = { x: c.x - b.x, y: c.y - b.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const m = Math.hypot(v1.x, v1.y) * Math.hypot(v2.x, v2.y) || 1e-6;
  return Math.acos(Math.max(-1, Math.min(1, dot / m))) * 180 / Math.PI;
}
const vis = (lm, ids) => ids.every(i => (lm[i].visibility ?? 1) > 0.5);

function repLogic(lm) {
  /* landmark ids: 0 nose · 11/12 shoulders · 13/14 elbows · 15/16 wrists · 23/24 hips · 25/26 knees · 27/28 ankles */
  let depth = 0, hint = '', phaseNow = repPhase;
  if (repEx === 'squat') {
    if (!vis(lm, [23, 24, 25, 26, 27, 28])) return { hint: 'step back — hips, knees & ankles must be in frame' };
    const kneeA = (angleAt(lm[23], lm[25], lm[27]) + angleAt(lm[24], lm[26], lm[28])) / 2;
    depth = Math.max(0, Math.min(1, (168 - kneeA) / 70));
    if (kneeA < 105) phaseNow = 'down';
    else if (kneeA > 160) phaseNow = 'up';
    hint = 'knee angle ' + Math.round(kneeA) + '° — ' + (phaseNow === 'down' ? 'good depth, stand up' : 'squat down past 105°');
    if (repPhase === 'down' && phaseNow === 'up') { repCount++; beep2(880, 0.12, 'square', 0.06); }
  } else if (repEx === 'jack') {
    if (!vis(lm, [11, 12, 15, 16, 27, 28])) return { hint: 'step back — full body in frame for jumping jacks' };
    const shoulderW = Math.abs(lm[11].x - lm[12].x) || 1e-6;
    const handsUp = lm[15].y < lm[0].y && lm[16].y < lm[0].y;
    const feetApart = Math.abs(lm[27].x - lm[28].x) > shoulderW * 1.5;
    const open = handsUp && feetApart;
    const closed = !handsUp && Math.abs(lm[27].x - lm[28].x) < shoulderW * 1.15;
    depth = (handsUp ? 0.5 : 0) + (feetApart ? 0.5 : 0);
    if (open) phaseNow = 'down';           /* "down" = open position */
    else if (closed) phaseNow = 'up';
    hint = open ? 'open — now close' : (closed ? 'closed — jump open, hands above head' : 'mid-jack…');
    if (repPhase === 'down' && phaseNow === 'up') { repCount++; beep2(880, 0.12, 'square', 0.06); }
  } else if (repEx === 'curl') {
    if (!vis(lm, [11, 12, 13, 14, 15, 16])) return { hint: 'face the camera — shoulders, elbows & wrists in frame' };
    const elbowA = (angleAt(lm[11], lm[13], lm[15]) + angleAt(lm[12], lm[14], lm[16])) / 2;
    depth = Math.max(0, Math.min(1, (160 - elbowA) / 105));
    if (elbowA < 60) phaseNow = 'down';    /* "down" = curled */
    else if (elbowA > 150) phaseNow = 'up';
    hint = 'elbow angle ' + Math.round(elbowA) + '° — ' + (phaseNow === 'down' ? 'good curl, extend back down' : 'curl both arms up past 60°');
    if (repPhase === 'down' && phaseNow === 'up') { repCount++; beep2(880, 0.12, 'square', 0.06); }
  } else if (repEx === 'head') {
    if (!vis(lm, [0, 11, 12])) return { hint: 'face the camera — head & shoulders in frame' };
    const shoulderW = Math.abs(lm[11].x - lm[12].x) || 1e-6;
    const off = (lm[0].x - (lm[11].x + lm[12].x) / 2) / shoulderW;
    depth = Math.max(0, Math.min(1, Math.abs(off) / 0.4));
    if (Math.abs(off) > 0.32) phaseNow = 'down';  /* "down" = turned */
    else if (Math.abs(off) < 0.12) phaseNow = 'up';
    hint = phaseNow === 'down' ? 'turned — come back to center' : 'turn your head fully left or right';
    if (repPhase === 'down' && phaseNow === 'up') { repCount++; beep2(880, 0.12, 'square', 0.06); }
  } else { /* arm raises */
    if (!vis(lm, [11, 12, 15, 16])) return { hint: 'face the camera — shoulders & wrists in frame' };
    const up = lm[15].y < lm[0].y && lm[16].y < lm[0].y;
    const down = lm[15].y > lm[11].y && lm[16].y > lm[12].y;
    const shoulderY = (lm[11].y + lm[12].y) / 2, wristY = (lm[15].y + lm[16].y) / 2;
    depth = Math.max(0, Math.min(1, (shoulderY - wristY + 0.15) / 0.45));
    if (up) phaseNow = 'down';             /* "down" = arms raised */
    else if (down) phaseNow = 'up';
    hint = up ? 'raised — lower your arms' : (down ? 'raise both arms above your head' : 'raising…');
    if (repPhase === 'down' && phaseNow === 'up') { repCount++; beep2(880, 0.12, 'square', 0.06); }
  }
  repPhase = phaseNow;
  return { depth, hint, phase: phaseNow };
}

const REP_PHASE_LBL = {
  squat: { down: 'DOWN ▾', up: 'UP ▴' },
  jack:  { down: 'OPEN ▴', up: 'CLOSED ▾' },
  raise: { down: 'RAISED ▴', up: 'LOWERED ▾' },
  curl:  { down: 'CURLED ▴', up: 'EXTENDED ▾' },
  head:  { down: 'TURNED ◂▸', up: 'CENTER ·' }
};

function repLoop() {
  const video = $('#rep-video'), cv = $('#rep-canvas'), ctx = cv.getContext('2d');
  const du = new mpMod.DrawingUtils(ctx);
  (function loop() {
    repRAF = requestAnimationFrame(loop);
    if (!repStream || video.readyState < 2) return;
    if (video.currentTime === repLastTime) return;
    repLastTime = video.currentTime;
    const res = poseLm.detectForVideo(video, performance.now());
    ctx.clearRect(0, 0, cv.width, cv.height);
    const lm = res.landmarks && res.landmarks[0];
    if (!lm) {
      $('#rep-hint').textContent = 'nobody in frame — step back so the camera can see you';
      $('#rep-phase').textContent = '—';
      return;
    }
    du.drawConnectors(lm, mpMod.PoseLandmarker.POSE_CONNECTIONS, { color: accent(), lineWidth: 2 });
    du.drawLandmarks(lm, { color: '#ffffff', lineWidth: 1, radius: 2.5 });
    const r = repLogic(lm);
    $('#rep-count').textContent = String(repCount);
    $('#rep-hint').textContent = r.hint || '';
    if (r.phase) $('#rep-phase').textContent = (REP_PHASE_LBL[repEx] && REP_PHASE_LBL[repEx][r.phase]) || r.phase.toUpperCase();
    $('#rep-depth').style.width = Math.round((r.depth || 0) * 100) + '%';
  })();
}

/* ================================================================
   E/06 · PUPPET FACE — Face Landmarker (478 landmarks + blendshapes)
   ================================================================ */
const pup = card('pup');
let faceLm = null, pupStream = null, pupRAF = 0, pupLastTime = -1, pupTeleAt = 0;
let blinkWas = false, blinkTimes = [];

pup.init.addEventListener('click', async () => {
  beginLoad(pup);
  pup.bar.classList.add('indet');
  pup.lbl.textContent = 'downloading face mesh model ~3.7 MB…';
  try {
    mpMod = mpMod || await import(MP);
    const fs = await mpMod.FilesetResolver.forVisionTasks(MP + '/wasm');
    const opts = delegate => ({
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate
      },
      runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true
    });
    try { faceLm = await mpMod.FaceLandmarker.createFromOptions(fs, opts('GPU')); }
    catch (_) { faceLm = await mpMod.FaceLandmarker.createFromOptions(fs, opts('CPU')); }
    pup.bar.classList.remove('indet');
    showBody(pup);
  } catch (e) { fail(pup, e); }
});

const pupCamBtn = $('#pup-cam');
pupCamBtn.addEventListener('click', async () => {
  if (pupStream) { stopPupCam(); return; }
  try {
    pupStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    const video = $('#pup-video');
    video.srcObject = pupStream;
    await video.play();
    const cv = $('#pup-canvas-cam');
    cv.width = video.videoWidth; cv.height = video.videoHeight;
    pupCamBtn.querySelector('span').textContent = 'Stop camera';
    $('#pup-stage').classList.add('live');
    setStatus(pup, 'LIVE ●', 'ok');
    blinkTimes = []; blinkWas = false;
    pupLoop();
  } catch (e) {
    $('#pup-tele').innerHTML = '<div class="det-none">camera unavailable — ' + e.message + '</div>';
  }
});
function stopPupCam() {
  cancelAnimationFrame(pupRAF);
  if (pupStream) pupStream.getTracks().forEach(t => t.stop());
  pupStream = null;
  $('#pup-stage').classList.remove('live');
  pupCamBtn.querySelector('span').textContent = 'Start camera';
  $('#pup-tele').innerHTML = '<div class="det-none">— camera off —</div>';
  const pc = $('#pup-canvas');
  pc.getContext('2d').clearRect(0, 0, pc.width, pc.height);
  setStatus(pup, 'READY', 'ok');
}

function headPose(lm) {
  /* landmark-based: robust and cheap. 1 nose tip · 234/454 cheeks · 10 forehead · 152 chin · 33/263 eye outer corners */
  const yaw = ((lm[1].x - (lm[234].x + lm[454].x) / 2) / (Math.abs(lm[454].x - lm[234].x) || 1e-6)) * 90;
  const pitch = ((lm[1].y - (lm[10].y + lm[152].y) / 2) / (Math.abs(lm[152].y - lm[10].y) || 1e-6)) * 90;
  const roll = Math.atan2(lm[263].y - lm[33].y, lm[263].x - lm[33].x) * 180 / Math.PI;
  return { yaw, pitch, roll };
}

function drawPuppet(B, pose) {
  const cv = $('#pup-canvas'), ctx = cv.getContext('2d');
  const w = cv.width, h = cv.height, ac = accent();
  ctx.clearRect(0, 0, w, h);
  ctx.save();
  /* mirror yaw so the puppet behaves like a mirror */
  const yaw = Math.max(-30, Math.min(30, -pose.yaw));
  const pitch = Math.max(-25, Math.min(25, pose.pitch));
  const roll = Math.max(-25, Math.min(25, -pose.roll));
  ctx.translate(w / 2 + yaw * 1.2, h / 2 + pitch * 1.2);
  ctx.rotate(roll * Math.PI / 180);

  const blinkL = B.eyeBlinkLeft || 0, blinkR = B.eyeBlinkRight || 0;
  const jaw = B.jawOpen || 0;
  const smile = ((B.mouthSmileLeft || 0) + (B.mouthSmileRight || 0)) / 2;
  const browUp = Math.max(B.browInnerUp || 0, ((B.browOuterUpLeft || 0) + (B.browOuterUpRight || 0)) / 2);

  /* head */
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = ac;
  ctx.fillStyle = 'rgba(110,245,200,0.05)';
  ctx.beginPath();
  ctx.arc(0, 0, 92, 0, Math.PI * 2);
  ctx.fill(); ctx.stroke();

  /* eyes — mirrored: puppet's left eye is viewer's left = user's right eye */
  const eyeY = -18, eyeDX = 36;
  const drawEye = (dx, blink) => {
    const openH = Math.max(1.5, 13 * (1 - blink));
    ctx.beginPath();
    if (blink > 0.62) {
      ctx.moveTo(dx - 14, eyeY);
      ctx.lineTo(dx + 14, eyeY);
      ctx.stroke();
    } else {
      ctx.ellipse(dx, eyeY, 14, openH, 0, 0, Math.PI * 2);
      ctx.stroke();
      /* pupil drifts with head yaw */
      ctx.beginPath();
      ctx.fillStyle = ac;
      ctx.arc(dx + yaw * 0.18, eyeY + pitch * 0.1, 4.5, 0, Math.PI * 2);
      ctx.fill();
    }
  };
  drawEye(-eyeDX, blinkR);
  drawEye(eyeDX, blinkL);

  /* brows */
  const browY = -40 - browUp * 12;
  ctx.beginPath();
  ctx.moveTo(-eyeDX - 15, browY + 3); ctx.lineTo(-eyeDX + 13, browY - 2);
  ctx.moveTo(eyeDX - 13, browY - 2); ctx.lineTo(eyeDX + 15, browY + 3);
  ctx.stroke();

  /* mouth — smile lifts corners, jaw opens it */
  const mY = 38, mW = 30 + smile * 14;
  const openH2 = 2 + jaw * 38;
  const corner = -smile * 16;
  ctx.beginPath();
  ctx.moveTo(-mW, mY + corner);
  ctx.quadraticCurveTo(0, mY - openH2 * 0.25, mW, mY + corner);
  ctx.quadraticCurveTo(0, mY + openH2, -mW, mY + corner);
  ctx.closePath();
  if (jaw > 0.08) { ctx.fillStyle = 'rgba(110,245,200,0.18)'; ctx.fill(); }
  ctx.stroke();

  ctx.restore();
}

function pupTele(B, pose) {
  const now = performance.now();
  if (now - pupTeleAt < 150) return;
  pupTeleAt = now;
  /* blinks per minute over a rolling 30s window */
  const cutoff = now - 30000;
  blinkTimes = blinkTimes.filter(t => t > cutoff);
  const bpm = blinkTimes.length * 2; /* 30s window → ×2 for per-minute */
  const pct = v => Math.round(Math.max(0, Math.min(1, v)) * 100) + '%';
  const smile = ((B.mouthSmileLeft || 0) + (B.mouthSmileRight || 0)) / 2;
  const browUp = B.browInnerUp || 0;
  $('#pup-tele').innerHTML =
    '<div class="face-line"><span class="k">blinks/min</span><span class="v">' + bpm + '</span></div>' +
    '<div class="face-line"><span class="k">smile</span><span class="v">' + pct(smile) + '</span></div>' +
    '<div class="face-line"><span class="k">jaw open</span><span class="v">' + pct(B.jawOpen || 0) + '</span></div>' +
    '<div class="face-line"><span class="k">brow raise</span><span class="v">' + pct(browUp) + '</span></div>' +
    '<div class="face-line"><span class="k">head yaw</span><span class="v">' + Math.round(-pose.yaw) + '°</span></div>' +
    '<div class="face-line"><span class="k">head pitch</span><span class="v">' + Math.round(pose.pitch) + '°</span></div>' +
    '<div class="face-line"><span class="k">head roll</span><span class="v">' + Math.round(-pose.roll) + '°</span></div>';
}

function pupLoop() {
  const video = $('#pup-video'), cv = $('#pup-canvas-cam'), ctx = cv.getContext('2d');
  const du = new mpMod.DrawingUtils(ctx);
  (function loop() {
    pupRAF = requestAnimationFrame(loop);
    if (!pupStream || video.readyState < 2) return;
    if (video.currentTime === pupLastTime) return;
    pupLastTime = video.currentTime;
    const res = faceLm.detectForVideo(video, performance.now());
    ctx.clearRect(0, 0, cv.width, cv.height);
    const lm = res.faceLandmarks && res.faceLandmarks[0];
    if (!lm) {
      $('#pup-tele').innerHTML = '<div class="det-none">show your face to the camera…</div>';
      return;
    }
    du.drawConnectors(lm, mpMod.FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: 'rgba(110,245,200,0.12)', lineWidth: 0.5 });
    du.drawConnectors(lm, mpMod.FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: accent(), lineWidth: 1.5 });

    const cats = res.faceBlendshapes && res.faceBlendshapes[0] && res.faceBlendshapes[0].categories || [];
    const B = {};
    for (const c of cats) B[c.categoryName] = c.score;

    /* blink edge detection */
    const blinkAvg = ((B.eyeBlinkLeft || 0) + (B.eyeBlinkRight || 0)) / 2;
    if (blinkAvg > 0.55 && !blinkWas) { blinkWas = true; blinkTimes.push(performance.now()); }
    else if (blinkAvg < 0.3) blinkWas = false;

    const pose = headPose(lm);
    drawPuppet(B, pose);
    pupTele(B, pose);
  })();
}

/* register camera stops for the accordion manager (in ai-lab-demos.js) */
window.__labStops = Object.assign(window.__labStops || {}, {
  rep: () => { if (repStream) stopRepCam(); },
  pup: () => { if (pupStream) stopPupCam(); }
});
})();
