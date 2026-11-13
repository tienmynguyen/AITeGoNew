import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { SkeletonOverlay } from './skeletonOverlay.js';
import { VRMEngine } from './vrmEngine.js';
import { SequenceBlender } from './sequenceBlender.js';
import { parseNpyBuffer } from './npyParser.js';

// Cache for loaded word motions { 'word': motionJSON }
const motionCache = new Map();
let manifestWords = [];
let activeSentenceWords = ['xin chào'];
let isMirrorMode = false;

// Three.js Core Variables
let scene, camera, renderer, controls;
let skeletonOverlay, vrmEngine, sequenceBlender;
let clock = new THREE.Clock();

// FPS tracking
let frameTimes = [];
let lastFpsUpdate = 0;

/**
 * Initialize Three.js Viewport
 */
function initScene() {
  const container = document.getElementById('canvas-container');
  const width = container.clientWidth || window.innerWidth;
  const height = container.clientHeight || window.innerHeight;

  // Scene
  scene = new THREE.Scene();
  scene.background = null; // Transparent to show CSS radial gradient

  // Camera positioned at Upper Body height
  camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 20.0);
  camera.position.set(0, 1.35, 1.75);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  container.appendChild(renderer.domElement);

  // OrbitControls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 1.25, 0); // Focus at chest / upper torso
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.5;
  controls.maxDistance = 5.0;
  controls.maxPolarAngle = Math.PI / 2 + 0.1;
  controls.update();

  // Studio Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
  scene.add(ambientLight);

  // Key Light (warm-white front right)
  const keyLight = new THREE.DirectionalLight(0xfff5ea, 1.1);
  keyLight.position.set(1.5, 2.8, 2.2);
  scene.add(keyLight);

  // Fill Light (soft cyan front left)
  const fillLight = new THREE.DirectionalLight(0xd4f4ff, 0.85);
  fillLight.position.set(-1.8, 2.2, 1.8);
  scene.add(fillLight);

  // Rim / Hair Light (crisp backlight from above)
  const rimLight = new THREE.DirectionalLight(0x00e5ff, 0.9);
  rimLight.position.set(0, 3.2, -2.5);
  scene.add(rimLight);

  // Soft circular floor grid
  const gridHelper = new THREE.GridHelper(3.0, 20, 0x00e5ff, 0x1f293d);
  gridHelper.position.y = 0.001;
  gridHelper.material.opacity = 0.35;
  gridHelper.material.transparent = true;
  scene.add(gridHelper);

  // Initialize Modules
  skeletonOverlay = new SkeletonOverlay(scene);
  vrmEngine = new VRMEngine(scene);
  sequenceBlender = new SequenceBlender();

  // Handle Resize
  window.addEventListener('resize', onWindowResize);
}

function onWindowResize() {
  const container = document.getElementById('canvas-container');
  if (!container) return;
  const width = container.clientWidth;
  const height = container.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

/**
 * Fetch motion JSON by word name
 */
async function loadMotion(word) {
  if (motionCache.has(word)) {
    return motionCache.get(word);
  }

  try {
    const encoded = encodeURIComponent(word);
    const res = await fetch(`/motions/${encoded}.json`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} fetching motion for ${word}`);
    }
    const data = await res.json();
    motionCache.set(word, data);
    return data;
  } catch (err) {
    console.error(`[Motion] Error loading motion for '${word}':`, err);
    return null;
  }
}

/**
 * Update Sequence and Playback
 */
async function updateSequencePlayback(wordList, smooth = true) {
  const motions = [];
  for (const w of wordList) {
    const m = await loadMotion(w);
    if (m) motions.push(m);
  }

  if (motions.length === 0) return;

  if (smooth) {
    sequenceBlender.updateSequenceSmooth(motions);
  } else {
    sequenceBlender.setSequence(motions);
  }

  renderSentenceChips();
  updateTimelineHUD();
}

/**
 * Expose window.updateVSLSequence for real-time word chaining
 */
window.updateVSLSequence = async function (wordArray) {
  if (!Array.isArray(wordArray) || wordArray.length === 0) return;
  activeSentenceWords = [...wordArray];
  await updateSequencePlayback(activeSentenceWords, true);
};

/**
 * Render Word Bank Badges
 */
function renderWordBank(filter = 'all', searchQuery = '') {
  const container = document.getElementById('word-bank-container');
  container.innerHTML = '';

  let filtered = manifestWords;

  // Filter by category
  if (filter === 'one_hand') {
    filtered = filtered.filter((w) => w.category === 'one_hand');
  } else if (filter === 'two_hands') {
    filtered = filtered.filter((w) => w.category === 'two_hands' && isNaN(Number(w.word)));
  } else if (filter === 'number') {
    filtered = filtered.filter((w) => !isNaN(Number(w.word)));
  }

  // Filter by search query
  if (searchQuery.trim()) {
    const q = searchQuery.toLowerCase().trim();
    filtered = filtered.filter((w) => w.word.toLowerCase().includes(q));
  }

  document.getElementById('word-count-badge').textContent = `${filtered.length} từ`;

  filtered.forEach((item) => {
    const badge = document.createElement('div');
    badge.className = 'word-badge';
    if (activeSentenceWords.length === 1 && activeSentenceWords[0] === item.word) {
      badge.classList.add('active');
    }

    const isNum = !isNaN(Number(item.word));
    const tagClass = item.category === 'one_hand' ? 'one_hand' : 'two_hands';
    const tagText = isNum ? 'Số' : item.category === 'one_hand' ? '1 Tay' : '2 Tay';

    badge.innerHTML = `
      <div class="word-badge-title">${item.word}</div>
      <div class="word-badge-meta">
        <span class="badge-tag ${tagClass}">${tagText}</span>
        <span>${item.frameCount}f</span>
      </div>
    `;

    // Click to append to sentence or play single word
    badge.addEventListener('click', async () => {
      // If clicking already single active word, restart it
      if (activeSentenceWords.length === 1 && activeSentenceWords[0] === item.word) {
        sequenceBlender.reset();
        sequenceBlender.isPlaying = true;
      } else {
        // Append to sentence sequence
        activeSentenceWords.push(item.word);
        await updateSequencePlayback(activeSentenceWords, true);
      }
      highlightActiveBadges();
    });

    container.appendChild(badge);
  });
}

function highlightActiveBadges() {
  const badges = document.querySelectorAll('.word-badge');
  badges.forEach((b) => {
    const title = b.querySelector('.word-badge-title')?.textContent;
    if (activeSentenceWords.includes(title)) {
      b.classList.add('active');
    } else {
      b.classList.remove('active');
    }
  });
}

/**
 * Render Sentence Chips in Sequence Builder
 */
function renderSentenceChips() {
  const container = document.getElementById('sentence-chips-box');
  const countBadge = document.getElementById('sequence-count');
  countBadge.textContent = `${activeSentenceWords.length} từ`;

  if (activeSentenceWords.length === 0) {
    container.innerHTML = `<span class="empty-chip-hint" id="empty-hint">Bấm các từ ở trên để ghép câu VSL</span>`;
    return;
  }

  container.innerHTML = '';
  activeSentenceWords.forEach((word, idx) => {
    const chip = document.createElement('div');
    chip.className = 'sentence-chip';
    chip.dataset.idx = idx;

    const currentItem = sequenceBlender.getCurrentItem();
    if (currentItem && currentItem.wordIndex === idx) {
      chip.classList.add('active-chip');
    }

    chip.innerHTML = `
      <span>${word}</span>
      <span class="chip-remove" data-remove="${idx}">&times;</span>
    `;

    container.appendChild(chip);
  });

  // Attach chip remove listeners
  container.querySelectorAll('.chip-remove').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const idxToRemove = parseInt(btn.dataset.remove, 10);
      activeSentenceWords.splice(idxToRemove, 1);
      await updateSequencePlayback(activeSentenceWords, true);
      highlightActiveBadges();
    });
  });
}

/**
 * UI Event Listeners & Sliders Setup
 */
function setupUIEventListeners() {
  // Search
  const searchInput = document.getElementById('search-input');
  searchInput.addEventListener('input', (e) => {
    const activeFilter = document.querySelector('.filter-tab.active')?.dataset.filter || 'all';
    renderWordBank(activeFilter, e.target.value);
  });

  // Filter Tabs
  document.querySelectorAll('.filter-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.filter-tab').forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      renderWordBank(tab.dataset.filter, searchInput.value);
    });
  });

  // Sentence Builder Buttons
  document.getElementById('btn-play-sentence').addEventListener('click', async () => {
    if (activeSentenceWords.length === 0) return;
    sequenceBlender.reset();
    sequenceBlender.isPlaying = true;
    updatePlayPauseIcon(true);
  });

  document.getElementById('btn-clear-sentence').addEventListener('click', async () => {
    activeSentenceWords = [];
    sequenceBlender.setSequence([]);
    renderSentenceChips();
    highlightActiveBadges();
    document.getElementById('hud-active-word').textContent = '—';
  });

  // Preset Sentences
  document.querySelectorAll('.preset-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const words = btn.dataset.preset.split(',');
      activeSentenceWords = words;
      await updateSequencePlayback(activeSentenceWords, true);
      highlightActiveBadges();
    });
  });

  // Camera Presets
  document.querySelectorAll('.camera-controls-hud .hud-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.camera-controls-hud .hud-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');

      const preset = btn.dataset.cam;
      if (preset === 'upper') {
        camera.position.set(0, 1.35, 1.75);
        controls.target.set(0, 1.25, 0);
      } else if (preset === 'close') {
        camera.position.set(0, 1.36, 1.05);
        controls.target.set(0, 1.30, 0);
      } else if (preset === 'full') {
        camera.position.set(0, 1.10, 2.80);
        controls.target.set(0, 0.90, 0);
      } else if (preset === 'side') {
        camera.position.set(1.20, 1.35, 1.35);
        controls.target.set(0, 1.25, 0);
      }
      controls.update();
    });
  });

  // Display Toggles
  const toggleSkeleton = document.getElementById('toggle-skeleton');
  toggleSkeleton.addEventListener('change', (e) => {
    skeletonOverlay.setVisible(e.target.checked);
  });

  const toggleVRM = document.getElementById('toggle-vrm');
  toggleVRM.addEventListener('change', (e) => {
    if (vrmEngine.currentVRM) {
      vrmEngine.currentVRM.scene.visible = e.target.checked;
    }
  });

  const toggleMirror = document.getElementById('toggle-mirror');
  toggleMirror.addEventListener('change', (e) => {
    isMirrorMode = e.target.checked;
    skeletonOverlay.setMirror(isMirrorMode);
  });

  // Sliders
  const sliderSpeed = document.getElementById('slider-speed');
  sliderSpeed.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    sequenceBlender.setPlaybackSpeed(val);
    document.getElementById('val-speed').textContent = `${val.toFixed(2)}x`;
  });

  const sliderBridge = document.getElementById('slider-bridge');
  sliderBridge.addEventListener('input', (e) => {
    const val = parseInt(e.target.value, 10);
    sequenceBlender.setBridgeFrameCount(val);
    document.getElementById('val-bridge').textContent = `${val} frames`;
  });

  const sliderSlerp = document.getElementById('slider-slerp');
  sliderSlerp.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    vrmEngine.setSlerpFactor(val);
    document.getElementById('val-slerp').textContent = val.toFixed(2);
  });

  // Playback Timeline Controls
  const btnPlayPause = document.getElementById('btn-play-pause');
  btnPlayPause.addEventListener('click', () => {
    const playing = sequenceBlender.togglePlay();
    updatePlayPauseIcon(playing);
  });

  document.getElementById('btn-step-prev').addEventListener('click', () => {
    sequenceBlender.isPlaying = false;
    sequenceBlender.step(-1);
    updatePlayPauseIcon(false);
  });

  document.getElementById('btn-step-next').addEventListener('click', () => {
    sequenceBlender.isPlaying = false;
    sequenceBlender.step(1);
    updatePlayPauseIcon(false);
  });

  document.getElementById('btn-reset').addEventListener('click', () => {
    sequenceBlender.reset();
  });

  const btnLoop = document.getElementById('btn-loop-toggle');
  btnLoop.addEventListener('click', () => {
    sequenceBlender.isLooping = !sequenceBlender.isLooping;
    btnLoop.style.color = sequenceBlender.isLooping ? 'var(--cyan-primary)' : 'var(--text-dim)';
  });

  // Timeline Progress Scrubber
  const timelineBar = document.getElementById('timeline-bar');
  timelineBar.addEventListener('click', (e) => {
    const rect = timelineBar.getBoundingClientRect();
    const progress = (e.clientX - rect.left) / rect.width;
    sequenceBlender.seek(progress);
  });

  // Drag and drop handler (.npy or .vrm)
  setupDragAndDrop();
}

function updatePlayPauseIcon(isPlaying) {
  const icon = document.getElementById('play-pause-icon');
  if (isPlaying) {
    icon.innerHTML = `<path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>`;
  } else {
    icon.innerHTML = `<polygon points="5 3 19 12 5 21 5 3"></polygon>`;
  }
}

/**
 * Drag and Drop Support (.vrm and .npy)
 */
function setupDragAndDrop() {
  const dropzone = document.getElementById('dropzone-box');
  const fileInput = document.getElementById('file-input-npy');

  dropzone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach((eventName) => {
    window.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    window.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  window.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleUploadedFiles(e.dataTransfer.files);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleUploadedFiles(e.target.files);
    }
  });
}

async function handleUploadedFiles(files) {
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const nameLower = file.name.toLowerCase();

    if (nameLower.endsWith('.vrm')) {
      console.log(`[Upload] Loading custom VRM: ${file.name}`);
      document.getElementById('model-status-text').textContent = `Đang nạp ${file.name}...`;
      try {
        await vrmEngine.loadModel(file);
        document.getElementById('model-status-text').textContent = `${file.name} (Đã sẵn sàng)`;
        document.getElementById('model-dot').style.backgroundColor = 'var(--accent-green)';
      } catch (err) {
        console.error('Failed to load dropped VRM:', err);
        document.getElementById('model-status-text').textContent = 'Lỗi nạp VRM';
        document.getElementById('model-dot').style.backgroundColor = 'var(--accent-red)';
      }
    } else if (nameLower.endsWith('.npy')) {
      console.log(`[Upload] Parsing custom .npy: ${file.name}`);
      try {
        const buffer = await file.arrayBuffer();
        const parsedMotion = parseNpyBuffer(buffer, file.name);
        motionCache.set(parsedMotion.word, parsedMotion);
        activeSentenceWords.push(parsedMotion.word);
        await updateSequencePlayback(activeSentenceWords, true);
        highlightActiveBadges();
      } catch (err) {
        console.error('Failed to parse .npy binary:', err);
        alert(`Lỗi đọc tệp .npy: ${err.message}`);
      }
    }
  }
}

/**
 * Update Timeline HUD & Telemetry
 */
function updateTimelineHUD() {
  const currentItem = sequenceBlender.getCurrentItem();
  const currentIdx = Math.floor(sequenceBlender.currentFrameIdx);
  const totalFrames = sequenceBlender.blendedFrames.length;

  // Active word HUD
  const activeWordEl = document.getElementById('hud-active-word');
  const bridgeBadgeEl = document.getElementById('hud-bridge-badge');

  if (currentItem) {
    if (currentItem.isBridge) {
      bridgeBadgeEl.style.display = 'inline-block';
      activeWordEl.textContent = currentItem.wordName;
    } else {
      bridgeBadgeEl.style.display = 'none';
      activeWordEl.textContent = currentItem.wordName;
    }
  }

  // Timeline Fill
  const percent = totalFrames > 0 ? (currentIdx / totalFrames) * 100 : 0;
  document.getElementById('timeline-fill').style.width = `${percent}%`;
  document.getElementById('time-current-frame').textContent = `Khung ${currentIdx}`;
  document.getElementById('time-total-frames').textContent = `${totalFrames}`;
  document.getElementById('time-percentage').textContent = `${Math.round(percent)}%`;

  // Telemetry
  const frame = sequenceBlender.getCurrentFrame();
  if (frame) {
    document.getElementById('telem-active-hand').textContent =
      frame.activeHandSide === 'both' ? 'Hai tay' :
      frame.activeHandSide === 'leftHand' ? 'Tay trái' :
      frame.activeHandSide === 'rightHand' ? 'Tay phải' : 'Không có';

    const ptCount = (frame.leftHand ? 21 : 0) + (frame.rightHand ? 21 : 0) + (frame.pose ? 8 : 0);
    document.getElementById('telem-joint-count').textContent = `${ptCount} điểm (3D)`;
  }
}

/**
 * Main Animation / Render Loop
 */
function animate() {
  requestAnimationFrame(animate);

  const delta = clock.getDelta();

  // Advance sequence playback
  sequenceBlender.advance(delta);

  // Get current frame data
  const currentFrame = sequenceBlender.getCurrentFrame();

  // Calibrate Cyan Skeleton anchor directly from VRM shoulders (Fix 1: Zero Z separation!)
  if (vrmEngine.isLoaded) {
    const shoulders = vrmEngine.getShoulderPositions();
    if (shoulders) {
      skeletonOverlay.setShoulderCalibration(shoulders.leftPos, shoulders.rightPos);
    }
  }

  // 1. Update Cyan 3D Skeleton (Untouched raw 3D depth)
  const skeletonWorldData = skeletonOverlay.update(currentFrame);

  // 2. Solve VRM Avatar from Cyan Skeleton (READ-ONLY)
  if (vrmEngine.isLoaded && skeletonWorldData) {
    vrmEngine.update(skeletonWorldData, delta);
  }

  // Update HUD
  updateTimelineHUD();

  // Update FPS Counter
  const now = performance.now();
  frameTimes.push(now);
  while (frameTimes.length > 0 && frameTimes[0] <= now - 1000) {
    frameTimes.shift();
  }
  if (now - lastFpsUpdate > 400) {
    document.getElementById('fps-counter').textContent = `${frameTimes.length} FPS`;
    lastFpsUpdate = now;
  }

  controls.update();
  renderer.render(scene, camera);
}

/**
 * Application Bootstrap
 */
async function bootstrap() {
  initScene();
  setupUIEventListeners();

  // 1. Load manifest.json
  try {
    const res = await fetch('/motions/manifest.json');
    const data = await res.json();
    manifestWords = data.words || [];
    renderWordBank('all');
  } catch (err) {
    console.error('Failed to load manifest.json:', err);
  }

  // 2. Load default VRM 0.0 Model (AvatarSample_A.vrm)
  try {
    document.getElementById('model-status-text').textContent = 'Đang tải AvatarSample_A.vrm...';
    await vrmEngine.loadModel('/models/AvatarSample_A.vrm');
    document.getElementById('model-status-text').textContent = 'AvatarSample_A.vrm (Đã sẵn sàng)';
    document.getElementById('model-dot').style.backgroundColor = 'var(--accent-green)';
    const shoulders = vrmEngine.getShoulderPositions();
    if (shoulders) {
      skeletonOverlay.setShoulderCalibration(shoulders.leftPos, shoulders.rightPos);
    }
  } catch (err) {
    console.warn('AvatarSample_A.vrm load fallback:', err);
    document.getElementById('model-status-text').textContent = 'Chế độ khung xương 3D';
  }

  // 3. Load initial word sequence
  await updateSequencePlayback(activeSentenceWords, false);
  highlightActiveBadges();

  // Highlight active chips on word change
  sequenceBlender.onWordChange = (wordIdx, wordName) => {
    document.querySelectorAll('.sentence-chip').forEach((chip) => {
      chip.classList.toggle('active-chip', parseInt(chip.dataset.idx, 10) === wordIdx);
    });
  };

  // Start Animation Loop
  animate();
}

bootstrap().catch(console.error);
